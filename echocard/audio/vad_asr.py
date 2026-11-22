"""VAD 切分 + SenseVoice 反偏见转写。

反偏见红线：
- 绝不删除任何语音片段；只附加客观特征元数据（audio_tags: bgm/music/low_volume/…）；
- 禁止主观定性标签（“废话/无关”等绝不出现）；
- 事件/情绪标签（SenseVoice 富文本 token）全部保留为客观记录。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .engines import Engines, pcm_to_float, rms_db
from .transcode import read_pcm16

log = logging.getLogger(__name__)

# SenseVoice 富文本 token：语言 <|zh|>、情绪 <|NEUTRAL|>、事件 <|Music|> 等
_TOKEN_RE = re.compile(r"<\|([A-Za-z_]+)\|>")
# 视为音乐/背景音乐类事件（客观标签）
MUSIC_EVENT_TOKENS = {"Music", "BGM", "MUSIC", "音乐", "背景音乐"}
OTHER_EVENT_TOKENS = {"Applause", "Laughter", "Cough", "Sneeze", "Speech", "Withnoise"}
LOW_VOLUME_DB = -34.0  # 低于该 RMS 视为低音量（客观特征）


@dataclass
class Segment:
    """一个语音/音频片段（挂钟时间 + 文本 + 客观标签）。"""

    start_s: float
    end_s: float
    text: str = ""
    language: str | None = None
    emotion: str | None = None
    events: list[str] = field(default_factory=list)
    audio_tags: list[str] = field(default_factory=list)
    rms_db: float = -120.0
    file_id: int | None = None
    # 声纹（聚类阶段填充）
    embedding: np.ndarray | None = None
    cluster_id: int | None = None
    speaker_name: str | None = None
    speaker_match_score: float | None = None

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s

    def wall_clock(self, file_start: datetime) -> tuple[datetime, datetime]:
        return (file_start + timedelta(seconds=self.start_s),
                file_start + timedelta(seconds=self.end_s))


def _parse_rich_text(raw: str, rms: float) -> tuple[str, str | None, list[str]]:
    """拆解 SenseVoice 富文本 → (纯文本, 情绪, 事件列表)。语言 token 单独处理。"""
    events: list[str] = []
    emotion = None
    language = None

    def _grab(m: re.Match) -> str:
        nonlocal emotion, language
        tok = m.group(1)
        if tok in MUSIC_EVENT_TOKENS or tok in OTHER_EVENT_TOKENS:
            if tok not in events:
                events.append(tok)
        elif tok in {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "EMO_UNKNOWN", "SIGH"}:
            emotion = tok
        elif tok in {"zh", "en", "ja", "ko", "yue"}:
            language = tok
        return ""

    text = _TOKEN_RE.sub(_grab, raw).strip()
    tags = list(events)
    if rms < LOW_VOLUME_DB:
        tags.append("low_volume")
    return text, emotion, tags


def _vad_segments(samples: np.ndarray, engines: Engines) -> list[tuple[int, int]]:
    """Silero VAD → [(start_sample, end_sample), ...]。

    注意：必须按 window_size 分块喂数据（整段一次性喂入会导致部分实现检测失败）。
    """
    import sherpa_onnx

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(engines.vad_model_path)
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.35
    cfg.silero_vad.min_speech_duration = 0.30
    cfg.sample_rate = 16000
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
    window = max(1, int(cfg.silero_vad.window_size))
    for i in range(0, len(samples), window):
        vad.accept_waveform(samples[i:i + window])
    try:
        vad.flush()
    except Exception:
        pass
    out: list[tuple[int, int]] = []
    while not vad.empty():
        seg = vad.front
        start = int(getattr(seg, "start", 0))
        s = np.asarray(seg.samples, dtype=np.float32)
        out.append((start, start + len(s)))
        vad.pop()
    return out


def _non_speech_gaps(total: int, segs: list[tuple[int, int]], rate: int,
                     min_gap_s: float = 5.0) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    prev = 0
    for s, e in segs:
        if s - prev >= int(min_gap_s * rate):
            gaps.append((prev, s))
        prev = max(prev, e)
    if total - prev >= int(min_gap_s * rate):
        gaps.append((prev, total))
    return gaps


def transcribe_file(engines: Engines, wav16k: str, file_start: datetime,
                    detect_music_gaps: bool = True) -> tuple[list[Segment], list[Segment]]:
    """转写一个 16kHz WAV。

    返回 (语音片段列表, 音乐候选片段列表)。
    音乐候选 = ① 语音片段上 SenseVoice 事件标签含音乐 ② 非语音区间 YAMNet 兜底。
    """
    pcm, rate = read_pcm16(wav16k)
    samples = pcm_to_float(pcm)
    total = len(samples)

    rec = engines.asr()
    speech: list[Segment] = []
    for s0, s1 in _vad_segments(samples, engines):
        seg_samples = samples[s0:s1]
        if len(seg_samples) < int(0.25 * rate):
            continue
        level = rms_db(seg_samples)
        stream = rec.create_stream()
        stream.accept_waveform(rate, seg_samples)
        rec.decode_stream(stream)
        raw_text = stream.result.text or ""
        text, emotion, tags = _parse_rich_text(raw_text, level)
        seg = Segment(
            start_s=s0 / rate, end_s=s1 / rate, text=text,
            language=None, emotion=emotion,
            events=[t for t in tags if t not in {"low_volume"}],
            audio_tags=tags, rms_db=level,
        )
        # 语言 token 兜底提取
        m = _TOKEN_RE.search(raw_text)
        if m and m.group(1) in {"zh", "en", "ja", "ko", "yue"}:
            seg.language = m.group(1)
        speech.append(seg)

    music: list[Segment] = [s for s in speech if any(t in MUSIC_EVENT_TOKENS for t in s.events)]

    if detect_music_gaps:
        tagger = engines.tagger()
        if tagger is not None:
            for g0, g1 in _non_speech_gaps(total, [(int(s.start_s * rate), int(s.end_s * rate)) for s in speech], rate):
                gap = samples[g0:g1]
                window = int(2.0 * rate)
                for w0 in range(0, len(gap) - int(0.5 * rate), window):
                    chunk = gap[w0:w0 + window]
                    if len(chunk) < int(0.5 * rate):
                        break
                    if tagger.music_score(chunk) >= 0.30:
                        music.append(Segment(
                            start_s=(g0 + w0) / rate, end_s=(g0 + len(chunk)) / rate,
                            events=["Music"], audio_tags=["music", "instrumental_or_bgm"],
                            rms_db=rms_db(chunk),
                        ))
    log.info("转写完成 %s：语音 %d 段，音乐候选 %d 个",
             wav16k, len(speech), len(music))
    return speech, music
