"""ffmpeg 转码与音频工具：16kHz 单声道 + 温和响度归一化 + Opus 瘦身 + 切片。"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# 16kHz 单声道 PCM16；loudnorm 单遍温和响度归一化（避免 32-bit float 低电平漏检）
NORMALIZE_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
OPUS_BITRATE_KBPS = 32  # mono ≈ 14-15MB/小时，符合规格书 15~24MB/小时目标


class FfmpegError(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FfmpegError("未找到 ffmpeg，请安装（winget install Gyan.FFmpeg）")
    return path


def ffprobe_bin() -> str:
    """独立定位 ffprobe（不能对 ffmpeg 路径做字符串替换——winget 目录名也含 ffmpeg）。"""
    direct = shutil.which("ffprobe")
    if direct:
        return direct
    f = Path(ffmpeg_bin())
    cand = f.with_name(f.name.lower().replace("ffmpeg", "ffprobe"))
    if cand.exists():
        return str(cand)
    raise FfmpegError("未找到 ffprobe")


def _run(args: list[str], timeout: int = 1800) -> None:
    proc = subprocess.run(
        args, capture_output=True, text=True, timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0:
        raise FfmpegError(f"{args[0]} 失败（exit {proc.returncode}）: {proc.stderr[-500:]}")


def to_16k_mono(src: str | Path, dst: str | Path, normalize: bool = True) -> Path:
    """原始 48kHz/32-bit float WAV → 16kHz 单声道 PCM16（含温和响度归一化）。

    注意：若源本身已是 Opus/压缩件（重转写场景），**跳过 loudnorm**——
    loudnorm 单遍处理超长压缩音频会异常退出（实测 exit 4294967294），
    且压缩件本已是归一化后的产物，二次归一化既无必要又拖慢 20 倍。
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    already_compressed = src.suffix.lower() in {".opus", ".m4a", ".mp3", ".aac", ".ogg"}
    filters = NORMALIZE_FILTER if (normalize and not already_compressed) else "anull"
    try:
        _run([ffmpeg_bin(), "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
              "-af", filters, "-c:a", "pcm_s16le", str(dst)])
    except FfmpegError:
        if not filters or filters == "anull":
            raise
        # 极短/异常文件 loudnorm 可能失败：去滤波重试（反偏见：宁可无损也不丢音频）
        log.warning("loudnorm 失败，改用无归一化重试：%s", src)
        _run([ffmpeg_bin(), "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
              "-c:a", "pcm_s16le", str(dst)])
    return dst


def wav_to_opus(src: str | Path, dst: str | Path, bitrate_kbps: int = OPUS_BITRATE_KBPS) -> Path:
    """原始 WAV → 高质量 Opus（语音/音乐兼顾，约 14-15MB/小时）。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg_bin(), "-y", "-i", str(src), "-c:a", "libopus",
          "-b:a", f"{bitrate_kbps}k", "-ac", "1", "-ar", "48000", str(dst)])
    return dst


def slice_wav(src: str | Path, dst: str | Path, start_s: float, end_s: float) -> Path:
    """从任意音频截取 [start_s, end_s] → 16kHz 单声道 WAV（识曲/试听用）。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.1, end_s - start_s)
    _run([ffmpeg_bin(), "-y", "-ss", f"{max(0, start_s):.3f}", "-t", f"{dur:.3f}",
          "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)])
    return dst


def probe_duration(path: str | Path) -> float:
    """ffprobe 秒数；失败返回 0.0。"""
    try:
        out = subprocess.run(
            [ffprobe_bin(), "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        info = json.loads(out.stdout or "{}")
        return float(info.get("format", {}).get("duration") or 0.0)
    except Exception:
        return 0.0


def read_pcm16(path: str | Path) -> tuple["bytearray", int]:
    """读取 16kHz 单声道 WAV → (PCM 字节, 采样率)。"""
    import wave

    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        assert w.getnchannels() == 1, "必须是单声道 16kHz WAV"
        data = bytearray(w.readframes(w.getnframes()))
    return data, rate
