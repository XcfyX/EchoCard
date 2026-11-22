"""听歌识曲：Provider 抽象 + ShazamIO 实现（代理 / 限流重试 / 结果落库缓存）。

- 音乐卡片一键识曲：传入 music_track id；
- 按时段查歌：任意起止挂钟时间（如 15:15-15:30）→ 从当日归档音频截取
  高音乐能量窗口（10~15 秒，失败滑动重采）送 Shazam；
- Provider 抽象预留 ACRCloud / AudD 扩展。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ..core.config import AppConfig
from ..core.db import Database
from ..core.utils import now_iso
from .transcode import ffmpeg_bin, probe_duration, slice_wav

log = logging.getLogger(__name__)

SLICE_S = 12.0          # 送识曲的窗口长度
ATTEMPTS = 3            # 单飞重试次数（滑动窗口重采）
BACKOFF_S = 2.0
QUERY_TIMEOUT_S = 25


@dataclass
class TrackInfo:
    ok: bool
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    cover_url: str | None = None
    match_ts: float | None = None
    provider: str = "shazam"
    reason: str = ""


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-300:])


class MusicProvider:
    """统一 Provider 接口（未来 ACRCloud/AudD 实现同样的 recognize）。"""

    name = "base"

    async def recognize(self, wav_path: str) -> TrackInfo:  # pragma: no cover
        raise NotImplementedError


class ShazamProvider(MusicProvider):
    """ShazamIO：免 Key 异步识曲；代理通过 trust_env + 环境变量注入。"""

    name = "shazam"

    def __init__(self, proxy_url: str | None):
        self.proxy_url = proxy_url

    async def recognize(self, wav_path: str) -> TrackInfo:
        os.environ.setdefault("HTTP_PROXY", self.proxy_url or "")
        os.environ.setdefault("HTTPS_PROXY", self.proxy_url or "")
        os.environ.setdefault("http_proxy", self.proxy_url or "")
        os.environ.setdefault("https_proxy", self.proxy_url or "")

        import aiohttp
        import shazamio

        session = aiohttp.ClientSession(trust_env=True)
        try:
            shz = shazamio.Shazam()
            out = await asyncio.wait_for(
                shz.recognize(wav_path), timeout=QUERY_TIMEOUT_S
            )
        finally:
            await session.close()

        track = out.get("track")
        if not track:
            return TrackInfo(ok=False, reason=out.get("matches") and "no_exact_match" or "not_found")
        meta = (track.get("sections") or [{}])[0].get("metadata") or []
        artist = album = None
        cover = (track.get("images") or {}).get("coverart")
        for m in meta:
            if m.get("title") == "Artist":
                artist = m.get("text")
            elif m.get("title") == "Album":
                album = m.get("text")
        return TrackInfo(ok=True, title=track.get("title"), artist=artist,
                         album=album, cover_url=cover,
                         match_ts=track.get("matchposition"))


def get_provider(cfg: AppConfig) -> MusicProvider:
    return ShazamProvider(cfg.proxy_url)


def _pick_best_window(src: str, start_s: float, end_s: float,
                      tmpdir: str, engine_tagger=None) -> Path | None:
    """在 [start_s, end_s] 内选一个高音乐能量窗口（有 YAMNet 用 YAMNet，否则取中段）。"""
    span = max(0.0, end_s - start_s)
    if span < 3:
        return None
    win = min(SLICE_S, span)
    center = start_s + span / 2
    if engine_tagger is not None:
        try:
            from .engines import pcm_to_float
            from .transcode import read_pcm16

            full = Path(tmpdir) / "scan.wav"
            slice_wav(src, full, start_s, end_s)
            pcm, rate = read_pcm16(full)
            samples = pcm_to_float(pcm)
            best_center, best_score = None, 0.0
            window = int(4.0 * rate)
            step = max(1, int(2.0 * rate))
            for s0 in range(0, max(1, len(samples) - window + 1), step):
                chunk = samples[s0:s0 + window]
                score = engine_tagger.music_score(chunk)
                if score > best_score:
                    best_center, best_score = start_s + (s0 + window / 2) / rate, score
            if best_center is not None and best_score >= 0.20:
                center = min(max(best_center, start_s + win / 2), start_s + span - win / 2)
        except Exception:
            log.warning("YAMNet 窗口选择失败，回退中段采样", exc_info=True)
    out = Path(tmpdir) / "query.wav"
    slice_wav(src, out, center - win / 2, center + win / 2)
    return out


def recognize_time_range(db: Database, cfg: AppConfig, day: str,
                         start_ts: str, end_ts: str) -> dict:
    """按时段查歌：day + 挂钟起止 HH:MM:SS（或含日期）→ Shazam → music_tracks 落库。"""
    from datetime import datetime

    def _parse(t: str) -> float:
        t = t.strip()
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                d = datetime.strptime(t, fmt)
                return d.hour * 3600 + d.minute * 60 + d.second
            except ValueError:
                continue
        raise ValueError(f"无法解析时间：{t}（示例 15:15 或 15:15:30）")

    s0, s1 = _parse(start_ts), _parse(end_ts)
    if s1 <= s0:
        s0, s1 = s1, s0

    # 当日音频来源：优先原始 WAV（含未压缩日），否则用 Opus 压缩件
    rows = db.query(
        "SELECT raw_path, compressed_path, recorded_ts FROM ingest_files"
        " WHERE recorded_at=? AND status IN ('imported','done','processing')", (day,)
    )
    if not rows:
        return {"ok": False, "reason": f"{day} 无已归档音频"}

    engines = None
    try:
        from .engines import get_engines

        engines = get_engines(cfg)
        tagger = engines.tagger()
    except Exception:
        tagger = None

    with tempfile.TemporaryDirectory(prefix="echocard-music-") as tmpdir:
        for row in rows:
            src = row["raw_path"] or row["compressed_path"]
            if not src or not Path(src).exists():
                continue
            base_s = 0.0
            if row["recorded_ts"]:
                try:
                    t0 = datetime.strptime(row["recorded_ts"], "%Y-%m-%d %H:%M:%S")
                    base_s = t0.hour * 3600 + t0.minute * 60 + t0.second
                except ValueError:
                    pass
            rel0, rel1 = s0 - base_s, s1 - base_s
            dur = probe_duration(src)
            rel1 = min(rel1, dur)
            rel0 = max(0.0, rel0)
            if rel1 - rel0 < 3:
                continue  # 该文件与所选时段无重叠，尝试下一个文件
            for attempt in range(ATTEMPTS):
                offset = random.uniform(0, max(0.0, (rel1 - rel0) - SLICE_S)) if attempt else 0
                a0 = rel0 + offset if attempt else (rel0 + rel1) / 2 - SLICE_S / 2
                a0 = max(0.0, min(a0, max(0.0, rel1 - 3)))
                a1 = min(rel1, a0 + SLICE_S)
                try:
                    wav = _pick_best_window(src, a0, a1, tmpdir, tagger)
                    if wav is None:
                        break
                    info = asyncio.run(get_provider(cfg).recognize(str(wav)))
                except (asyncio.TimeoutError, TimeoutError):
                    info = TrackInfo(ok=False, reason="timeout")
                except Exception as e:
                    info = TrackInfo(ok=False, reason=str(e)[:200])
                if info.ok:
                    cur = db.execute(
                        "INSERT INTO music_tracks(day, start_ts, end_ts, audio_path, status,"
                        " title, artist, album, cover_url, provider, match_ts, recognized_at)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (day, start_ts, end_ts, src, "recognized", info.title, info.artist,
                         info.album, info.cover_url, info.provider, info.match_ts, now_iso()),
                    )
                    return {"ok": True, "track_id": cur.lastrowid, "title": info.title,
                            "artist": info.artist, "album": info.album,
                            "cover_url": info.cover_url}
                time.sleep(BACKOFF_S * (attempt + 1))
        cur = db.execute(
            "INSERT INTO music_tracks(day, start_ts, end_ts, status, reason, recognized_at)"
            " VALUES(?,?,?,?,?,?)",
            (day, start_ts, end_ts, "failed", "shazam_no_match_or_network",
             now_iso()),
        )
        return {"ok": False, "track_id": cur.lastrowid,
                "reason": "识曲失败（网络/代理/无匹配），已记录可重试"}
