"""音乐片段检测与合并：SenseVoice 事件标签 + YAMNet 非语音区间兜底 → 音乐节点。

合并规则：起点/终点相近（重叠或间隔 <= GAP_MERGE_S 秒）的候选合并为一个
带精确挂钟起止时间的音乐卡片，写入 music_tracks（status=detected）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .vad_asr import MUSIC_EVENT_TOKENS, Segment

log = logging.getLogger(__name__)

GAP_MERGE_S = 6.0
MIN_TRACK_S = 3.0


def merge_music_candidates(candidates: list[Segment]) -> list[tuple[float, float, list[str]]]:
    """合并相近的音乐候选 → [(start_s, end_s, 来源标签), ...] 按起点排序。"""
    if not candidates:
        return []
    spans = sorted(((c.start_s, c.end_s) for c in candidates), key=lambda x: x[0])
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= GAP_MERGE_S:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e, ["music"]) for s, e in merged if e - s >= MIN_TRACK_S]


def wall_clock_tracks(file_start: datetime, tracks: list[tuple[float, float, list[str]]]):
    """[(开始挂钟时间, 结束挂钟时间, 标签), ...]"""
    return [
        (file_start + timedelta(seconds=s), file_start + timedelta(seconds=e), tags)
        for s, e, tags in tracks
    ]


def music_tag_for_segment(seg: Segment) -> str | None:
    """该片段是否命中音乐事件标签。"""
    return next((t for t in seg.events if t in MUSIC_EVENT_TOKENS), None)
