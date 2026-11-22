"""归档后磁盘清理：AI 总结已回流的日期，可清理未 ⭐ 星标的压缩音频。

永久保留：星标原声切片 + 完整清洗文字稿 + 音乐记录 + 压缩照片 + AI 最终总结。
清理对象仅限：该日「未被星标片段所属文件」的 Opus 压缩音频（可单独保留未识别
的音乐切片来源文件）。提供配置开关（自动）与手动按钮两种触发方式。
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..core.config import AppConfig
from ..core.db import Database

log = logging.getLogger(__name__)


def cleanup_day(db: Database, cfg: AppConfig, day: str,
                keep_unrecognized_music: bool = True) -> dict:
    """清理某日未星标文件的 Opus 压缩音频。返回 {removed, freed_bytes, kept}。"""
    removed, freed = [], 0
    starred_files = {
        r["file_id"] for r in db.query(
            "SELECT DISTINCT file_id FROM audio_segments WHERE day=? AND starred=1"
            " AND file_id IS NOT NULL", (day,)
        )
    }
    rows = db.query(
        "SELECT id, compressed_path, raw_path FROM ingest_files WHERE recorded_at=?",
        (day,),
    )
    for row in rows:
        if row["id"] in starred_files:
            continue
        path = row["compressed_path"]
        if not path:
            continue
        # 该文件是否被未识别的音乐节点引用（可选保留）
        if keep_unrecognized_music:
            used_by_music = db.query_one(
                "SELECT 1 FROM music_tracks WHERE file_id=? AND status != 'recognized'"
                " LIMIT 1", (row["id"],)
            )
            if used_by_music:
                continue
        p = Path(path)
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
            removed.append(p.name)
        db.execute("UPDATE ingest_files SET compressed_path=NULL WHERE id=?", (row["id"],))
    log.info("清理 %s：删除 %d 个压缩音频，释放 %.1fMB", day, len(removed), freed / 1e6)
    return {"day": day, "removed": removed, "freed_bytes": freed, "kept": len(rows) - len(removed)}


def auto_cleanup_if_enabled(db: Database, cfg: AppConfig, day: str) -> dict | None:
    """AI 总结回流后按配置开关自动清理（cleanup_after_summary=True 时）。"""
    if not cfg.cleanup_after_summary:
        return None
    summarized = db.query_one(
        "SELECT 1 FROM daily_summaries WHERE day_range=?", (day,)
    )
    if not summarized:
        return None
    return cleanup_day(db, cfg, day)
