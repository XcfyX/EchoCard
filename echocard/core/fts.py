"""FTS5 全文检索：中文按双字 bigram 分词（unicode61 分词器下的通用 CJK 方案）。

写入时把文本转为空格分隔的双字序列，检索时对查询词做同样处理并按短语匹配，
任意 ≥2 字的中文子串均可命中；英文/数字按原词保留。ref 列（UNINDEXED）存放
指向具体记录的标识（如 seg:12 / music:3 / todo:5）。
"""
from __future__ import annotations

import re
import sqlite3

from .db import Database

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff\u3040-\u30ff]|\S")
_CJK = r"[\u4e00-\u9fff\u3040-\u30ff]"


def _bigrams(chars: list[str]) -> list[str]:
    if len(chars) == 1:
        return [chars[0]]
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def cjk_index_text(text: str) -> str:
    """索引文本 → bigram 序列（CJK 相邻双字组合，英文/数字整词保留）。"""
    if not text:
        return ""
    out: list[str] = []
    cjk_run: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        if re.match(_CJK, tok):
            cjk_run.append(tok)
        else:
            if cjk_run:
                out.extend(_bigrams(cjk_run))
                cjk_run = []
            out.append(tok)
    if cjk_run:
        out.extend(_bigrams(cjk_run))
    return " ".join(out)


def cjk_query(text: str) -> str:
    """查询词 → FTS5 短语匹配表达式。"""
    indexed = cjk_index_text(text)
    return '"' + indexed.replace('"', '""') + '"'


def fts_insert(db: Database, kind: str, day_range: str, text: str, ref: str = "") -> None:
    db.execute(
        "INSERT INTO fts_memory(kind, day_range, text, ref) VALUES(?,?,?,?)",
        (kind, day_range, cjk_index_text(text), ref),
    )


def rebuild_fts(db: Database) -> int:
    """全量重建索引（启动时调用，保证旧数据同样可检索）。"""
    db.execute("DELETE FROM fts_memory")
    n = 0
    for r in db.query("SELECT day_range, content_md, tags, facts FROM daily_summaries"):
        fts_insert(db, "summary", r["day_range"],
                   f"{r['content_md']} {r['tags'] or ''} {r['facts'] or ''}")
        n += 1
    for r in db.query("SELECT id, day, text FROM audio_segments WHERE text IS NOT NULL"):
        fts_insert(db, "segment", r["day"], r["text"], f"seg:{r['id']}")
        n += 1
    for r in db.query("SELECT id, day, title, artist, album FROM music_tracks"):
        if r["title"] or r["artist"]:
            fts_insert(db, "music", r["day"],
                       f"{r['title'] or ''} {r['artist'] or ''} {r['album'] or ''}",
                       f"music:{r['id']}")
            n += 1
    for r in db.query(
        "SELECT t.id, s.day_range AS day, t.content FROM todos t"
        " JOIN daily_summaries s ON s.id = t.summary_id"
    ):
        fts_insert(db, "todo", r["day"], r["content"], f"todo:{r['id']}")
        n += 1
    return n


def search(db: Database, query: str, limit: int = 30) -> list[dict]:
    """全局搜索：返回 [{kind, day_range, snippet, ref}]。"""
    q = cjk_query(query)
    try:
        rows = db.query(
            "SELECT kind, day_range, ref,"
            " snippet(fts_memory, 2, '<b>', '</b>', '…', 12) AS snip"
            " FROM fts_memory WHERE fts_memory MATCH ? ORDER BY rank LIMIT ?",
            (q, limit),
        )
    except sqlite3.OperationalError:
        return []
    return [{"kind": r["kind"], "day_range": r["day_range"],
             "snippet": r["snip"] or "", "ref": r["ref"] or ""} for r in rows]
