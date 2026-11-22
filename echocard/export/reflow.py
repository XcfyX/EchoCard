"""AI 总结三通道回流：拖入 .zip / 拖入 .md / 直接粘贴 Markdown。

容错解析 YAML Frontmatter（AI 偶发漏写/错写也能安全入库）：
- 严格 YAML 优先；失败退化为逐行 key: value + 缩进列表解析；
- day 缺省取今天；日期格式自动纠正（YYYY-MM-DD / YYYY-MM-DD_to_YYYY-MM-DD）。
回流内容写入 daily_summaries + todos，并写入 FTS5 全文索引。
"""
from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from datetime import date, datetime
from pathlib import Path

import yaml

from ..core.config import AppConfig
from ..core.db import Database
from ..core.fts import fts_insert
from ..core.utils import now_iso, today_str

log = logging.getLogger(__name__)

_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}(_to_\d{4}-\d{2}-\d{2})?$")


def parse_frontmatter(md_text: str) -> tuple[dict, str]:
    """返回 (meta dict, 正文 body)。容错：无 frontmatter / 坏 YAML 均安全。"""
    text = md_text.strip()
    meta: dict = {}
    body = text
    m = re.match(r"^```?ya?ml\s*\n(.*?)\n```?\s*\n?(.*)$", text, re.S) \
        or re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if m:
        raw_yaml, body = m.group(1), m.group(2).strip()
        try:
            parsed = yaml.safe_load(raw_yaml)
            meta = parsed if isinstance(parsed, dict) else {}
        except yaml.YAMLError:
            meta = _naive_parse(raw_yaml)
        meta["_raw"] = raw_yaml
    else:
        body = text
    return meta, body


def _naive_parse(raw: str) -> dict:
    """坏 YAML 时的逐行兜底解析：支持 key: value 与 '- item' 列表。"""
    out: dict = {}
    current_list: list | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        s = line.strip()
        if s.startswith("- "):
            if current_list is not None:
                current_list.append(s[2:].strip())
            continue
        m = re.match(r"^([\w_]+)\s*:\s*(.*)$", s)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if val.startswith("[") and val.endswith("]"):
                items = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
                out[key] = items
                current_list = out[key] if all(isinstance(i, str) for i in items) else None
            elif val == "":
                out[key] = []
                current_list = out[key]
            else:
                out[key] = val.strip("'\"")
                current_list = None
    return out


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def _normalize_day(value) -> str:
    if value:
        m = _DAY_RE.search(str(value))
        if m:
            return m.group(0)
    return today_str()


def import_summary(db: Database, cfg: AppConfig, md_text: str,
                   source: str = "reflow") -> dict:
    """回流入库：daily_summaries + todos + FTS。返回摘要信息。"""
    meta, body = parse_frontmatter(md_text)
    day = _normalize_day(meta.get("date") or meta.get("day") or meta.get("日期"))
    tags = _as_list(meta.get("tags") or meta.get("标签"))
    todos = _as_list(meta.get("todos") or meta.get("待办"))
    facts = _as_list(meta.get("facts") or meta.get("事实"))
    music = _as_list(meta.get("music") or meta.get("歌曲") or meta.get("音乐"))
    mood = str(meta.get("mood") or meta.get("情绪") or "")

    cur = db.execute(
        "INSERT INTO daily_summaries(day_range, content_md, frontmatter, tags,"
        " todos_json, facts, music, source, imported_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (day, body, meta.get("_raw"), json.dumps(tags, ensure_ascii=False),
         json.dumps(todos, ensure_ascii=False), json.dumps(facts, ensure_ascii=False),
         json.dumps(music, ensure_ascii=False), source, now_iso()),
    )
    summary_id = cur.lastrowid
    for t in todos:
        db.execute(
            "INSERT INTO todos(summary_id, content, done, created_at) VALUES(?,?,0,?)",
            (summary_id, t, now_iso()),
        )
    fts_insert(db, "summary", day, f"{body} {' '.join(tags)} {' '.join(facts)}")
    log.info("AI 总结已回流：%s（todos=%d, tags=%d, source=%s）", day, len(todos), len(tags), source)
    return {"ok": True, "summary_id": summary_id, "day": day,
            "todos": len(todos), "tags": tags, "facts": facts, "music": music,
            "mood": mood}


def import_from_zip(db: Database, cfg: AppConfig, zip_bytes: bytes) -> dict:
    """通道 1：AI 生成的 *_ai_summary.zip（内含 summary.md / meta.json）。"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        md_name = next((n for n in names if n.lower().endswith(".md")), None)
        if not md_name:
            raise ValueError("zip 内未找到 .md 总结文件")
        md_text = zf.read(md_name).decode("utf-8", errors="replace")
        meta_json = next((n for n in names if n.endswith("meta.json")), None)
    result = import_summary(db, cfg, md_text, source="reflow_zip")
    return result


def import_from_md(db: Database, cfg: AppConfig, md_text: str) -> dict:
    """通道 2：直接拖入 .md 文件。"""
    return import_summary(db, cfg, md_text, source="reflow_md")
