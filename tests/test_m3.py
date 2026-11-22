"""M3 测试：投喂包导出 / 三通道回流 / FTS5 中文检索 / 星标清理 / LLM 客户端。"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from echocard.core.db import Database
from echocard.core.fts import rebuild_fts, search
from echocard.desktop.server import create_app
from echocard.export.cleanup import cleanup_day
from echocard.export.feedpack import build_markdown, export_days
from echocard.export.reflow import import_summary, parse_frontmatter
from echocard.core.events import EventBus
from echocard.ingest.dji_import import DeviceImporter
from conftest import make_jpeg, make_wav

SAMPLE_MD = """---
date: 2026-11-22
tags: [复盘, 会议, 咖啡]
mood: 平稳偏积极
todos:
  - 把季度报告发给张三
  - 预约周六的羽毛球场地
facts:
  - 妈妈的生日是 3 月 8 日
music:
  - 晴天 - 周杰伦
---

# 今日复盘

上午和团队开了规划会，下午在咖啡馆改方案……（正文自由深度）
"""


def _insert_audio_and_photo(db, cfg, tmp_path):
    db.execute(
        "INSERT INTO devices(kind, serial, bound_at) VALUES('dji_mic_tx','TEST','t')")
    dev = db.query_one("SELECT id FROM devices")["id"]
    raw = make_wav(tmp_path / "DJI_01_20261122_100000.wav", seconds=2)
    db.execute(
        "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, recorded_ts, raw_path, compressed_path, status, imported_at)"
        " VALUES(?,'usb',?,'sha1',?,?,?,?,?,'imported','t')",
        (dev, str(raw), raw.stat().st_size, "2026-11-22", "2026-11-22 10:00:00",
         str(raw), str(raw), ),
    )
    fid = db.query_one("SELECT id FROM ingest_files")["id"]
    db.execute(
        "INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
        " end_offset, text, cluster_id, speaker_name, audio_tags, starred)"
        " VALUES(?, '2026-11-22','10:00:00','10:01:00',0,60,?,1,?,?,1)",
        (fid, "今天下午三点开会讨论季度目标", "说话人 1", '["speech"]'),
    )
    db.execute(
        "INSERT INTO music_tracks(file_id, day, start_ts, end_ts, status, title, artist)"
        " VALUES(?, '2026-11-22','15:18:10','15:21:05','recognized','晴天','周杰伦')",
        (fid,))
    img = tmp_path / "10-05-00_01.jpg"
    img.write_bytes(make_jpeg((400, 300)))
    db.execute(
        "INSERT INTO photos(device_id, source, src_sha256, captured_at, captured_ts,"
        " original_path, compressed_path, imported_at)"
        " VALUES(?, 'lan', 'ph1', '2026-11-22', '2026-11-22 10:05:00', ?, ?, 't')",
        (dev, str(img), str(img)))


# ---------------------------------------------------------------- 导出
def test_export_single_day(db, cfg, tmp_path):
    _insert_audio_and_photo(db, cfg, tmp_path)
    result = export_days(db, cfg, ["2026-11-22"])
    folder = Path(result["folder"])
    assert result["photos"] == 1
    assert (folder / "投喂AI.md").exists()
    assert (folder / "photos").is_dir() and len(list((folder / "photos").iterdir())) == 1
    assert Path(result["zip"]).exists()
    with zipfile.ZipFile(result["zip"]) as zf:
        names = zf.namelist()
        assert any(n.endswith("投喂AI.md") for n in names)
        assert any("/photos/" in n for n in names)
    md = (folder / "投喂AI.md").read_text(encoding="utf-8")
    assert "反偏见纠错声明" in md, "内置 Prompt 必须包含反偏见声明"
    assert "说话人 1" in md and "晴天" in md
    assert "base64" not in md.lower(), "禁止在 Markdown 嵌入 Base64 图片"
    assert "photos/10-05-00_01.jpg" in md


def test_export_multi_day_naming(db, cfg, tmp_path):
    _insert_audio_and_photo(db, cfg, tmp_path)
    result = export_days(db, cfg, ["2026-11-22", "2026-11-23"])
    assert "2026-11-22_to_2026-11-23" in Path(result["zip"]).name


def test_prompt_editable(db, cfg, tmp_path):
    cfg.export_prompt = "自定义提示词 {days} / {date_value}"
    md, _ = build_markdown(db, cfg, ["2026-11-22"])
    assert md.startswith("自定义提示词 2026-11-22 / 2026-11-22")


# ---------------------------------------------------------------- 回流
def test_parse_frontmatter_yaml_and_naive():
    meta, body = parse_frontmatter(SAMPLE_MD)
    assert meta["tags"] == ["复盘", "会议", "咖啡"]
    assert meta["todos"][0] == "把季度报告发给张三"
    assert "今日复盘" in body
    # 坏 YAML 兜底
    bad = "---\ndate: 2026-11-22\ntags: [a, b]\ntodos:\n  - x\n  - y\n---\n正文"
    meta2, body2 = parse_frontmatter(bad)
    assert meta2.get("tags") == ["a", "b"]
    assert "正文" in body2


def test_reflow_paste_creates_todos_and_fts(db, cfg):
    result = import_summary(db, cfg, SAMPLE_MD, source="reflow_paste")
    assert result["ok"] and result["day"] == "2026-11-22"
    assert result["todos"] == 2
    todos = db.query("SELECT * FROM todos ORDER BY id")
    assert len(todos) == 2 and todos[0]["done"] == 0
    row = db.query_one("SELECT * FROM daily_summaries")
    assert json.loads(row["tags"]) == ["复盘", "会议", "咖啡"]


def test_fts_chinese_search(db, cfg):
    import_summary(db, cfg, SAMPLE_MD)
    rebuild_fts(db)
    hits = search(db, "季度报告")
    assert any(h["kind"] == "todo" for h in hits), "待办可被全文检索"
    hits2 = search(db, "咖啡馆")
    assert any(h["kind"] == "summary" for h in hits2)
    assert search(db, "不存在的词组xyz") == []


# ---------------------------------------------------------------- 清理
def test_cleanup_keeps_starred_removes_unstarred(db, cfg, tmp_path):
    _insert_audio_and_photo(db, cfg, tmp_path)
    opus = tmp_path / "starred.opus"
    opus.write_bytes(b"x" * 100)
    fid = db.query_one("SELECT id FROM ingest_files")["id"]
    db.execute("UPDATE ingest_files SET compressed_path=? WHERE id=?", (str(opus), fid))
    # 星标片段属于该文件 → 保留
    result = cleanup_day(db, cfg, "2026-11-22")
    assert opus.exists() and result["removed"] == []
    # 无星标 → 清理
    db.execute("UPDATE audio_segments SET starred=0 WHERE day='2026-11-22'")
    result2 = cleanup_day(db, cfg, "2026-11-22")
    assert not opus.exists() and len(result2["removed"]) == 1


# ---------------------------------------------------------------- API
def test_api_export_reflow_search(db, cfg, tmp_path):
    _insert_audio_and_photo(db, cfg, tmp_path)
    app = create_app(db, cfg, EventBus(), DeviceImporter(db, cfg))
    client = TestClient(app)

    r = client.post("/api/v1/export", json={"days": ["2026-11-22"]})
    assert r.status_code == 200 and r.json()["photos"] == 1

    r2 = client.post("/api/v1/reflow/text", json={"text": SAMPLE_MD})
    assert r2.json()["ok"] and r2.json()["todos"] == 2

    r3 = client.get("/api/v1/search", params={"q": "季度报告"})
    assert r3.status_code == 200 and r3.json()["results"]

    todo_id = db.query_one("SELECT id FROM todos")["id"]
    r4 = client.post(f"/api/v1/todos/{todo_id}/toggle", json={"done": True})
    assert r4.json()["ok"]
    assert db.query_one("SELECT done FROM todos WHERE id=?", (todo_id,))["done"] == 1

    r5 = client.get("/api/v1/summary/1")
    assert r5.status_code == 200 and len(r5.json()["todos"]) == 2

    # API Key 未配置 → 400 而非 500；status available=False
    assert client.get("/api/v1/llm/status").json()["available"] is False
    r6 = client.post("/api/v1/llm/summary/2026-11-22")
    assert r6.status_code == 400


def test_reflow_zip_channel(db, cfg, tmp_path):
    from echocard.export.reflow import import_from_zip

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("summary.md", SAMPLE_MD)
    result = import_from_zip(db, cfg, buf.getvalue())
    assert result["ok"] and result["day"] == "2026-11-22"
