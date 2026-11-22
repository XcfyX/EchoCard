"""配置与数据库基础测试。"""
from __future__ import annotations

import sqlite3

from echocard.core.config import AppConfig
from echocard.core.db import Database


def test_default_redline_switches_off():
    c = AppConfig()
    assert c.delete_after_verify is False, "设备清空红线开关默认必须关闭"
    assert c.auto_music_recognition is False


def test_config_roundtrip(tmp_path):
    p = tmp_path / "cfg.json"
    c = AppConfig.load(p)
    c.data_root = str(tmp_path / "data")
    c.watched_photo_dirs = [r"D:\Sync\Photos"]
    c.save(p)
    c2 = AppConfig.load(p)
    assert c2.data_root == str(tmp_path / "data")
    assert c2.watched_photo_dirs == [r"D:\Sync\Photos"]


def test_config_ignores_unknown_keys(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"unknown_key": 1, "theme": "light"}', encoding="utf-8")
    c = AppConfig.load(p)
    assert c.theme == "light"


def test_db_wal_and_fts5_capable(tmp_path):
    db = Database(tmp_path / "t.db")
    mode = db.query_one("PRAGMA journal_mode")["journal_mode"]
    assert mode.lower() == "wal"
    # python.org 官方 Windows 轮子自带 FTS5（M3 全文检索依赖）
    try:
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_probe USING fts5(text)")
    except sqlite3.OperationalError:
        raise AssertionError("FTS5 不可用")
    db.execute("INSERT INTO fts_probe(text) VALUES('你好世界 hello')")
    assert db.query_one("SELECT * FROM fts_probe WHERE fts_probe MATCH 'hello'")


def test_schema_idempotent(tmp_path):
    p = tmp_path / "t.db"
    Database(p)
    Database(p)  # 二次打开重复执行 schema 不报错
    Database(p).query_one("SELECT COUNT(*) c FROM devices")


def test_unique_hash_constraint(db):
    db.execute(
        "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, raw_path, imported_at) VALUES(NULL,'usb','a','H1',10,'2026-11-22','a','t')"
    )
    try:
        db.execute(
            "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
            " recorded_at, raw_path, imported_at) VALUES(NULL,'usb','b','H1',10,'2026-11-22','b','t')"
        )
        raise AssertionError("重复哈希应当触发 UNIQUE 约束")
    except sqlite3.IntegrityError:
        pass
