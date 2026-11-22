"""M4 测试：日历状态 / 切片与照片端点 / 开机自启注册表 / 托盘图标。"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import echocard.desktop.autostart as autostart
from echocard.core.events import EventBus
from echocard.desktop.server import create_app
from echocard.ingest.dji_import import DeviceImporter
from conftest import make_jpeg, make_wav


def _client(db, cfg, tmp_path):
    # 准备一个真实切片文件
    clips = Path(cfg.data_root) / "audio" / "2026-11-22" / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    make_wav(clips / "speaker1_1.wav", seconds=1)
    return TestClient(create_app(db, cfg, EventBus(), DeviceImporter(db, cfg)))


def _seed_day(db, tmp_path):
    db.execute("INSERT INTO devices(kind, serial, bound_at) VALUES('dji_mic_tx','S1','t')")
    raw = make_wav(tmp_path / "a.wav", seconds=1)
    db.execute(
        "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, recorded_ts, raw_path, status, imported_at)"
        " VALUES(1,'usb',?,'h1',?, '2026-11-22','2026-11-22 10:00:00',?, 'done','t')",
        (str(raw), raw.stat().st_size, str(raw)))
    db.execute(
        "INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
        " end_offset, text, cluster_id, speaker_name, audio_tags)"
        " VALUES(1,'2026-11-22','10:00:00','10:01:00',0,60,'早上好',1,'说话人 1','[]')")


def test_days_endpoint_states(db, cfg, tmp_path):
    _seed_day(db, tmp_path)
    db.execute(
        "INSERT INTO daily_summaries(day_range, content_md, imported_at)"
        " VALUES('2026-11-23','历史总结','t')")
    client = _client(db, cfg, tmp_path)
    data = client.get("/api/v1/days").json()
    by_day = {d["day"]: d for d in data["days"]}
    assert by_day["2026-11-22"]["state"] == "blue"      # 已清洗待投喂
    assert by_day["2026-11-23"]["state"] == "green"     # 已回流归档
    assert by_day["2026-11-22"]["historical"] is False


def test_audio_clip_validation(db, cfg, tmp_path):
    client = _client(db, cfg, tmp_path)
    assert client.get("/api/v1/audio/clip", params={"day": "2026-11-22",
                                                    "name": "speaker1_1.wav"}).status_code == 200
    assert client.get("/api/v1/audio/clip", params={"day": "2026-11-22",
                                                    "name": "../../etc/passwd"}).status_code == 400
    assert client.get("/api/v1/audio/clip", params={"day": "bad-date",
                                                    "name": "speaker1_1.wav"}).status_code == 400


def test_static_photo_blocks_traversal(db, cfg, tmp_path):
    client = _client(db, cfg, tmp_path)
    img = Path(cfg.data_root) / "photos" / "2026-11-22" / "p.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(make_jpeg((100, 80)))
    assert client.get("/static-photo", params={"src": str(img)}).status_code == 200
    assert client.get("/static-photo", params={"src": r"C:\Windows\win.ini"}).status_code == 400
    assert client.get("/static-photo", params={"src": "relative/path.jpg"}).status_code == 400


def test_autostart_registry_roundtrip(monkeypatch):
    monkeypatch.setattr(autostart, "VALUE_NAME", "EchoCardSelfTest")
    autostart.set_enabled(False)
    assert autostart.is_enabled() is False
    autostart.set_enabled(True)
    assert autostart.is_enabled() is True
    cmd = autostart._command()
    assert "echocard" in cmd.lower()
    autostart.set_enabled(False)
    assert autostart.is_enabled() is False


def test_tray_icon_generated():
    from echocard.desktop.tray import _icon_image

    img = _icon_image()
    assert img.size == (64, 64) and img.getpixel((32, 32))[3] == 255


def test_photo_include_toggle(db, cfg, tmp_path):
    _seed_day(db, tmp_path)
    img = tmp_path / "x.jpg"
    img.write_bytes(make_jpeg((100, 80)))
    db.execute(
        "INSERT INTO photos(device_id, source, src_sha256, captured_at, captured_ts,"
        " original_path, compressed_path, imported_at)"
        " VALUES(1,'lan','px','2026-11-22','2026-11-22 10:05:00',?,?,'t')",
        (str(img), str(img)))
    client = _client(db, cfg, tmp_path)
    pid = db.query_one("SELECT id FROM photos")["id"]
    assert db.query_one("SELECT include_in_export FROM photos WHERE id=?", (pid,))["include_in_export"] == 1
    client.post(f"/api/v1/photos/{pid}/include", json={"starred": False})
    assert db.query_one("SELECT include_in_export FROM photos WHERE id=?", (pid,))["include_in_export"] == 0
