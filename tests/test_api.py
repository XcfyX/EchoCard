"""REST API 测试：状态 / 上传双通道 / 配置白名单 / 静态控制台页。"""
from __future__ import annotations

import io

from fastapi.testclient import TestClient

from echocard.core.events import EventBus
from echocard.desktop.server import create_app
from echocard.ingest.dji_import import DeviceImporter
from conftest import make_jpeg, make_wav_bytes


def _client(db, cfg):
    events = EventBus()
    app = create_app(db, cfg, events, DeviceImporter(db, cfg, events))
    return TestClient(app)


def test_status_and_static_pages(db, cfg):
    client = _client(db, cfg)
    r = client.get("/api/v1/status")
    assert r.status_code == 200
    body = r.json()
    assert body["delete_after_verify"] is False, "状态接口必须如实暴露红线开关"
    assert client.get("/").status_code == 200
    assert client.get("/upload").status_code == 200


def test_lan_photo_upload(db, cfg):
    client = _client(db, cfg)
    r = client.post(
        "/api/v1/ingest/photo",
        files={"file": ("phone.jpg", make_jpeg((800, 600)), "image/jpeg")},
        data={"captured_at": "2026-11-22 08:30:00", "device_id": "pixel-9",
              "bookmark": "班会"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["day"] == "2026-11-22"
    row = db.query_one("SELECT * FROM photos")
    assert row["bookmark"] == "班会" and row["source"] == "lan"
    dev = db.query_one("SELECT * FROM devices WHERE kind='lan_api'")
    assert dev["serial"] == "pixel-9"


def test_lan_audio_upload_and_dedup(db, cfg):
    client = _client(db, cfg)
    wav = make_wav_bytes()
    r1 = client.post(
        "/api/v1/ingest/audio",
        files={"file": ("clip.wav", wav, "audio/wav")},
        data={"captured_at": "2026-11-22 08:00:00", "device_id": "esp32-s3"},
    )
    assert r1.status_code == 200 and r1.json()["ok"]
    r2 = client.post(
        "/api/v1/ingest/audio",
        files={"file": ("clip.wav", wav, "audio/wav")},
    )
    assert r2.json()["duplicate"] is True
    assert db.query_one("SELECT COUNT(*) c FROM ingest_files")["c"] == 1
    row = db.query_one("SELECT * FROM ingest_files")
    assert row["source"] == "lan_api" and row["date_fallback"] == 0


def test_lan_audio_rejects_non_wav(db, cfg):
    client = _client(db, cfg)
    r = client.post("/api/v1/ingest/audio",
                    files={"file": ("song.mp3", b"ID3", "audio/mpeg")})
    assert r.status_code == 400


def test_config_patch_whitelist(db, cfg):
    client = _client(db, cfg)
    r = client.patch("/api/v1/config", json={"watched_photo_dirs": [r"D:\Sync\Photos"]})
    assert r.json()["ok"]
    assert cfg.watched_photo_dirs == [r"D:\Sync\Photos"]
    # 白名单之外的字段被拒绝
    r2 = client.patch("/api/v1/config", json={"api_key": "hack"})
    assert r2.status_code == 422 or r2.status_code == 400


def test_usb_scan_and_manual_bind_404(db, cfg):
    client = _client(db, cfg)
    assert client.get("/api/v1/usb/scan").status_code == 200
    assert client.post("/api/v1/usb/Z:/bind").status_code == 404
