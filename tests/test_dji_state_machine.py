"""模拟 U 盘安全导入状态机测试（M1 红线验收核心）。

覆盖：安全清空流程 / 清空开关默认关闭 / 中途断连 / 大小损坏 / 零字节 /
非目标U盘保护 / 重复插拔去重 / 设备时间漂移 / 隐藏与系统文件保护 /
标记认亲 / 序列号记忆认亲。
"""
from __future__ import annotations

import hashlib
import os
from datetime import date
from pathlib import Path

import pytest

from echocard.core.config import AppConfig
from echocard.core.db import Database
from echocard.ingest.dji_import import DeviceImporter, ImportReport
from echocard.ingest.usb_monitor import UsbMonitor
from conftest import make_volume, make_wav

AUDIO_SUB = "DJI_Audio_001"


def _importer(db: Database, cfg: AppConfig) -> DeviceImporter:
    return DeviceImporter(db, cfg)


# ---------------------------------------------------------------- 1. 安全清空
def test_import_and_safe_delete(db, cfg, tmp_path):
    cfg.delete_after_verify = True
    dev = tmp_path / "device"
    vol = make_volume(dev)
    f1 = make_wav(dev / AUDIO_SUB / "DJI_01_20250512_174014.WAV")
    f2 = make_wav(dev / AUDIO_SUB / "DJI_02_20250512_175043.WAV", freq=880)

    report = _importer(db, cfg).import_volume(vol)

    assert report.imported == 2
    assert report.failed == 0
    assert report.deleted_on_device == 2, "校验+落库成功后应删除单个源文件"
    assert not f1.exists() and not f2.exists()
    rows = db.query("SELECT * FROM ingest_files ORDER BY id")
    assert len(rows) == 2
    assert all(r["deleted_on_device"] == 1 for r in rows)
    # 归属日期来自文件名内嵌时间戳
    assert {r["recorded_at"] for r in rows} == {"2025-05-12"}
    # 本地落盘内容与源文件一致（SHA-256 复核）
    for r in rows:
        local = Path(r["raw_path"])
        assert local.exists() and local.stat().st_size == r["size_bytes"]
        assert hashlib.sha256(local.read_bytes()).hexdigest() == r["src_sha256"]
    # 不残留临时文件
    assert not list((tmp_path / "data" / "audio" / "2025-05-12" / "raw").glob(".tmp-*"))


# ---------------------------------------------------------------- 2. 开关默认关闭
def test_delete_disabled_keeps_source(db, cfg, tmp_path):
    assert cfg.delete_after_verify is False, "默认配置红线：清空开关必须关闭"
    dev = tmp_path / "device"
    vol = make_volume(dev)
    f1 = make_wav(dev / AUDIO_SUB / "DJI_01_20250512_174014.WAV")

    report = _importer(db, cfg).import_volume(vol)

    assert report.imported == 1 and report.deleted_on_device == 0
    assert f1.exists(), "清空开关关闭时源文件必须保留"


# ---------------------------------------------------------------- 3. 中途断连
def test_mid_disconnect_keeps_unfinished_source(db, cfg, tmp_path, monkeypatch):
    cfg.delete_after_verify = True
    dev = tmp_path / "device"
    vol = make_volume(dev)
    f1 = make_wav(dev / AUDIO_SUB / "DJI_01_20250512_174014.WAV")
    f2 = make_wav(dev / AUDIO_SUB / "DJI_02_20250512_175043.WAV")

    orig = DeviceImporter._copy_with_hash
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("设备中断连接（模拟拔出）")
        return orig(src, dst)

    monkeypatch.setattr(DeviceImporter, "_copy_with_hash", staticmethod(flaky))
    report = _importer(db, cfg).import_volume(vol)

    assert report.imported == 1 and report.failed == 1
    assert not f1.exists(), "第 1 个文件已完整校验落库，允许清空"
    assert f2.exists(), "断连文件必须保留在设备上"
    assert any("中断" in e or "断开" in e for e in report.errors)


# ---------------------------------------------------------------- 4. 大小损坏
def test_size_mismatch_not_deleted(db, cfg, tmp_path, monkeypatch):
    cfg.delete_after_verify = True
    dev = tmp_path / "device"
    vol = make_volume(dev)
    f1 = make_wav(dev / AUDIO_SUB / "DJI_01_20250512_174014.WAV")

    orig = DeviceImporter._copy_with_hash

    def truncated(src, dst):
        sha, copied = orig(src, dst)
        return sha, copied - 128  # 模拟拷贝截断

    monkeypatch.setattr(DeviceImporter, "_copy_with_hash", staticmethod(truncated))
    report = _importer(db, cfg).import_volume(vol)

    assert report.imported == 0 and report.failed == 1
    assert f1.exists(), "校验失败绝不删除源文件"
    assert db.query_one("SELECT COUNT(*) c FROM ingest_files")["c"] == 0


# ---------------------------------------------------------------- 5. 零字节
def test_zero_byte_file_not_deleted(db, cfg, tmp_path):
    cfg.delete_after_verify = True
    dev = tmp_path / "device"
    vol = make_volume(dev)
    good = make_wav(dev / AUDIO_SUB / "DJI_01_20250512_174014.WAV")
    zero = dev / AUDIO_SUB / "DJI_00_20250512_170000.WAV"
    zero.write_bytes(b"")

    report = _importer(db, cfg).import_volume(vol)

    assert report.imported == 1 and report.failed == 1
    assert zero.exists(), "零字节坏文件保留（仅标记，不删除）"
    assert not good.exists()


# ---------------------------------------------------------------- 6. 非目标U盘保护
def test_unbound_volume_untouched(db, cfg, tmp_path):
    dev = tmp_path / "usbdrive"
    vol = make_volume(
        dev, marker=False, vid="0abcd", serial="AAA111",
        vendor="SanDisk", product="Cruzer_Blade",
    )
    f1 = make_wav(dev / "music" / "song.WAV")

    report = _importer(db, cfg).import_volume(vol)

    assert report is None, "未认亲卷必须返回 None 且绝不触碰"
    assert f1.exists()
    assert db.query_one("SELECT COUNT(*) c FROM ingest_files")["c"] == 0
    assert db.query_one("SELECT COUNT(*) c FROM devices")["c"] == 0


def test_autobind_disabled_dji_not_touched(db, tmp_path):
    cfg = AppConfig(data_root=str(tmp_path / "data"), auto_bind_dji=False)
    cfg._path = tmp_path / "config.json"  # type: ignore[attr-defined]
    dev = tmp_path / "device"
    vol = make_volume(dev, marker=False)
    f1 = make_wav(dev / AUDIO_SUB / "DJI_01_20250512_174014.WAV")

    assert _importer(db, cfg).import_volume(vol) is None
    assert f1.exists()


# ---------------------------------------------------------------- 7. 重复插拔去重
def test_replug_dedup(db, cfg, tmp_path):
    dev = tmp_path / "device"
    vol = make_volume(dev)
    f1 = make_wav(dev / AUDIO_SUB / "DJI_01_20250512_174014.WAV")
    f2 = make_wav(dev / AUDIO_SUB / "DJI_02_20250512_175043.WAV", freq=880)

    r1 = _importer(db, cfg).import_volume(vol)
    assert r1.imported == 2
    r2 = _importer(db, cfg).import_volume(vol)  # 当天再次插拔
    assert r2.imported == 0
    assert r2.skipped_dedup == 2, "内容哈希去重：重复插拔不产生重复记录"
    assert r2.failed == 0 and r2.deleted_on_device == 0
    assert f1.exists() and f2.exists()
    assert db.query_one("SELECT COUNT(*) c FROM ingest_files")["c"] == 2


# ---------------------------------------------------------------- 8. 时间漂移兜底
def test_time_drift_fallback_keeps_order(db, cfg, tmp_path):
    dev = tmp_path / "device"
    vol = make_volume(dev)
    old = 1546300800  # 2019-01-01（早于 2023，视为设备时间漂移）
    p1 = make_wav(dev / AUDIO_SUB / "REC001.WAV")
    p2 = make_wav(dev / AUDIO_SUB / "REC002.WAV", freq=660)
    os.utime(p1, (old, old))
    os.utime(p2, (old, old))

    report = _importer(db, cfg).import_volume(vol)

    assert report.imported == 2
    rows = db.query("SELECT * FROM ingest_files ORDER BY id")
    today = date.today().isoformat()
    assert all(r["recorded_at"] == today for r in rows)
    assert all(r["date_fallback"] == 1 for r in rows)
    assert rows[0]["src_path"].endswith("REC001.WAV"), "按文件序保序"


# ---------------------------------------------------------------- 9. 隐藏/系统文件保护
def test_hidden_and_system_files_untouched(db, cfg, tmp_path):
    dev = tmp_path / "device"
    vol = make_volume(dev)
    good = make_wav(dev / AUDIO_SUB / "DJI_01_20250512_174014.WAV")
    syswav = make_wav(dev / "System Volume Information" / "sys.wav")
    hiddenwav = make_wav(dev / ".trash" / "h.wav")
    dotfile = dev / AUDIO_SUB / ".DS_STORE"
    dotfile.write_bytes(b"x")
    txt = dev / AUDIO_SUB / "notes.txt"
    txt.write_text("hello", encoding="utf-8")

    report = _importer(db, cfg).import_volume(vol)

    assert report.imported == 1
    assert syswav.exists() and hiddenwav.exists()
    assert dotfile.exists() and txt.exists()
    assert db.query_one("SELECT COUNT(*) c FROM ingest_files")["c"] == 1


# ---------------------------------------------------------------- 10. 认亲路径
def test_marker_only_recognition(db, cfg, tmp_path):
    dev = tmp_path / "stick"
    vol = make_volume(dev, marker=True, vid=None, serial="FRESH01",
                      vendor="Generic", product="USB_DISK")
    make_wav(dev / "rec" / "clip.WAV")

    report = _importer(db, cfg).import_volume(vol)
    assert report is not None and report.imported == 1, "仅凭 .echocard_source 标记也应认亲"
    row = db.query_one("SELECT * FROM devices")
    assert row["kind"] == "generic_usb"


def test_serial_remembered_after_marker_removed(db, cfg, tmp_path):
    dev = tmp_path / "device"
    vol = make_volume(dev)  # 首次绑定（marker）
    make_wav(dev / AUDIO_SUB / "DJI_01_20250512_174014.WAV")
    _importer(db, cfg).import_volume(vol)
    assert db.query_one("SELECT COUNT(*) c FROM devices")["c"] == 1

    (dev / ".echocard_source").unlink()  # 用户误删标记
    vol2 = make_volume(dev, marker=False)  # 同序列号再次插入
    make_wav(dev / AUDIO_SUB / "DJI_02_20250512_175043.WAV", freq=880)  # 新增文件
    report = _importer(db, cfg).import_volume(vol2)
    assert report is not None and report.imported == 1, "序列号记忆认亲"
    assert db.query_one("SELECT COUNT(*) c FROM devices")["c"] == 1


# ---------------------------------------------------------------- 11. 事件与防重入
def test_monitor_ignores_unbound_and_publishes_events(db, cfg, tmp_path):
    from echocard.core.events import EventBus

    events = EventBus()
    seen: list[dict] = []
    events.subscribe("usb.unbound_ignored", lambda d: seen.append(d))
    dev = tmp_path / "usbdrive"
    vol = make_volume(dev, marker=False, vid="0abcd", serial="AAA111",
                      vendor="SanDisk", product="Cruzer_Blade")
    make_wav(dev / "song.WAV")

    monitor = UsbMonitor(db, cfg, _importer(db, cfg), events,
                         interval=999, enumerator=lambda: [vol], inline=True)
    monitor.scan_once()
    assert seen and seen[0]["drive"] == "X:", "未绑定设备必须显式告知「已忽略」"
    assert db.query_one("SELECT COUNT(*) c FROM ingest_files")["c"] == 0


def test_day_ready_event_published(db, cfg, tmp_path):
    from echocard.core.events import EventBus

    events = EventBus()
    ready: list[dict] = []
    events.subscribe("pipeline.day_ready", lambda d: ready.append(d))
    dev = tmp_path / "device"
    vol = make_volume(dev)
    make_wav(dev / AUDIO_SUB / "DJI_01_20250512_174014.WAV")

    monitor = UsbMonitor(db, cfg, DeviceImporter(db, cfg, events), events,
                         interval=999, enumerator=lambda: [vol], inline=True)
    monitor.scan_once()
    assert ready and ready[0]["day"] == "2025-05-12"
