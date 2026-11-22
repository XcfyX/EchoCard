"""设备认亲与绑定落库。

安全闸门：`recognize_volume` 未返回 bound=True 之前，导入器绝不触碰该卷。
首次绑定的写入操作只有两处：
  1. devices 表落库（本地数据库）；
  2. `bind_volume_manually`（界面向导主动调用）写 .echocard_source 标记文件。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..core.config import AppConfig, DEVICE_MARKER_NAME
from ..core.db import Database
from ..core.utils import now_iso
from .usb_volumes import VolumeInfo

log = logging.getLogger(__name__)


@dataclass
class BindDecision:
    bound: bool
    device_id: int | None
    reason: str  # marker / vid_or_product / serial_remembered /
                 # not_recognized / dji_detected_but_autobind_disabled
    volume: VolumeInfo


def recognize_volume(vol: VolumeInfo, db: Database, cfg: AppConfig) -> BindDecision:
    existing = None
    if vol.usbstor_serial:
        existing = db.query_one("SELECT * FROM devices WHERE serial=?", (vol.usbstor_serial,))

    # 1) 设备根目录标记文件（向导式绑定的产物）——最可靠
    if vol.marker_exists():
        return _bind(vol, db, existing, "marker")

    # 2) 之前绑定过的设备（序列号记忆）——用户清了标记也能认出
    if existing is not None:
        db.execute(
            "UPDATE devices SET last_seen=?, volume_label=? WHERE id=?",
            (now_iso(), vol.volume_label, existing["id"]),
        )
        return BindDecision(True, existing["id"], "serial_remembered", vol)

    # 3) DJI 系硬件（VID 候选 / USBSTOR 厂商·产品串）
    if vol.looks_like_dji():
        if cfg.auto_bind_dji:
            return _bind(vol, db, existing, "vid_or_product")
        return BindDecision(False, None, "dji_detected_but_autobind_disabled", vol)

    # 4) 普通U盘/相机卡 —— 绝不触碰
    return BindDecision(False, None, "not_recognized", vol)


def _bind(vol: VolumeInfo, db: Database, existing, reason: str) -> BindDecision:
    kind = "dji_mic_tx" if vol.is_dji_vid() else "generic_usb"
    if existing is not None:
        db.execute(
            "UPDATE devices SET kind=?, vid=?, pid=?, volume_label=?, product_desc=?,"
            " marker_root=?, last_seen=? WHERE id=?",
            (kind, vol.vid, vol.pid, vol.volume_label, vol.product_desc,
             vol.marker_path, now_iso(), existing["id"]),
        )
        dev_id = existing["id"]
    else:
        cur = db.execute(
            "INSERT INTO devices(kind, vid, pid, serial, volume_label, product_desc,"
            " marker_root, bound_at, last_seen) VALUES(?,?,?,?,?,?,?,?,?)",
            (kind, vol.vid, vol.pid, vol.usbstor_serial, vol.volume_label,
             vol.product_desc, vol.marker_path, now_iso(), now_iso()),
        )
        dev_id = cur.lastrowid
    log.info("设备已绑定：kind=%s serial=%s 原因=%s 盘符=%s:",
             kind, vol.usbstor_serial, reason, vol.drive_letter)
    return BindDecision(True, dev_id, reason, vol)


def bind_volume_manually(vol: VolumeInfo, db: Database) -> BindDecision:
    """向导式一键绑定：向设备根写入 .echocard_source 标记并落库（仅限用户主动操作）。"""
    marker = Path(vol.root) / DEVICE_MARKER_NAME
    marker.write_text("echo card bound device\n", encoding="ascii")
    vol.marker_path = str(marker)
    existing = None
    if vol.usbstor_serial:
        existing = db.query_one("SELECT * FROM devices WHERE serial=?", (vol.usbstor_serial,))
    return _bind(vol, db, existing, "marker")
