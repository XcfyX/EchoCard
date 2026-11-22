"""USB 可移动卷监听：轮询认亲 + 触发导入 + 明确记录「未绑定已忽略」。

M1 采用 2 秒轮询（跨机型可靠、易于测试）；M4 桌面化时可叠加
WM_DEVICECHANGE 设备通知做即时唤醒，轮询保留为兜底。
"""
from __future__ import annotations

import logging
import threading

from ..core.config import AppConfig
from ..core.db import Database
from ..core.events import EventBus
from .dji_import import DeviceImporter
from .usb_volumes import VolumeInfo, list_usb_volumes

log = logging.getLogger(__name__)


class UsbMonitor:
    def __init__(
        self,
        db: Database,
        cfg: AppConfig,
        importer: DeviceImporter,
        events: EventBus,
        interval: float = 2.0,
        enumerator=None,
        inline: bool = False,
    ):
        self.db = db
        self.cfg = cfg
        self.importer = importer
        self.events = events
        self.interval = interval
        self.inline = inline  # 测试模式：同步执行导入（不另起线程）
        self._enumerate = enumerator or list_usb_volumes  # 可注入便于测试
        self._seen: dict[str, str] = {}  # 盘符 -> stable_key
        self._busy: set[str] = set()     # 正在导入的 stable_key，防重入
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="usb-monitor", daemon=True)
        self._thread.start()
        log.info("USB 监听已启动（每 %.1f 秒扫描）", self.interval)

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception:
                log.exception("USB 轮询异常")
            self._stop.wait(self.interval)

    def scan_once(self) -> None:
        vols: list[VolumeInfo] = self._enumerate()
        current: dict[str, str] = {}
        for vol in vols:
            key = vol.stable_key()
            current[vol.drive_letter] = key
            if vol.drive_letter not in self._seen:
                self._on_arrival(vol)
        for letter in set(self._seen) - set(current):
            self.events.publish("usb.device_removed", {"drive": f"{letter}:"})
            log.info("可移动卷已移除：%s:", letter)
        self._seen = current

    def _on_arrival(self, vol: VolumeInfo) -> None:
        self.events.publish("usb.device_arrived",
                            {"drive": f"{vol.drive_letter}:", "serial": vol.usbstor_serial,
                             "label": vol.volume_label, "dji_like": vol.looks_like_dji()})
        key = vol.stable_key()
        with self._lock:
            if key in self._busy:
                log.info("%s: 正在导入中，忽略重复触发", vol.drive_letter)
                return
            self._busy.add(key)
        if self.inline:
            self._import_worker(vol, key)
        else:
            threading.Thread(target=self._import_worker, args=(vol, key),
                             name=f"import-{vol.drive_letter}", daemon=True).start()

    def _import_worker(self, vol: VolumeInfo, key: str) -> None:
        try:
            report = self.importer.import_volume(vol)
            if report is None:
                self.events.publish("usb.unbound_ignored", {
                    "drive": f"{vol.drive_letter}:",
                    "note": "未认亲设备已忽略（绝不触碰）",
                })
        except Exception:
            log.exception("导入线程异常：%s:", vol.drive_letter)
        finally:
            with self._lock:
                self._busy.discard(key)
