"""本地同步目录监控（Syncthing / 网盘相册目录）→ 自动归档新增照片。

采用「事件入队 + 工作线程等文件大小稳定后再入库」策略，
避免在网盘/同步工具尚未写完文件时就误读半截文件。
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler

from ..core.config import AppConfig
from ..core.db import Database
from .photo_ingest import ingest_photo_file, is_supported

log = logging.getLogger(__name__)

_STABLE_ROUNDS = 2      # 连续 N 轮大小一致
_STABLE_INTERVAL = 0.7  # 每轮间隔秒


class _NewFileHandler(FileSystemEventHandler):
    def __init__(self, sink: "queue.Queue[str]"):
        self._sink = sink

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if is_supported(str(event.src_path)):
            self._sink.put(str(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        # Syncthing/浏览器下载常见「先写临时名再改名」
        if event.is_directory:
            return
        dest = str(event.dest_path)
        if is_supported(dest):
            self._sink.put(dest)


class PhotoWatcher:
    def __init__(self, db: Database, cfg: AppConfig):
        self.db = db
        self.cfg = cfg
        self._q: "queue.Queue[str]" = queue.Queue()
        self._observer = Observer()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- 生命周期 ----
    def start(self) -> None:
        self._reschedule()
        self._worker = threading.Thread(target=self._worker_loop, name="photo-watch", daemon=True)
        self._worker.start()
        log.info("照片目录监控已启动：%s", self.cfg.watched_photo_dirs or "（未配置目录）")

    def stop(self) -> None:
        self._stop.set()
        self._observer.stop()
        self._observer.join(timeout=3)

    def reschedule(self) -> None:
        """配置变更后重新挂载监控目录（供设置页调用）。"""
        self._reschedule()

    def _reschedule(self) -> None:
        self._observer.unschedule_all()  # 公开 API，跨平台实现均支持
        for d in self.cfg.watched_photo_dirs:
            p = Path(d)
            if p.is_dir():
                self._observer.schedule(_NewFileHandler(self._q), str(p), recursive=False)
                log.info("监控照片目录：%s", d)
            else:
                log.warning("监控目录不存在，跳过：%s", d)

    # ---- 工作线程 ----
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if not self._wait_stable(path):
                    continue
                res = ingest_photo_file(self.db, self.cfg, path, source="watch_dir")
                if not res.ok:
                    log.warning("照片入库失败：%s（%s）", path, res.reason)
            except Exception:
                log.exception("照片监控处理异常：%s", path)

    @staticmethod
    def _wait_stable(path: str) -> bool:
        last, stable = -1, 0
        while stable < _STABLE_ROUNDS:
            if not Path(path).exists():
                return False  # 被移走/删除（可能已被同步工具改名）
            try:
                size = Path(path).stat().st_size
            except OSError:
                return False
            if size == last and size > 0:
                stable += 1
            else:
                stable, last = 0, size
            time.sleep(_STABLE_INTERVAL)
        return True
