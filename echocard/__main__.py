"""EchoCard 启动入口。

模式：
  python -m echocard            → 独立桌面窗口 + 托盘 + Toast（默认）
  python -m echocard --dev      → 前台本地服务（开发/验收模式，浏览器访问）
  python -m echocard --no-usb   → 禁用 USB 监听
  python -m echocard --window-only → 只开桌面窗口（服务线程内启动）
"""
from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from .core.config import APP_VERSION, AppConfig
from .core.db import Database
from .core.events import EventBus
from .core.logging_setup import setup_logging
from .ingest.dji_import import DeviceImporter
from .ingest.photo_watch import PhotoWatcher
from .ingest.usb_monitor import UsbMonitor

log = logging.getLogger(__name__)


def _build_runtime(cfg: AppConfig, args) -> tuple:
    """装配 DB / 事件 / 流水线 / 监听器 → (app, db, cfg, events, watcher, monitor, pipeline)."""
    setup_logging(cfg.subdir("logs"))
    log.info("EchoCard v%s 启动 · 数据目录：%s", APP_VERSION, cfg.data_root)

    db = Database(cfg.subdir("db") / "echocard.db")
    events = EventBus()
    importer = DeviceImporter(db, cfg, events)

    from .core.fts import rebuild_fts

    indexed = rebuild_fts(db)
    if indexed:
        log.info("FTS5 全文索引就绪：%d 条记录", indexed)

    from .pipeline import Pipeline

    pipeline = Pipeline(db, cfg, events)
    pipeline.recover_stalled()  # 复位上次中断遗留的 processing 行
    events.subscribe(
        "pipeline.day_ready",
        lambda d: pipeline.enqueue(d["day"])
        if d.get("day") and d["day"] >= cfg.auto_transcribe_since else None,
    )
    pipeline.start_worker()

    monitor: UsbMonitor | None = None
    if not args.no_usb:
        monitor = UsbMonitor(db, cfg, importer, events)
        monitor.start()
    watcher = PhotoWatcher(db, cfg)
    watcher.start()

    threading.Thread(target=pipeline.run_auto, daemon=True).start()

    from .desktop.server import create_app

    app = create_app(db, cfg, events, importer, watcher=watcher,
                     monitor=monitor, pipeline=pipeline)
    return app, db, watcher, monitor, pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="echocard", description="EchoCard 本地服务")
    parser.add_argument("--config", help="配置文件路径（默认 %%APPDATA%%/EchoCard/config.json）")
    parser.add_argument("--data-root", help="覆盖数据根目录（便携/调试）")
    parser.add_argument("--dev", action="store_true", help="前台运行服务（浏览器访问，不开窗口）")
    parser.add_argument("--no-usb", action="store_true", help="禁用 USB 监听")
    parser.add_argument("--version", action="version", version=f"EchoCard {APP_VERSION}")
    args = parser.parse_args()

    cfg = AppConfig.load(Path(args.config) if args.config else None)
    if args.data_root:
        cfg.data_root = args.data_root
    cfg.save()

    app, db, watcher, monitor, pipeline = _build_runtime(cfg, args)
    url = f"http://127.0.0.1:{cfg.server_port}"

    if args.dev:
        log.info("控制台：%s      手机传图（局域网）：%s/upload", url, url)
        try:
            import uvicorn

            uvicorn.run(app, host="127.0.0.1", port=cfg.server_port, log_level="warning")
        finally:
            watcher.stop()
            if monitor:
                monitor.stop()
            pipeline.stop_worker()
            db.close()
        return

    # ---------- 桌面模式：服务线程 + PyWebView 窗口 + 托盘 + Toast ----------
    from .desktop.toast import notify
    from .desktop.tray import build_tray
    from .desktop.window import WindowShell

    def _serve():
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=cfg.server_port,
                    log_level="warning", access_log=False)

    threading.Thread(target=_serve, daemon=True, name="uvicorn").start()

    shell = WindowShell(url)
    quit_event = threading.Event()
    icon = build_tray(
        on_open=shell.open,
        on_quit=quit_event.set,
        server_url=url,
    )
    notify("EchoCard 已启动", f"本地记忆工作台就绪：{url}")
    shell.open()
    log.info("桌面模式运行中（%s）· 托盘常驻 · 完全退出请走托盘菜单", url)

    try:
        while not quit_event.wait(1.0):
            pass
    finally:
        log.info("EchoCard 完全退出")
        pipeline.stop_worker()
        watcher.stop()
        if monitor:
            monitor.stop()
        try:
            icon.stop()
        except Exception:
            pass
        db.close()


if __name__ == "__main__":
    main()
