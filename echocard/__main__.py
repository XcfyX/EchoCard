"""EchoCard 启动入口。

M1（当前）：python -m echocard --dev
  本地服务 + USB 认亲监听 + 照片目录监控（前台运行，控制台打印地址）
M4（规划）：--window 以 PyWebView 独立窗口启动 + pystray 托盘 + Toast 通知
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .core.config import APP_VERSION, AppConfig
from .core.db import Database
from .core.events import EventBus
from .core.logging_setup import setup_logging
from .ingest.dji_import import DeviceImporter
from .ingest.photo_watch import PhotoWatcher
from .ingest.usb_monitor import UsbMonitor

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(prog="echocard", description="EchoCard 本地服务")
    parser.add_argument("--config", help="配置文件路径（默认 %%APPDATA%%/EchoCard/config.json）")
    parser.add_argument("--data-root", help="覆盖数据根目录（便携/调试）")
    parser.add_argument("--dev", action="store_true", help="前台运行并打印访问地址")
    parser.add_argument("--no-usb", action="store_true", help="禁用 USB 监听（纯服务模式）")
    parser.add_argument("--version", action="version", version=f"EchoCard {APP_VERSION}")
    args = parser.parse_args()

    cfg = AppConfig.load(Path(args.config) if args.config else None)
    if args.data_root:
        cfg.data_root = args.data_root
    cfg.save()

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
    # 导入完成 → 分界线内日期自动入队（历史日期不自动转写）
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

    # 启动即扫描补全未处理日期（线程内执行，避免阻塞服务启动）
    import threading as _threading

    _threading.Thread(target=pipeline.run_auto, daemon=True).start()

    from .desktop.server import create_app

    app = create_app(db, cfg, events, importer, watcher=watcher,
                     monitor=monitor, pipeline=pipeline)
    url = f"http://127.0.0.1:{cfg.server_port}"
    log.info("控制台：%s      手机传图（局域网）：%s/upload", url, url)

    try:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=cfg.server_port, log_level="warning")
    finally:
        watcher.stop()
        if monitor:
            monitor.stop()
        db.close()


if __name__ == "__main__":
    main()
