"""Windows Toast 通知（win11toast 封装，线程内 fire-and-forget）。"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


def notify(title: str, message: str) -> None:
    def _run():
        try:
            import asyncio

            from win11toast import toast

            asyncio.run(toast(title, message, duration="short"))
        except Exception:
            log.debug("Toast 通知失败（可忽略）：%s - %s", title, message)

    threading.Thread(target=_run, daemon=True, name="toast").start()
