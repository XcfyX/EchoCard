"""PyWebView 独立窗口壳：无地址栏/标签页感，关闭窗口缩进托盘继续运行。"""
from __future__ import annotations

import logging
import threading

from .toast import notify

log = logging.getLogger(__name__)


class WindowShell:
    """管理 webview 窗口生命周期（单例窗口，可从托盘重开）。"""

    def __init__(self, url: str, title: str = "EchoCard"):
        self.url = url
        self.title = title
        self._window = None
        self._lock = threading.Lock()

    def open(self, path: str = "") -> None:
        target = self.url + path if path.startswith("/") else self.url

        def _open_async():
            import webview

            with self._lock:
                if self._window is not None:
                    try:
                        self._window.show()
                        return
                    except Exception:
                        self._window = None
                try:
                    win = webview.create_window(
                        self.title, target, width=1320, height=860,
                        min_size=(1000, 660), background_color="#0f1115",
                    )
                    win.events.closed += self._on_closed
                    self._window = win
                    webview.start(gui="edgechromium" if hasattr(webview, "gui") else None)
                except Exception:
                    log.exception("打开主窗口失败")
                    notify("EchoCard", "窗口打开失败，请重试或使用浏览器访问 " + self.url)

        threading.Thread(target=_open_async, daemon=True, name="webview").start()

    def _on_closed(self) -> None:
        self._window = None
        notify("EchoCard 已缩进托盘", "后台服务继续运行 · 点托盘图标可重新打开")
