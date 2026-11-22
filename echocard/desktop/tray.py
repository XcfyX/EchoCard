"""系统托盘（pystray）：打开主窗口 / 扫码传图 / 查看状态 / 完全退出。"""
from __future__ import annotations

import logging
import threading

from PIL import Image, ImageDraw

log = logging.getLogger(__name__)


def _icon_image(size: int = 64) -> Image.Image:
    """程序托盘图标：深蓝底 + E 字（运行时生成，无外部资源依赖）。"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([2, 2, size - 2, size - 2], radius=14,
                        fill=(24, 34, 68, 255), outline=(108, 140, 255, 255), width=2)
    d.text((size / 2 - 9, size / 2 - 14), "E", fill=(230, 235, 255, 255),
           font_size=int(size * 0.6))
    return img


def build_tray(on_open, on_quit, server_url: str):
    """返回已 detach 启动的托盘图标。on_open/on_quit 由桌面壳提供。"""
    import pystray

    def _status_item():
        from .toast import notify

        notify("EchoCard 状态", f"服务运行中：{server_url}")

    menu = pystray.Menu(
        pystray.MenuItem("打开主窗口", lambda *_: on_open(), default=True),
        pystray.MenuItem("打开扫码传图", lambda *_: on_open("/upload")),
        pystray.MenuItem("查看状态", lambda *_: _status_item()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("完全退出", lambda *_: on_quit()),
    )
    icon = pystray.Icon("EchoCard", _icon_image(), "EchoCard — 本地记忆工作台", menu)
    threading.Thread(target=icon.run_detached, daemon=True, name="tray").start()
    return icon
