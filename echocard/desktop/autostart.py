"""开机自启：当前用户注册表 Run 键（无需管理员权限）。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import winreg

log = logging.getLogger(__name__)

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "EchoCard"


def _command() -> str:
    py = Path(sys.executable)  # .venv 里的 pythonw.exe 优先
    exe = py.with_name("pythonw.exe") if py.name.lower() == "python.exe" else py
    if not exe.exists():
        exe = py
    return f'"{exe}" -m echocard'


def set_enabled(enabled: bool) -> None:
    try:
        if enabled:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                                winreg.KEY_SET_VALUE) as k:
                winreg.SetValueEx(k, VALUE_NAME, 0, winreg.REG_SZ, _command())
            log.info("开机自启已开启：%s", _command())
        else:
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0,
                                    winreg.KEY_SET_VALUE) as k:
                    winreg.DeleteValue(k, VALUE_NAME)
                log.info("开机自启已关闭")
            except FileNotFoundError:
                pass
    except OSError:
        log.exception("注册表操作失败")


def is_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, VALUE_NAME)
            return True
    except FileNotFoundError:
        return False
