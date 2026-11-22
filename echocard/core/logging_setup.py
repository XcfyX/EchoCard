"""日志初始化：控制台 + 数据目录内滚动文件。"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FMT))
    root.addHandler(console)
    file_h = RotatingFileHandler(
        log_dir / "echocard.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_h.setFormatter(logging.Formatter(_FMT))
    root.addHandler(file_h)
