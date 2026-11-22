"""通用小工具：时间与路径。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

DT_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y:%m:%d %H:%M:%S",  # EXIF 传统格式
    "%Y-%m-%d",
)


def now_iso() -> str:
    """本地 ISO 时间串（约定：整个项目时间戳统一用本地时间 YYYY-MM-DD HH:MM:SS）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def parse_dt(value: str | None) -> datetime | None:
    """宽松解析用户/API/EXIF 提供的时间描述；失败返回 None。"""
    if not value:
        return None
    value = value.strip()
    for fmt in DT_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def unique_path(path: Path) -> Path:
    """目标文件已存在时追加 -1/-2 后缀，绝不覆盖已有文件。"""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 10000):
        candidate = path.with_name(f"{stem}-{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法为 {path} 生成唯一文件名")
