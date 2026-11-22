"""M1 测试公共夹具：临时配置/数据库 + 模拟可移动卷 + WAV/JPEG 生成器。"""
from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path

import pytest
from PIL import Image

from echocard.core.config import AppConfig
from echocard.core.db import Database
from echocard.ingest.usb_volumes import VolumeInfo


@pytest.fixture()
def cfg(tmp_path: Path) -> AppConfig:
    c = AppConfig(data_root=str(tmp_path / "data"), delete_after_verify=False)
    c._path = tmp_path / "config.json"  # type: ignore[attr-defined]
    return c


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


def make_wav(path: Path, seconds: float = 0.5, freq: float = 440.0, rate: int = 16000) -> Path:
    """生成合法 PCM16 单声道 WAV（不依赖外部文件）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        n = int(seconds * rate)
        for i in range(n):
            v = int(30000 * math.sin(2 * math.pi * freq * i / rate))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return path


def make_wav_bytes(seconds: float = 0.3) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = bytearray()
        for i in range(int(seconds * 16000)):
            frames += struct.pack("<h", int(30000 * math.sin(2 * math.pi * 440 * i / 16000)))
        w.writeframes(bytes(frames))
    return buf.getvalue()


def make_jpeg(size: tuple[int, int] = (1600, 900)) -> bytes:
    img = Image.new("RGB", size, (120, 140, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def make_volume(
    root: Path,
    *,
    marker: bool = True,
    vid: str | None = "10d6",
    pid: str | None = "b00b",
    serial: str | None = "0123456798AB",
    vendor: str | None = "___DJI",
    product: str | None = "WIRELESS_MIC_TX2",
    letter: str = "X",
    label: str = "",
) -> VolumeInfo:
    """构造一个模拟可移动卷（root 为临时目录，代替真实盘符）。"""
    root.mkdir(parents=True, exist_ok=True)
    marker_path = None
    if marker:
        mp = root / ".echocard_source"
        mp.write_text("bound", encoding="ascii")
        marker_path = str(mp)
    return VolumeInfo(
        drive_letter=letter,
        root=str(root),
        volume_label=label,
        drive_type=2,
        is_usb=True,
        vid=vid,
        pid=pid,
        usbstor_serial=serial,
        vendor_desc=vendor,
        product_desc=product,
        marker_path=marker_path,
    )
