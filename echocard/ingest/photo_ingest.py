"""照片归档管道：EXIF 时间归属、内容哈希去重、面向大模型的压缩图。

反偏见/反误导设计：只归档与标注客观信息（拍摄时间、来源），解析失败的照片
保留原图索引并标记失败原因，绝不擅自删除。
"""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import exifread
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()  # iPhone HEIC 支持（拍板第 9 项）
except ImportError:  # pragma: no cover
    pillow_heif = None

from ..core.config import AppConfig
from ..core.db import Database
from ..core.utils import now_iso, parse_dt, unique_path

log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
COMPRESS_LONG_EDGE = 1024
COMPRESS_QUALITY = 82
MIN_SANE_YEAR = 2023


@dataclass
class PhotoResult:
    ok: bool
    photo_id: int | None = None
    day: str | None = None
    duplicate: bool = False
    reason: str = ""


def is_supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_EXTS


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exif_datetime(data: bytes) -> datetime | None:
    try:
        tags = exifread.process_file(io.BytesIO(data), details=False)
        raw = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if raw:
            return parse_dt(str(raw))
    except Exception:
        pass
    return None


def ingest_photo_bytes(
    db: Database,
    cfg: AppConfig,
    data: bytes,
    *,
    source: str,
    original_name: str,
    captured_at: str | None = None,
    file_mtime: datetime | None = None,
    device_id: int | None = None,
    bookmark: str | None = None,
) -> PhotoResult:
    ext = Path(original_name).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        return PhotoResult(False, reason=f"不支持的图片格式 {ext}")

    sha = _sha256(data)
    if db.query_one("SELECT 1 FROM photos WHERE src_sha256=?", (sha,)):
        return PhotoResult(True, duplicate=True, reason="内容哈希已存在，跳过重复照片")

    # 拍摄时间优先级：显式指定（扫码页/API）> EXIF DateTimeOriginal > 文件 mtime > 当日
    fallback = False
    dt = parse_dt(captured_at)
    if dt is None:
        dt = _exif_datetime(data)
    if dt is None and file_mtime is not None:
        dt = file_mtime
    if dt is None or dt.year < MIN_SANE_YEAR:
        dt = datetime.now()
        fallback = True
    day = dt.date().isoformat()

    # 压缩图解析（失败则只留原图索引并报告，不删任何东西）
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        compressed_bytes = _compress(img)
    except Exception as e:
        originals_dir = cfg.subdir("photos", day, "originals")
        orig = unique_path(originals_dir / Path(original_name).name)
        orig.write_bytes(data)
        return PhotoResult(False, reason=f"图片解析失败（已保留原图索引 {orig.name}）：{e}")

    originals_dir = cfg.subdir("photos", day, "originals")
    orig_path = unique_path(originals_dir / Path(original_name).name)
    orig_path.write_bytes(data)

    seq = db.query_one("SELECT COUNT(*) AS c FROM photos WHERE captured_at=?", (day,))["c"] + 1
    comp_dir = cfg.subdir("photos", day)
    comp_path = unique_path(comp_dir / f"{dt:%H-%M-%S}_{seq:02d}.jpg")
    comp_path.write_bytes(compressed_bytes)

    try:
        cur = db.execute(
            "INSERT INTO photos(device_id, source, src_sha256, captured_at, captured_ts,"
            " date_fallback, original_path, compressed_path, bookmark, imported_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (device_id, source, sha, day, dt.strftime("%Y-%m-%d %H:%M:%S"),
             1 if fallback else 0, str(orig_path), str(comp_path), bookmark, now_iso()),
        )
    except Exception:
        orig_path.unlink(missing_ok=True)
        comp_path.unlink(missing_ok=True)
        raise
    log.info("照片已归档：%s → %s（原图 %s）", original_name, comp_path, orig_path.name)
    return PhotoResult(True, photo_id=cur.lastrowid, day=day)


def ingest_photo_file(
    db: Database,
    cfg: AppConfig,
    path: str | Path,
    *,
    source: str = "watch_dir",
    captured_at: str | None = None,
    device_id: int | None = None,
    bookmark: str | None = None,
) -> PhotoResult:
    p = Path(path)
    try:
        data = p.read_bytes()
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
    except OSError as e:
        return PhotoResult(False, reason=f"读取失败：{e}")
    return ingest_photo_bytes(
        db, cfg, data, source=source, original_name=p.name,
        captured_at=captured_at, file_mtime=mtime,
        device_id=device_id, bookmark=bookmark,
    )


def _compress(img: Image.Image) -> bytes:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = COMPRESS_LONG_EDGE / max(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=COMPRESS_QUALITY, optimize=True)
    return buf.getvalue()
