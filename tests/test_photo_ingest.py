"""照片管道测试：EXIF 归属 / mtime 兜底 / 去重 / 压缩图尺寸 / HEIC / 非法格式。"""
from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path

from PIL import Image

from echocard.ingest.photo_ingest import ingest_photo_bytes, ingest_photo_file
from conftest import make_jpeg


def _jpeg_with_datetime() -> bytes:
    img = Image.new("RGB", (200, 150), (10, 120, 60))
    ex = img.getexif()
    ex[0x0132] = "2026:03:15 10:20:30"  # Image DateTime
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=ex)
    return buf.getvalue()


def test_exif_date_honored(db, cfg):
    res = ingest_photo_bytes(db, cfg, _jpeg_with_datetime(), source="lan",
                             original_name="P1.jpg")
    assert res.ok and res.day == "2026-03-15"
    row = db.query_one("SELECT * FROM photos")
    assert row["captured_ts"] == "2026-03-15 10:20:30"
    assert Path(row["original_path"]).exists()
    assert Path(row["compressed_path"]).exists()


def test_mtime_fallback(db, cfg, tmp_path):
    p = tmp_path / "P2.jpg"
    p.write_bytes(make_jpeg((300, 200)))
    ts = 1768176000  # 2026-01-12 04:00:00 UTC 附近的本地时间
    os.utime(p, (ts, ts))
    res = ingest_photo_file(db, cfg, p, source="watch_dir")
    assert res.ok
    row = db.query_one("SELECT * FROM photos")
    assert row["date_fallback"] == 0
    assert row["captured_at"] == datetime.fromtimestamp(ts).date().isoformat()


def test_dedup_by_hash(db, cfg):
    data = make_jpeg((400, 300))
    r1 = ingest_photo_bytes(db, cfg, data, source="lan", original_name="a.jpg")
    r2 = ingest_photo_bytes(db, cfg, data, source="lan", original_name="b.jpg")
    assert r1.ok and not r1.duplicate
    assert r2.ok and r2.duplicate
    assert db.query_one("SELECT COUNT(*) c FROM photos")["c"] == 1


def test_compressed_long_edge_and_naming(db, cfg):
    res = ingest_photo_bytes(db, cfg, make_jpeg((3000, 2000)), source="lan",
                             original_name="big.jpg", captured_at="2026-11-22 09:05:03")
    assert res.ok
    from PIL import Image as I

    img = I.open(db.query_one("SELECT * FROM photos")["compressed_path"])
    assert max(img.size) <= 1024
    name = Path(db.query_one("SELECT * FROM photos")["compressed_path"]).name
    assert name.startswith("09-05-03_01.jpg")


def test_unsupported_format(db, cfg):
    res = ingest_photo_bytes(db, cfg, b"GIF89a", source="lan", original_name="x.gif")
    assert not res.ok and "不支持" in res.reason


def test_corrupt_image_keeps_original_index(db, cfg):
    data = b"\xff\xd8\xff\xe0broken" + b"\x00" * 100
    res = ingest_photo_bytes(db, cfg, data, source="lan", original_name="bad.jpg")
    assert not res.ok and "解析失败" in res.reason
    # 反偏见：原图索引保留在 originals 目录
    day_dir = cfg.subdir("photos", res.day or datetime.now().date().isoformat())
    assert list((day_dir / "originals").glob("bad*.jpg"))


def test_heic_supported(db, cfg):
    try:
        img = Image.new("RGB", (640, 480), (200, 80, 40))
        buf = io.BytesIO()
        img.save(buf, format="HEIF")
    except Exception:
        import pytest

        pytest.skip("pillow-heif 编码不可用")
    res = ingest_photo_bytes(db, cfg, buf.getvalue(), source="lan",
                             original_name="iphone.heic")
    assert res.ok and res.day
