"""时间轴顺序与短句归类的回归测试（2026-11-23 用户反馈「时间轴乱七八糟」）。

根因：
1. 时间轴按「文件内偏移 start_offset」排序 → 跨文件完全错序
   （12:44 的 50 秒位置排在 07:37 的 122 秒位置之前）；
2. 27% 的段落因时长 <1s 被聚类跳过 → speaker_name 为空，时间轴大片空白。
"""
from __future__ import annotations

import numpy as np
import pytest

from echocard.audio.diarize import assign_short_segments, cluster_segments
from echocard.audio.vad_asr import Segment
from conftest import make_jpeg


def _make_srv(db, cfg):
    from fastapi.testclient import TestClient

    from echocard.core.events import EventBus
    from echocard.desktop.server import create_app
    from echocard.ingest.dji_import import DeviceImporter

    return TestClient(create_app(db, cfg, EventBus(), DeviceImporter(db, cfg)))


def _seed(db, rows):
    """rows: [(file_id, start_ts, start_offset, text, cluster_id, speaker)]"""
    for fid, ts, off, text, cid, name in rows:
        db.execute(
            "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
            " recorded_at, raw_path, status, imported_at)"
            " SELECT NULL,'usb',?,'h'||?,10,'2026-11-23','r','done','t'"
            " WHERE NOT EXISTS (SELECT 1 FROM ingest_files WHERE src_sha256='h'||?)",
            (f"p{fid}", fid, fid),
        )
        fid_real = db.query_one(
            "SELECT id FROM ingest_files WHERE src_sha256=?", (f"h{fid}",))["id"]
        db.execute(
            "INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
            " end_offset, text, cluster_id, speaker_name, audio_tags)"
            " VALUES(?,'2026-11-23',?,?,?,?,?,?,?,'[]')",
            (fid_real, ts, ts, off, off + 2, text, cid, name))


def test_timeline_orders_by_wall_clock_not_file_offset(db, cfg):
    """跨文件：07:37 的文件内偏移很大，仍必须排在 12:44 之前。"""
    _seed(db, [
        # 文件 A：12:44 的段落，文件内偏移很小（50s）
        (1, "12:44:35", 50.7, "下午的话", 1, "说话人 1"),
        (1, "12:44:39", 54.8, "继续下午", 1, "说话人 1"),
        # 文件 B：07:37 的段落，文件内偏移很大（122s）
        (2, "07:37:29", 122.5, "早上的话", 2, "说话人 2"),
        (2, "07:42:19", 160.2, "上午的话", 2, "说话人 2"),
    ])
    client = _make_srv(db, cfg)
    segs = client.get("/api/v1/day/2026-11-23/timeline").json()["segments"]
    times = [s["start_ts"] for s in segs]
    assert times == sorted(times), f"时间轴必须按挂钟时间升序，实际 {times}"
    assert times[0] == "07:37:29" and times[-1] == "12:44:39"


def test_export_markdown_also_orders_by_wall_clock(db, cfg):
    from echocard.export.feedpack import build_markdown

    _seed(db, [
        (1, "12:44:35", 50.7, "下午的话", 1, "说话人 1"),
        (2, "07:37:29", 122.5, "早上的话", 2, "说话人 2"),
    ])
    md, _ = build_markdown(db, cfg, ["2026-11-23"])
    assert md.index("早上的话") < md.index("下午的话"), "投喂包的语音时间轴也须按挂钟时间"


# ---------------------------------------------------------------- 短句归属
def test_short_segments_inherit_nearest_speaker():
    """<1s 的短句应就近归入同文件说话人，而不是留空。"""
    emb = np.zeros(192, dtype=np.float32)
    emb[0] = 1.0  # 有效非零向量（零向量会被防御性剔除）
    long_a = Segment(start_s=0.0, end_s=4.0, text="长句A", file_id=1)
    long_a.embedding = emb.copy()
    short = Segment(start_s=4.5, end_s=5.0, text="嗯", file_id=1)   # 紧跟其后
    long_a2 = Segment(start_s=20.0, end_s=24.0, text="长句A2", file_id=1)
    long_a2.embedding = emb.copy()
    far_short = Segment(start_s=60.0, end_s=60.4, text="嗯", file_id=1)  # 距任何长句都很远
    other_file = Segment(start_s=4.5, end_s=5.0, text="嗯", file_id=2)  # 另一个文件，不应跨文件继承

    cluster_segments([long_a, long_a2], 0.5)
    n = assign_short_segments([long_a, long_a2, short, far_short, other_file], max_gap_s=3.0)
    assert short.cluster_id == long_a.cluster_id, "紧邻短句应继承前一句说话人"
    assert far_short.cluster_id is None, "距长句过远的短句不应被强行归属"
    assert other_file.cluster_id is None, "不得跨文件继承"
    assert n == 1


def test_cluster_segments_tolerates_zero_vectors():
    """零向量/NaN 不得让聚类崩溃（防御性剔除）。"""
    bad = Segment(start_s=0, end_s=4, text="坏向量", file_id=1)
    bad.embedding = np.zeros(192, dtype=np.float32)
    good = Segment(start_s=5, end_s=9, text="好向量", file_id=1)
    emb = np.zeros(192, dtype=np.float32); emb[0] = 1.0
    good.embedding = emb
    groups = cluster_segments([bad, good], 0.5)
    assert good.cluster_id is not None, "有效向量必须被聚类"
    assert bad.cluster_id is None, "零向量应被跳过而不是污染距离矩阵"


def test_short_segments_do_not_overwrite_existing_clusters():
    a = Segment(start_s=0, end_s=4, text="A", file_id=1)
    a.cluster_id = 7
    a.embedding = np.zeros(192, dtype=np.float32)
    b = Segment(start_s=4, end_s=4.5, text="嗯", file_id=1)
    b.cluster_id = 3  # 已有编号（理论上短句无 embedding，但防御性检查）
    assign_short_segments([a, b], max_gap_s=3.0)
    assert b.cluster_id == 3, "已有聚类的短句不得被改写"


def test_timeline_returns_speaker_for_short_segments(db, cfg):
    """短句已归属后，时间轴 API 必须返回 speaker_name（不再为空）。"""
    _seed(db, [
        (1, "07:37:29", 0.0, "长句", 5, "疑似-我 (说话人 5)"),
        (1, "07:37:33", 4.0, "嗯", 5, "疑似-我 (说话人 5)"),
    ])
    client = _make_srv(db, cfg)
    segs = client.get("/api/v1/day/2026-11-23/timeline").json()["segments"]
    assert all(s["speaker_name"] for s in segs), "所有段落都应带说话人称呼"
