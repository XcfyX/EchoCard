"""聚类 / 声纹库 / 音乐合并 单元测试（纯算法，不依赖模型）。"""
from __future__ import annotations

import json

import numpy as np

from echocard.audio.diarize import cluster_centroid, cluster_segments, soft_match
from echocard.audio.music_detect import merge_music_candidates
from echocard.audio.speakers import learn_from_cluster
from echocard.audio.vad_asr import Segment


_DIR_A = np.tile(np.array([1.0, -1.0], dtype=np.float32), 96)   # 正交方向 A
_DIR_B = np.tile(np.array([1.0, 1.0], dtype=np.float32), 96)    # 正交方向 B


def _seg(s: float, e: float, direction: str = "A", rms: float = -20.0) -> Segment:
    seg = Segment(start_s=s, end_s=e, rms_db=rms)
    base = _DIR_A if direction == "A" else _DIR_B
    seg.embedding = base / np.linalg.norm(base)
    return seg


def test_cluster_two_speakers():
    segs = [
        _seg(0, 5, "B"), _seg(6, 9, "B"),           # 说话人 A（更长 → 编号 1）
        _seg(10, 14, "A"), _seg(15, 17, "A"),       # 说话人 B（正交方向）
    ]
    cluster_segments(segs, distance_threshold=0.5)
    by_cluster = {}
    for s in segs:
        by_cluster.setdefault(s.cluster_id, []).append(s)
    assert len(by_cluster) == 2
    assert len(by_cluster[1]) == 2  # 总时长更长的组 = 说话人 1


def test_representative_selection_prefers_loud():
    from echocard.audio.diarize import pick_representative

    segs = [_seg(0, 4, "A", rms=-30), _seg(5, 9, "A", rms=-12), _seg(10, 14, "A", rms=-20)]
    for s in segs:
        s.cluster_id = 1
    reps = pick_representative(segs, 1, count=2)
    assert len(reps) == 2 and reps[0].rms_db == -12 and reps[1].rms_db == -20


def test_soft_match_thresholds(tmp_path, db):
    vec = (_DIR_A / np.linalg.norm(_DIR_A)).astype(np.float32)
    learn_from_cluster(db, [vec], "我")
    # 完全一致 → 高置信度
    name, score, status = soft_match(db, vec.copy())
    assert status == "matched" and name == "我" and score > 0.99
    # 明显不同（正交）→ 未匹配
    other = (_DIR_B / np.linalg.norm(_DIR_B)).astype(np.float32)
    name2, score2, status2 = soft_match(db, other)
    assert status2 == "unmatched" and name2 is None


def test_voiceprint_updates_blend_not_explode(db):
    base = np.full(192, 0.9, dtype=np.float32)
    base /= np.linalg.norm(base)
    learn_from_cluster(db, [base], "我")
    row = db.query_one("SELECT * FROM speaker_profiles WHERE name='我'")
    n1 = len(json.loads(row["prototypes"]))
    for _ in range(10):  # 相似样本反复学习 → 原型数量不膨胀
        noisy = base + np.random.default_rng(0).normal(0, 0.01, 192).astype(np.float32)
        learn_from_cluster(db, [noisy], "我")
    row2 = db.query_one("SELECT * FROM speaker_profiles WHERE name='我'")
    assert len(json.loads(row2["prototypes"])) == n1
    assert row2["sample_count"] == 11


def test_merge_music_candidates():
    a = Segment(start_s=0, end_s=10, events=["Music"])
    b = Segment(start_s=13, end_s=20, events=["Music"])   # 间隔 3s → 合并
    c = Segment(start_s=60, end_s=70, events=["Music"])   # 间隔 40s → 独立
    out = merge_music_candidates([a, b, c])
    assert len(out) == 2
    assert abs(out[0][0] - 0) < 1e-6 and abs(out[0][1] - 20) < 1e-6
    assert abs(out[1][0] - 60) < 1e-6


def test_merge_music_filters_short():
    a = Segment(start_s=0, end_s=2, events=["Music"])  # < 3s → 过滤
    assert merge_music_candidates([a]) == []
