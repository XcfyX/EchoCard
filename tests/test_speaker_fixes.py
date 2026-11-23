"""说话人聚类/改名/合并/跨天声纹回归测试（2026-11-23 线上问题）。

对应缺陷：
1. 按文件独立聚类 → 同一天编号重复（说话人 1 实际有几十个人）
2. 改名后无法试听（切片文件与 DB 路径不一致）
3. 合并后名字按调用顺序丢失已命名的人名
4. 重处理不清理旧切片 → UI 播放孤儿文件
5. 跨天声纹匹配阈值不当导致「同一个人识别不出来」
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from echocard.audio.diarize import (cluster_centroid, cluster_segments,
                                    pick_representative, soft_match)
from echocard.audio.speakers import learn_from_cluster
from echocard.audio.vad_asr import Segment
from echocard.pipeline import Pipeline


class FakeEngines:
    provider_used = "cpu"

    def embedding(self, samples):
        v = np.full(192, 1.0 if (len(samples) // 8000) % 2 == 0 else 0.25, dtype=np.float32)
        return v / np.linalg.norm(v)


def _seed_ingest(db, day="2026-11-23", fid=1, sha=None):
    """插入父表 ingest_files 行，满足 audio_segments 的外键约束。"""
    db.execute(
        "INSERT INTO ingest_files(id, device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, raw_path, status, imported_at)"
        " VALUES(?,NULL,'usb',?,?,10,?,?,'done','t')",
        (fid, f"p{fid}", sha or f"h{fid}", day, f"r{fid}"))


def _seg(start, end, fam="A", file_id=1, rms=-20.0):
    base = np.tile(np.array([1.0, -1.0] if fam == "A" else [1.0, 1.0], dtype=np.float32), 96)
    base /= np.linalg.norm(base)
    s = Segment(start_s=start, end_s=end, text="t", rms_db=rms, file_id=file_id)
    s.embedding = base
    return s


# ---------------------------------------------------------------- 1. 全局聚类
def test_day_level_clustering_is_global_across_files():
    """两个文件的片段必须一起聚类：同一人跨文件得到同一编号，不会各从 1 开始。"""
    segs = [
        _seg(0, 5, "A", file_id=1), _seg(6, 9, "A", file_id=1),
        _seg(10, 15, "B", file_id=2), _seg(16, 19, "B", file_id=2),
        _seg(20, 24, "A", file_id=2),          # 文件 2 里也有 A 的声音
    ]
    groups = cluster_segments(segs, distance_threshold=0.5)
    assert len(groups) == 2, f"应聚成 2 人，实际 {len(groups)}"
    a_ids = {s.cluster_id for s in segs if s.embedding[0] > 0 and s.embedding[1] < 0}
    b_ids = {s.cluster_id for s in segs if s.embedding[1] > 0}
    assert len(a_ids) == 1 and len(b_ids) == 1, "同一人跨文件必须同一编号"
    # 人数最多的组编号靠前
    sizes = {cid: len(v) for cid, v in groups.items()}
    assert max(sizes, key=sizes.get) == 1


def test_start_id_offsets_new_clusters():
    segs = [_seg(0, 5, "A"), _seg(6, 9, "B")]
    groups = cluster_segments(segs, distance_threshold=0.5, start_id=51)
    assert min(groups) >= 51, "重处理时新编号必须顺延，避免与保留分段冲突"


# ---------------------------------------------------------------- 2. 代表切片
def test_representative_covers_multiple_files():
    segs = [_seg(0, 4, "A", file_id=1, rms=-10),
            _seg(5, 9, "A", file_id=2, rms=-30),
            _seg(10, 14, "A", file_id=3, rms=-20)]
    for s in segs:
        s.cluster_id = 1
    reps = pick_representative(segs, 1, count=2)
    assert len(reps) == 2
    assert len({r.file_id for r in reps}) == 2, "代表切片应覆盖不同文件"


def test_representative_fallback_when_no_ideal_duration():
    segs = [_seg(0, 1, "A"), _seg(2, 2.5, "A")]
    for s in segs:
        s.cluster_id = 1
    reps = pick_representative(segs, 1, count=2)
    assert len(reps) == 2, "无 3~8 秒理想片段时应退回 RMS 最高者，而不是空列表"


# ---------------------------------------------------------------- 3. 跨天声纹
def test_soft_match_uses_max_prototype_and_cross_day_thresholds(db):
    """跨天同人相似度 0.5 左右应判定为 suspected，而不是完全认不出。"""
    rng = np.random.default_rng(7)
    base = rng.normal(size=192).astype(np.float32)
    base /= np.linalg.norm(base)
    learn_from_cluster(db, [base], "我")

    # 精确构造 cos ≈ 0.5 的「同一个人、隔天换了佩戴」样本
    ortho = rng.normal(size=192).astype(np.float32)
    ortho -= ortho @ base * base
    ortho /= np.linalg.norm(ortho)
    noisy = 0.5 * base + np.sqrt(1 - 0.25) * ortho
    noisy /= np.linalg.norm(noisy)
    score = float(noisy @ base)
    assert abs(score - 0.5) < 0.02, f"构造失败：{score}"

    name, s, status = soft_match(db, noisy, high=0.62, mid=0.42)
    assert name == "我" and status == "suspected", \
        f"跨天同人 0.5 应判 suspected（实际 {status}, score={s:.3f}）"

    # 略高于 high 阈值 → matched
    strong = 0.75 * base + np.sqrt(1 - 0.5625) * ortho
    strong /= np.linalg.norm(strong)
    name2, _s2, status2 = soft_match(db, strong, high=0.62, mid=0.42)
    assert status2 == "matched" and name2 == "我"

    # 完全不同的人 → unmatched
    other = rng.normal(size=192).astype(np.float32)
    other /= np.linalg.norm(other)
    _n3, s3, status3 = soft_match(db, other, high=0.62, mid=0.42)
    assert status3 == "unmatched", f"陌生人不应匹配（score={s3:.3f}）"


def test_soft_match_ignores_malformed_prototypes(db):
    db.execute("INSERT INTO speaker_profiles(name, prototypes, sample_count, updated_at)"
               " VALUES('坏档案','[]',0,'t')")
    vec = np.zeros(192, dtype=np.float32); vec[0] = 1.0
    name, score, status = soft_match(db, vec)
    assert status in ("matched", "suspected", "unmatched")  # 不抛异常即可
    assert name != "坏档案"


# ---------------------------------------------------------------- 4. 改名/合并（服务层）
@pytest.fixture()
def pipeline_with_clips(db, cfg, monkeypatch):
    import echocard.pipeline as pl
    from conftest import make_wav

    def fake_to_16k(src, dst, normalize=True):
        dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(src).read_bytes()); return dst

    def fake_opus(src, dst, bitrate_kbps=32):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"x"); return Path(dst)

    monkeypatch.setattr(pl, "to_16k_mono", fake_to_16k)
    monkeypatch.setattr(pl, "wav_to_opus", fake_opus)
    return Pipeline(db, cfg, engines=FakeEngines())


def test_rename_renames_clip_files_and_updates_db(db, cfg, tmp_path, pipeline_with_clips):
    _seed_ingest(db)
    p = pipeline_with_clips
    clips = Path(cfg.data_root) / "audio" / "2026-11-23" / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        (clips / f"speaker1_{i}.wav").write_bytes(b"RIFF")
    db.execute("INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
               " end_offset, text, cluster_id, speaker_name, audio_tags, segment_path)"
               " VALUES(1,'2026-11-23','09:00:00','09:00:05',0,5,'t',1,'说话人 1','[]',?)",
               (str(clips / "speaker1_1.wav"),))
    db.execute("INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
               " end_offset, text, cluster_id, speaker_name, audio_tags, segment_path)"
               " VALUES(1,'2026-11-23','09:01:00','09:01:05',1,5,'t',1,'说话人 1','[]',?)",
               (str(clips / "speaker1_2.wav"),))

    p.rename_speaker("2026-11-23", 1, "我")

    # 磁盘：旧的 speaker1_*.wav 必须消失，新的「我_*.wav」存在
    assert not list(clips.glob("speaker1_*.wav")), "改名后旧切片文件必须清理"
    assert (clips / "我_1.wav").exists() and (clips / "我_2.wav").exists()
    # DB：路径必须同步更新，否则 UI 播放会 404（这正是「改名后无法试听」的根因）
    rows = db.query("SELECT segment_path, speaker_name FROM audio_segments"
                    " WHERE day='2026-11-23' AND cluster_id=1")
    assert all(r["speaker_name"] == "我" for r in rows)
    for r in rows:
        assert r["segment_path"] and Path(r["segment_path"]).exists(), \
            f"DB 路径必须指向真实文件：{r['segment_path']}"


def test_merge_keeps_existing_human_name(db, cfg, tmp_path, pipeline_with_clips):
    _seed_ingest(db)
    """合并时若一侧已命名为「我」，即使另一侧是 keep_id 也不能变成说话人 N。"""
    db.execute("INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
               " end_offset, text, cluster_id, speaker_name, audio_tags)"
               " VALUES(1,'2026-11-23','09:00:00','09:00:05',0,5,'t',2,'我','[]')")
    db.execute("INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
               " end_offset, text, cluster_id, speaker_name, audio_tags)"
               " VALUES(1,'2026-11-23','09:01:00','09:01:05',1,5,'t',9,'说话人 9','[]')")
    p = pipeline_with_clips

    # keep_id=9（中性编号）并入已命名的 2 —— 名字必须沿用「我」
    p.merge_speakers("2026-11-23", keep_id=9, gone_id=2, name=None)
    names = {r["speaker_name"] for r in db.query(
        "SELECT speaker_name FROM audio_segments WHERE day='2026-11-23'")}
    assert names == {"我"}, f"合并后必须沿用已命名的人名，实际 {names}"
    assert db.query_one("SELECT COUNT(*) c FROM audio_segments WHERE cluster_id=9")["c"] == 2


def test_merge_prefers_larger_group_when_both_named(db, cfg, tmp_path, pipeline_with_clips):
    _seed_ingest(db)
    db.execute("INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
               " end_offset, text, cluster_id, speaker_name, audio_tags)"
               " VALUES(1,'2026-11-23','09:00:00','09:00:05',0,5,'t',1,'曹开礼','[]')")
    for i in range(3):
        db.execute("INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
                   " end_offset, text, cluster_id, speaker_name, audio_tags)"
                   " VALUES(1,'2026-11-23',?,?,?,5,'t',2,'我','[]')",
                   (f"09:0{i+1}:00", f"09:0{i+1}:05", i))
    p = pipeline_with_clips
    p.merge_speakers("2026-11-23", keep_id=1, gone_id=2, name=None)
    names = {r["speaker_name"] for r in db.query(
        "SELECT speaker_name FROM audio_segments WHERE day='2026-11-23'")}
    assert names == {"我"}, f"两侧都有名字时应保留片段更多的一方，实际 {names}"


def test_merge_explicit_name_wins(db, cfg, tmp_path, pipeline_with_clips):
    _seed_ingest(db)
    db.execute("INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
               " end_offset, text, cluster_id, speaker_name, audio_tags)"
               " VALUES(1,'2026-11-23','09:00:00','09:00:05',0,5,'t',1,'我','[]')")
    db.execute("INSERT INTO audio_segments(file_id, day, start_ts, end_ts, start_offset,"
               " end_offset, text, cluster_id, speaker_name, audio_tags)"
               " VALUES(1,'2026-11-23','09:01:00','09:01:05',1,5,'t',2,'说话人 2','[]')")
    p = pipeline_with_clips
    p.merge_speakers("2026-11-23", keep_id=1, gone_id=2, name="张三")
    names = {r["speaker_name"] for r in db.query(
        "SELECT speaker_name FROM audio_segments WHERE day='2026-11-23'")}
    assert names == {"张三"}


def test_merge_removes_gone_cluster_clips(db, cfg, tmp_path, pipeline_with_clips):
    _seed_ingest(db)
    clips = Path(cfg.data_root) / "audio" / "2026-11-23" / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    (clips / "speaker2_1.wav").write_bytes(b"RIFF")
    p = pipeline_with_clips
    p.merge_speakers("2026-11-23", keep_id=1, gone_id=2, name=None)
    assert not (clips / "speaker2_1.wav").exists(), "被并入聚类的切片必须清理，避免 UI 播放孤儿"
