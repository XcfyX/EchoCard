"""流水线编排测试：历史豁免 / 倒序补全 / 压缩瘦身 / 改名学声纹 / 合并聚类。

不依赖真实模型：monkeypatch 转写与压缩函数，注入确定性假 Embedding。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import echocard.pipeline as pl
from echocard.audio.vad_asr import Segment
from echocard.pipeline import Pipeline
from conftest import make_wav


class FakeEngines:
    def __init__(self):
        self.provider_used = "cpu"

    def embedding(self, samples) -> np.ndarray:
        """确定性假声纹：按样本长度奇偶给出两个不同向量族。"""
        v = 1.0 if (len(samples) // 16000) % 2 == 0 else 0.25
        out = np.full(192, v, dtype=np.float32)
        return out / np.linalg.norm(out)


def _fake_transcribe_factory(segments_by_file):
    """segments_by_file: {文件名 stem: (speech_segs, music_segs)}"""

    def fake_transcribe(engines, wav16k, file_start, detect_music_gaps=True):
        stem = Path(wav16k).stem
        return segments_by_file[stem]

    return fake_transcribe


def _seg(s, e, rms=-20.0):
    return Segment(start_s=s, end_s=e, text=f"片段{s}", rms_db=rms,
                   audio_tags=["speech"], events=["Speech"])


def _insert_day(db, day, stem, seconds=8):
    """插入一条 ingest_files 记录并生成真实 16k 原始 WAV。"""
    raw = make_wav(Path(db.path).parent / "raw" / f"{stem}.wav", seconds=seconds, rate=16000)
    cur = db.execute(
        "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, recorded_ts, raw_path, status, imported_at)"
        " VALUES(NULL,'usb',?,?,?,?,?,?,'imported','t')",
        (str(raw), "fakesha" + stem, raw.stat().st_size, day, f"{day} 10:00:00", str(raw)),
    )
    return cur.lastrowid, raw


@pytest.fixture()
def patch_all(monkeypatch, tmp_path):
    """把 pipeline 的重函数全部换成可控假实现。"""
    def fake_to_16k(src, dst, normalize=True):
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(src).read_bytes())  # 16k 假数据原样复制（read_pcm16 可读）
        return dst

    def fake_opus(src, dst, bitrate_kbps=32):
        Path(dst).write_bytes(Path(src).read_bytes()[:1024])
        return Path(dst)

    monkeypatch.setattr(pl, "to_16k_mono", fake_to_16k)
    monkeypatch.setattr(pl, "wav_to_opus", fake_opus)
    return patch_all


def test_pending_days_recent_only_and_newest_first(db, cfg, patch_all):
    for day, stem in [("2026-11-22", "A"), ("2026-11-23", "B"), ("2025-05-12", "C")]:
        _insert_day(db, day, stem)
    p = Pipeline(db, cfg)
    assert p.pending_days() == ["2026-11-23", "2026-11-22"], "仅 >= 分界线，且最新优先"
    assert p.historical_days() == ["2025-05-12"], "历史归档日单独列出"


def test_process_recent_day_full_flow(db, cfg, patch_all, monkeypatch):
    fid, raw = _insert_day(db, "2026-11-22", "DJI_01_20261122_100000", seconds=8)
    speech = [_seg(0.5, 4.5, -18), _seg(5, 7.5, -25)]
    music = [Segment(start_s=8.0, end_s=15.0, events=["Music"],
                     audio_tags=["music"], rms_db=-30)]
    monkeypatch.setattr(pl, "transcribe_file",
                        _fake_transcribe_factory({"DJI_01_20261122_100000": (speech, music)}))
    p = Pipeline(db, cfg, engines=FakeEngines())
    result = p.process_day("2026-11-22", keep_raw=False)

    assert result["ok"] and result["segments"] == 2 and result["speakers"] >= 1
    segs = db.query("SELECT * FROM audio_segments ORDER BY start_offset")
    assert len(segs) == 2
    assert segs[0]["text"] == "片段0.5"
    assert segs[0]["day"] == "2026-11-22"
    assert segs[0]["start_ts"] == "10:00:00", "挂钟时间来自文件录制起点"
    assert json.loads(segs[0]["audio_tags"]) == ["speech"], "客观标签保留"
    # 音乐节点落库
    tracks = db.query("SELECT * FROM music_tracks WHERE day='2026-11-22'")
    assert len(tracks) == 1 and tracks[0]["status"] == "detected"
    # 瘦身：Opus 存在、原 WAV 删除、状态 done
    row = db.query_one("SELECT * FROM ingest_files WHERE id=?", (fid,))
    assert row["status"] == "done" and row["compressed_path"]
    assert Path(row["compressed_path"]).exists()
    assert not Path(row["raw_path"]).exists(), "非历史日：原始 WAV 应被安全瘦身删除"
    # 声纹向量落盘
    assert (Path(cfg.data_root) / "audio" / "2026-11-22" / "embeddings.npz").exists()


def test_historical_day_manual_trigger_keeps_raw(db, cfg, patch_all, monkeypatch):
    fid, raw = _insert_day(db, "2025-05-12", "DJI_01_20250512_170000", seconds=6)
    monkeypatch.setattr(pl, "transcribe_file",
                        _fake_transcribe_factory(
                            {"DJI_01_20250512_170000": ([_seg(0, 4)], [])}))
    p = Pipeline(db, cfg, engines=FakeEngines())
    result = p.process_day("2025-05-12")  # 手动触发（不传 keep_raw）

    assert result["ok"] and result["keep_raw"] is True
    row = db.query_one("SELECT * FROM ingest_files WHERE id=?", (fid,))
    assert row["status"] == "done"
    assert Path(row["raw_path"]).exists(), "历史归档日：原 WAV 必须保留"
    assert not row["compressed_path"] or Path(row["compressed_path"]).exists()


def test_rename_learns_voiceprint_and_updates_rows(db, cfg, patch_all, monkeypatch):
    fid, raw = _insert_day(db, "2026-11-22", "DJI_01_20261122_100000", seconds=8)
    monkeypatch.setattr(
        pl, "transcribe_file",
        _fake_transcribe_factory(
            {"DJI_01_20261122_100000": ([_seg(0, 4), _seg(5, 7.5)], [])}))
    p = Pipeline(db, cfg, engines=FakeEngines())
    p.process_day("2026-11-22", keep_raw=False)
    cid = db.query_one("SELECT cluster_id FROM audio_segments")["cluster_id"]

    learned = p.rename_speaker("2026-11-22", cid, "我")
    assert learned > 0
    prof = db.query_one("SELECT * FROM speaker_profiles WHERE name='我'")
    assert prof is not None and len(json.loads(prof["prototypes"])) >= 1
    names = {r["speaker_name"] for r in db.query(
        "SELECT speaker_name FROM audio_segments WHERE day='2026-11-22'")}
    assert names == {"我"}, "改名后全文称呼刷新"


def test_merge_clusters(db, cfg, patch_all, monkeypatch):
    fid, raw = _insert_day(db, "2026-11-22", "DJI_01_20261122_100000", seconds=8)
    monkeypatch.setattr(
        pl, "transcribe_file",
        _fake_transcribe_factory(
            {"DJI_01_20261122_100000": ([_seg(0, 4), _seg(5, 7.5)], [])}))
    p = Pipeline(db, cfg, engines=FakeEngines())
    p.process_day("2026-11-22", keep_raw=False)
    # 人工把后一段改成 cluster 2，再合并回 1
    seg2 = db.query_one(
        "SELECT id FROM audio_segments ORDER BY start_offset DESC LIMIT 1")["id"]
    db.execute("UPDATE audio_segments SET cluster_id=2 WHERE id=?", (seg2,))
    p.merge_speakers("2026-11-22", keep_id=1, gone_id=2, name="我")
    ids = {r["cluster_id"] for r in db.query(
        "SELECT cluster_id FROM audio_segments WHERE day='2026-11-22'")}
    assert ids == {1}, "合并后被并编号消失"
    assert db.query_one("SELECT COUNT(*) c FROM speaker_profiles WHERE name='我'")["c"] == 1


def test_worker_auto_scans_on_idle(db, cfg, patch_all, monkeypatch):
    """空闲时自动扫描补全：入队 2026-11-22 并处理成功。"""
    _insert_day(db, "2026-11-22", "DJI_01_20261122_100000", seconds=6)
    monkeypatch.setattr(pl, "transcribe_file",
                        _fake_transcribe_factory(
                            {"DJI_01_20261122_100000": ([_seg(0, 3)], [])}))
    p = Pipeline(db, cfg, engines=FakeEngines())
    p.run_auto()
    assert p.pending_days() == ["2026-11-22"]
    p.process_day("2026-11-22", keep_raw=False)
    assert p.pending_days() == [], "处理完成后不再视为待处理"
