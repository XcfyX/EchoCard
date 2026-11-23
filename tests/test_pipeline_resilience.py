"""流水线健壮性回归测试（2026-11-23 线上问题）：

1. 同一天并发调用 process_day 必须互斥，绝不重复转写；
2. 重处理某日时，已 done 文件的分段不得被删除（幂等/作用域写库）；
3. 启动恢复：中断遗留的 processing 行复位为 imported，pending 才能清零；
4. 进度快照：阶段单调推进、含文件计数，供 UI 显示真实进度；
5. 手动触发只入队不另起线程（队列语义：queued / already_queued / already_processing）。
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

import echocard.pipeline as pl
from echocard.audio.vad_asr import Segment
from echocard.pipeline import Pipeline


class FakeEngines:
    provider_used = "cpu"

    def embedding(self, samples) -> np.ndarray:
        v = np.full(192, 1.0 if (len(samples) // 8000) % 2 == 0 else 0.25, dtype=np.float32)
        return v / np.linalg.norm(v)


def _fake_transcribe(engines, wav16k, file_start, detect_music_gaps=True):
    stem = Path(wav16k).stem
    n = 2 if stem.endswith("A") else 3
    segs = [Segment(start_s=i * 2.0, end_s=i * 2.0 + 1.8, text=f"{stem}#{i}", rms_db=-20.0)
            for i in range(n)]
    for s in segs:
        s.audio_tags = ["speech"]
    return segs, []


@pytest.fixture()
def sandbox(monkeypatch):
    from conftest import make_wav

    def fake_to_16k(src, dst, normalize=True):
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(src).read_bytes())
        return dst

    def fake_opus(src, dst, bitrate_kbps=32):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(Path(src).read_bytes()[:2048])
        return Path(dst)

    monkeypatch.setattr(pl, "to_16k_mono", fake_to_16k)
    monkeypatch.setattr(pl, "wav_to_opus", fake_opus)
    monkeypatch.setattr(pl, "transcribe_file", _fake_transcribe)
    return make_wav


def _add_file(db, cfg_audio_root: Path, day: str, stem: str, status="imported",
              seconds=6, make_wav=__import__("conftest").make_wav):
    raw = make_wav(Path(cfg_audio_root) / "raw" / f"{stem}.wav", seconds=seconds, rate=16000)
    cur = db.execute(
        "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, recorded_ts, raw_path, status, imported_at)"
        " VALUES(NULL,'usb',?,?,?,?,?,?,?,'t')",
        (str(raw), "sha" + stem, raw.stat().st_size, day, f"{day} 09:00:00", str(raw), status))
    return cur.lastrowid, raw


# ---------------------------------------------------------------- 1. 并发互斥
def test_concurrent_process_day_is_exclusive(db, cfg, sandbox, tmp_path):
    fid, raw = _add_file(db, tmp_path / "audio", "2026-11-23", "X_A")
    p = Pipeline(db, cfg, engines=FakeEngines())
    results: list[dict] = []

    def run():
        results.append(p.process_day("2026-11-23", keep_raw=True))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 只有一个线程真正执行，其余被互斥拒绝
    ran = [r for r in results if r.get("ok")]
    skipped = [r for r in results if r.get("note") == "already running"]
    assert len(ran) == 1 and len(skipped) == 3, f"并发必须互斥：{results}"
    # 且分段没有被写成多份
    assert db.query_one("SELECT COUNT(*) c FROM audio_segments")["c"] == 2


# ---------------------------------------------------------------- 2. 幂等保留
def test_reprocess_keeps_done_files_segments(db, cfg, sandbox, tmp_path):
    # A 已处理完成；B 是新文件（模拟用户又插了一次卡）
    fid_a, _ = _add_file(db, tmp_path / "audio", "2026-11-23", "X_A")
    fid_b, _ = _add_file(db, tmp_path / "audio", "2026-11-23", "X_B")
    db.execute("UPDATE ingest_files SET status='none' WHERE id=?", (fid_a,))
    p = Pipeline(db, cfg, engines=FakeEngines())

    p.process_day("2026-11-23", keep_raw=True)          # 同时处理 A+B
    a_segs = db.query("SELECT id FROM audio_segments WHERE file_id=?", (fid_a,))
    b_segs = db.query("SELECT id FROM audio_segments WHERE file_id=?", (fid_b,))
    assert len(a_segs) == 2 and len(b_segs) == 3

    # 再把 A 标记回未处理（模拟重跑），只应重写 A，B 的分段必须完好
    db.execute("UPDATE ingest_files SET status='imported' WHERE id=?", (fid_a,))
    r = p.process_day("2026-11-23", keep_raw=True)
    assert r["ok"]
    assert db.query_one(
        "SELECT COUNT(*) c FROM audio_segments WHERE file_id=?", (fid_b,))["c"] == 3, \
        "已 done 文件的转写分段绝不能被误删"
    assert db.query_one(
        "SELECT COUNT(*) c FROM audio_segments WHERE file_id=?", (fid_a,))["c"] == 2


# ---------------------------------------------------------------- 3. 启动恢复
def test_recover_stalled_processing_rows(db, cfg):
    db.execute(
        "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, raw_path, status, imported_at)"
        " VALUES(NULL,'usb','p1','h1',10,'2026-11-23','p1','processing','t')")
    p = Pipeline(db, cfg, engines=FakeEngines())
    assert p.pending_days() == ["2026-11-23"]
    n = p.recover_stalled()
    assert n == 1
    assert db.query_one("SELECT status FROM ingest_files")["status"] == "imported"


# ---------------------------------------------------------------- 4. 进度快照
def test_progress_is_monotonic_and_reports_files(db, cfg, sandbox, tmp_path):
    _add_file(db, tmp_path / "audio", "2026-11-23", "X_A")
    _add_file(db, tmp_path / "audio", "2026-11-23", "X_B")
    p = Pipeline(db, cfg, engines=FakeEngines())
    seen: list[dict] = []
    if p.events is None:
        from echocard.core.events import EventBus

        p.events = EventBus()
    p.events.subscribe("pipeline.stage", lambda d: seen.append(d))

    p.run_auto()
    p.process_day("2026-11-23", keep_raw=True)   # 直接跑（队列语义另测）

    assert seen, "必须发布阶段进度事件"
    pcts = [s["percent"] for s in seen]
    assert max(pcts) >= 90, f"进度应推进到接近完成：{pcts}"
    assert all(b >= a for a, b in zip(pcts, pcts[1:])), f"进度必须单调递增：{pcts}"
    stages = [s["stage"] for s in seen]
    for must in ("transcribe", "embed", "cluster", "write", "opus"):
        assert must in stages, f"缺少阶段 {must}"
    transcribe_events = [s for s in seen if s["stage"] == "transcribe" and s["file_total"]]
    assert transcribe_events and transcribe_events[-1]["file_total"] == 2, \
        "转写阶段必须汇报文件计数"
    assert transcribe_events[-1]["file_index"] == 2, "最后一个文件应完成计数"


# ---------------------------------------------------------------- 5. 入队语义
def test_queue_day_semantics(db, cfg):
    p = Pipeline(db, cfg, engines=FakeEngines())
    r1 = p.queue_day("2026-11-23", keep_raw=True, reason="manual")
    assert r1["status"] == "queued" and r1["position"] == 1
    r2 = p.queue_day("2026-11-23", reason="auto")
    assert r2["status"] == "already_queued" and r2["position"] == 1
    r3 = p.queue_day("2026-11-24", reason="manual")
    assert r3["status"] == "queued" and r3["position"] == 2
    snap = p.queue_snapshot()
    assert [x["day"] for x in snap] == ["2026-11-23", "2026-11-24"]


def test_scan_reports_detail_and_note(db, cfg):
    db.execute(
        "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, raw_path, status, imported_at)"
        " VALUES(NULL,'usb','q1','h1',10,'2026-11-23','q1','imported','t')")
    p = Pipeline(db, cfg, engines=FakeEngines())
    # 绕过模型检查，直接看 run_auto 的扫描结果结构
    days = p.run_auto()
    snap = p.progress_snapshot()
    if days:  # 模型就绪时才会入队
        res = snap["last_scan_result"]
        assert res["days"] == days and res["queued"] >= 1
        assert "2026-11-23" in res["detail"]
        assert res["detail"]["2026-11-23"]["files"] >= 1
        assert res["note"]
    else:
        res = snap["last_scan_result"]
        assert res and "模型未就绪" in res["note"], "模型缺失时必须给出明确提示"


def test_day_detail_reports_counts(db, cfg):
    db.execute(
        "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, raw_path, status, imported_at)"
        " VALUES(NULL,'usb','d1','h1',10,'2026-11-23','d1','done','t')")
    db.execute(
        "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, raw_path, status, imported_at)"
        " VALUES(NULL,'usb','d2','h2',10,'2026-11-23','d2','imported','t')")
    p = Pipeline(db, cfg, engines=FakeEngines())
    d = p.day_detail("2026-11-23")
    assert d["files"] == 2 and d["done"] == 1 and d["pending"] == 1 and d["percent"] == 50
    assert d["historical"] is False
