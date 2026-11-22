"""日级流水线编排：转码 → VAD/ASR 转写 → 声纹聚类软匹配 → 音乐检测 → Opus 瘦身。

业务规则（拍板补充）：
- 历史归档分界线 auto_transcribe_since：早于该日期的录音只归档保留原 WAV，
  不自动转写；用户可在界面手动触发（keep_raw=True，保留原 WAV）；
- 自动补全：>= 分界线的所有未处理日期，按日期倒序（最新优先）一次性补全，
  无论分几天同步进来；
- 瘦身：转写/声纹/音乐完成后原 WAV → Opus（约 15MB/小时）并删除原始巨型 WAV；
- 反偏见：片段只增不删，audio_tags 只存客观特征。
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .core.config import AppConfig
from .core.db import Database
from .core.events import EventBus
from .audio import diarize, speakers
from .audio.engines import Engines, get_engines
from .audio.music_detect import merge_music_candidates
from .audio.transcode import read_pcm16, slice_wav, to_16k_mono, wav_to_opus
from .audio.vad_asr import Segment, transcribe_file

log = logging.getLogger(__name__)


def _hms(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


class Pipeline:
    def __init__(self, db: Database, cfg: AppConfig, events: EventBus | None = None,
                 engines: Engines | None = None):
        self.db = db
        self.cfg = cfg
        self.events = events
        self._engines = engines
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self.current_day: str | None = None

    # ---------------------------------------------------------------- 引擎
    @property
    def engines(self) -> Engines:
        if self._engines is None:
            self._engines = get_engines(self.cfg)
        return self._engines

    # ---------------------------------------------------------------- 调度
    def start_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, name="pipeline", daemon=True)
        self._worker.start()
        log.info("流水线工作线程已启动")

    def stop_worker(self) -> None:
        self._stop.set()

    def enqueue(self, day: str) -> None:
        with self._lock:
            if day not in self._queue:
                self._queue.append(day)

    def _worker_loop(self) -> None:
        import time

        last_scan = 0.0
        while not self._stop.is_set():
            with self._lock:
                day = self._queue.pop(0) if self._queue else None
            if day is None:
                # 每 2 分钟自动扫描补全（覆盖分几天同步进来的场景）
                if time.time() - last_scan >= 120:
                    last_scan = time.time()
                    try:
                        for d in self.run_auto():
                            pass
                    except Exception:
                        log.exception("流水线自动扫描异常")
                self._stop.wait(5.0)
                continue
            try:
                self.process_day(day, keep_raw=False)
            except Exception:
                log.exception("流水线处理 %s 异常", day)
                self._publish("pipeline.day_failed", {"day": day})

    def _publish(self, topic: str, data: dict) -> None:
        if self.events:
            try:
                self.events.publish(topic, data)
            except Exception:
                log.exception("事件发布失败：%s", topic)

    def pending_days(self) -> list[str]:
        """>= 分界线、存在未完成文件的日期，按日期倒序（最新优先）。"""
        since = self.cfg.auto_transcribe_since
        rows = self.db.query(
            "SELECT recorded_at AS d, SUM(CASE WHEN status != 'done' THEN 1 ELSE 0 END)"
            " AS pending FROM ingest_files WHERE recorded_at >= ?"
            " GROUP BY recorded_at", (since,)
        )
        return sorted(
            (r["d"] for r in rows if r["pending"] and r["pending"] > 0),
            reverse=True,
        )

    def historical_days(self) -> list[str]:
        """早于分界线的历史归档日期（保留原 WAV，仅手动触发）。"""
        rows = self.db.query(
            "SELECT DISTINCT recorded_at AS d FROM ingest_files WHERE recorded_at < ?"
            " ORDER BY d DESC", (self.cfg.auto_transcribe_since,)
        )
        return [r["d"] for r in rows]

    def run_auto(self) -> list[str]:
        """全量自动补全：扫描所有未处理日期并入队（最新日期优先出结果）。"""
        from .audio.engines import resolve_model

        try:
            resolve_model(self.cfg, "sensevoice-int8.onnx")
        except FileNotFoundError:
            return []  # 模型未就绪：静默等待 setup_models.py 完成
        days = self.pending_days()
        for d in days:
            self.enqueue(d)
        if days:
            self._publish("pipeline.scan_done", {"queued_days": days})
        return days

    # ---------------------------------------------------------------- 单日处理
    def process_day(self, day: str, keep_raw: bool | None = None) -> dict:
        if keep_raw is None:
            keep_raw = day < self.cfg.auto_transcribe_since  # 历史日手动触发保留原 WAV
        rows = self.db.query(
            "SELECT id, raw_path, recorded_ts FROM ingest_files"
            " WHERE recorded_at=? AND status != 'done' ORDER BY id", (day,)
        )
        if not rows:
            return {"ok": True, "note": "no pending files"}

        self.current_day = day
        self._publish("pipeline.day_start", {"day": day, "files": len(rows)})
        speech_all: list[Segment] = []
        music_all: list[Segment] = []
        file_info: dict[int, tuple[Path, datetime, Path]] = {}  # fid -> (16k, start_dt, raw)
        failed_files: list[str] = []

        try:
            # ---- 1. 转码 + 转写 ----
            for row in rows:
                fid = row["id"]
                raw = Path(row["raw_path"])
                self.db.execute("UPDATE ingest_files SET status='processing' WHERE id=?", (fid,))
                start_dt = self._file_start(row, day)
                try:
                    wav16k = to_16k_mono(
                        raw, self.cfg.subdir("audio", day, "16k") / f"{raw.stem}.wav")
                    speech, music = transcribe_file(self.engines, str(wav16k), start_dt)
                    for s in [*speech, *music]:
                        s.file_id = fid
                    speech_all.extend(speech)
                    music_all.extend(music)
                    file_info[fid] = (wav16k, start_dt, raw)
                except Exception as e:
                    log.exception("文件转写失败：%s", raw)
                    self.db.execute(
                        "UPDATE ingest_files SET status='failed', error=? WHERE id=?",
                        (str(e)[:300], fid),
                    )
                    self._publish("pipeline.file_failed",
                                        {"day": day, "file": raw.name, "error": str(e)[:200]})
                    failed_files.append(raw.name)

            if not file_info:
                self._publish("pipeline.day_done",
                                    {"day": day, "segments": 0, "note": "all files failed"})
                return {"ok": False, "note": "all files failed"}

            # ---- 2. 声纹 Embedding + 当日无监督聚类 ----
            extractor = self.engines.embed()
            for fid, (wav16k, _s, _r) in file_info.items():
                pcm, rate = read_pcm16(wav16k)
                samples = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
                for seg in [s for s in speech_all if s.file_id == fid and s.duration >= 1.0]:
                    seg.embedding = extractor.compute(
                        samples[int(seg.start_s * rate):int(seg.end_s * rate)])
            diarize.cluster_segments(speech_all, self.cfg.cluster_distance_threshold)

            # ---- 3. 说话人视图（软匹配）+ 代表切片 ----
            speakers_view: dict[int, dict] = {}
            for cid in sorted({s.cluster_id for s in speech_all if s.cluster_id}):
                name, score, status = None, None, "unmatched"
                centroid = diarize.cluster_centroid(speech_all, cid)
                if centroid is not None:
                    name, score, status = diarize.soft_match(self.db, centroid)
                speakers_view[cid] = {
                    "cluster_id": cid, "name": name, "status": status,
                    "score": round(score, 4) if score is not None else None,
                    "display": self._display_name(cid, name, status),
                }
            reps = {cid: diarize.pick_representative(speech_all, cid, count=3)
                    for cid in speakers_view}

            # ---- 4. 写库：audio_segments + 代表切片 + 声纹 npz ----
            self.db.execute("DELETE FROM audio_segments WHERE day=?", (day,))
            emb_store: dict[int, np.ndarray] = {}
            clips_dir = self.cfg.subdir("audio", day, "clips")
            for fid, (wav16k, start_dt, _raw) in file_info.items():
                for seg in sorted([s for s in speech_all if s.file_id == fid],
                                  key=lambda s: s.start_s):
                    st = start_dt + timedelta(seconds=seg.start_s)
                    en = start_dt + timedelta(seconds=seg.end_s)
                    view = speakers_view.get(seg.cluster_id, {})
                    clip_path = None
                    if seg.cluster_id in reps and seg in reps[seg.cluster_id]:
                        idx = reps[seg.cluster_id].index(seg) + 1
                        cpath = clips_dir / f"speaker{seg.cluster_id}_{idx}.wav"
                        try:
                            slice_wav(str(wav16k), str(cpath), seg.start_s, seg.end_s)
                            clip_path = str(cpath)
                        except Exception:
                            log.warning("代表切片失败 cluster=%s", seg.cluster_id, exc_info=True)
                    cur = self.db.execute(
                        "INSERT INTO audio_segments(file_id, day, start_ts, end_ts,"
                        " start_offset, end_offset, text, cluster_id, speaker_name,"
                        " speaker_match_score, asr_confidence, audio_tags, segment_path)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (fid, day, _hms(st), _hms(en), seg.start_s, seg.end_s,
                         seg.text, seg.cluster_id, view.get("display"), view.get("score"),
                         None, json.dumps(seg.audio_tags, ensure_ascii=False), clip_path),
                    )
                    if seg.embedding is not None:
                        emb_store[cur.lastrowid] = seg.embedding

            if emb_store:
                npz = self.cfg.subdir("audio", day) / "embeddings.npz"
                np.savez_compressed(
                    npz, ids=np.asarray(sorted(emb_store)),
                    embs=np.stack([emb_store[k] for k in sorted(emb_store)]),
                )

            # ---- 5. 音乐节点（合并候选 → 挂钟起止）----
            self.db.execute("DELETE FROM music_tracks WHERE day=? AND status='detected'", (day,))
            for fid, (wav16k, start_dt, _raw) in file_info.items():
                fid_music = [m for m in music_all if m.file_id == fid]
                for s, e, _tags in merge_music_candidates(fid_music):
                    self.db.execute(
                        "INSERT INTO music_tracks(file_id, day, start_ts, end_ts,"
                        " audio_path, status) VALUES(?,?,?,?,?,'detected')",
                        (fid, day, _hms(start_dt + timedelta(seconds=s)),
                         _hms(start_dt + timedelta(seconds=e)), str(wav16k)),
                    )

            # ---- 6. Opus 瘦身：原始 WAV → Opus →（非历史日）删除原始 WAV ----
            for fid, (wav16k, _s, raw) in file_info.items():
                opus = self.cfg.subdir("audio", day, "opus") / f"{raw.stem}.opus"
                wav_to_opus(raw, opus)
                self.db.execute(
                    "UPDATE ingest_files SET compressed_path=? WHERE id=?", (str(opus), fid))
                if not keep_raw:
                    raw.unlink(missing_ok=True)  # 已转写完成的本地副本，安全瘦身
                    wav16k.unlink(missing_ok=True)
                self.db.execute(
                    "UPDATE ingest_files SET status='done', error=NULL WHERE id=?", (fid,))
                self._publish("pipeline.file_done",
                                    {"day": day, "file": raw.name, "opus": opus.name})
        finally:
            self.current_day = None

        self._publish("pipeline.day_done", {
            "day": day, "segments": len(speech_all),
            "speakers": len(speakers_view), "keep_raw": keep_raw,
            "failed_files": failed_files,
        })
        return {"ok": True, "day": day, "segments": len(speech_all),
                "speakers": len(speakers_view), "failed_files": failed_files,
                "keep_raw": keep_raw}

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _display_name(cid: int, name: str | None, status: str) -> str:
        if status == "matched" and name:
            return name
        if status == "suspected" and name:
            return f"疑似-{name} (说话人 {cid})"
        return f"说话人 {cid}"

    @staticmethod
    def _file_start(row, day: str) -> datetime:
        if row["recorded_ts"]:
            try:
                return datetime.strptime(row["recorded_ts"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        try:
            return datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            return datetime.now()

    # ---------------------------------------------------------------- UI 联动
    def rename_speaker(self, day: str, cluster_id: int, new_name: str) -> int:
        """改名即学：更新该日称呼 → 抽取该聚类优质切片 Embedding 更新声纹库。"""
        rows = self.db.query(
            "SELECT id FROM audio_segments WHERE day=? AND cluster_id=?", (day, cluster_id)
        )
        seg_ids = [r["id"] for r in rows]
        learned = 0
        npz = self.cfg.subdir("audio", day) / "embeddings.npz"
        if seg_ids and npz.exists():
            data = np.load(npz)
            id2idx = {int(i): k for k, i in enumerate(data["ids"])}
            embs = [data["embs"][id2idx[i]] for i in seg_ids if i in id2idx]
            learned = speakers.learn_from_cluster(self.db, embs, new_name)
        self.db.execute(
            "UPDATE audio_segments SET speaker_name=? WHERE day=? AND cluster_id=?",
            (new_name, day, cluster_id),
        )
        self._publish("speaker.renamed",
                            {"day": day, "cluster": cluster_id, "name": new_name,
                             "learned": learned})
        return learned

    def merge_speakers(self, day: str, keep_id: int, gone_id: int,
                       name: str | None = None) -> int:
        """合并被拆散的两个聚类，可选合并后命名并学习。"""
        self.db.execute(
            "UPDATE audio_segments SET cluster_id=? WHERE day=? AND cluster_id=?",
            (keep_id, day, gone_id),
        )
        moved = self.db.query_one(
            "SELECT COUNT(*) c FROM audio_segments WHERE day=? AND cluster_id=?",
            (day, keep_id),
        )["c"]
        learned = self.rename_speaker(day, keep_id, name) if name else 0
        self._publish("speaker.merged",
                            {"day": day, "keep": keep_id, "gone": gone_id, "moved": moved})
        return learned
