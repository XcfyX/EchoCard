"""日级流水线编排：转码 → VAD/ASR 转写 → 声纹聚类软匹配 → 音乐检测 → Opus 瘦身。

业务规则（拍板补充）：
- 历史归档分界线 auto_transcribe_since：早于该日期的录音只归档保留原 WAV，
  不自动转写；用户可在界面手动触发（keep_raw=True，保留原 WAV）；
- 自动补全：>= 分界线的所有未处理日期，按日期倒序（最新优先）一次性补全；
- 瘦身：转写/声纹/音乐完成后原 WAV → Opus（约 15MB/小时）并删除原始巨型 WAV；
- 反偏见：片段只增不删，audio_tags 只存客观特征。

并发与一致性（2026-11-23 修复）：
- 所有日处理都在单一工作线程串行执行；手动触发只入队，不再另起线程，
  杜绝同一天被两个 process_day 同时跑（此前导致重复转写 + 互相删库）；
- 写库按「本次处理的文件」作用域删除，已 done 的文件分段绝不被误删；
- 保留分段与新分段共存时，新聚类编号从既有最大编号之后顺延，避免 ID 冲突；
- 启动时把中断遗留的 processing 行复位为 imported，pending 才可能清零。
"""
from __future__ import annotations

import json
import logging
import threading
import time
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

# 进度阶段（权重之和 100）
STAGES = [
    ("prepare", "准备引擎", 3),
    ("transcribe", "转码 + 转写", 62),
    ("embed", "声纹提取", 10),
    ("cluster", "说话人聚类", 5),
    ("write", "写入时间轴", 10),
    ("music", "音乐节点", 2),
    ("opus", "压缩瘦身", 8),
]


def _hms(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


_INVALID_FS_CHARS = set('\\/:*?"<>|')


def _safe_name(name: str) -> str:
    """把人名转成安全的文件名前缀（保留中文，剔除文件系统非法字符）。"""
    cleaned = "".join(c for c in name if c not in _INVALID_FS_CHARS).strip()
    return cleaned or "speaker"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Pipeline:
    def __init__(self, db: Database, cfg: AppConfig, events: EventBus | None = None,
                 engines: Engines | None = None):
        self.db = db
        self.cfg = cfg
        self.events = events
        self._engines = engines
        self._queue: list[dict] = []          # [{day, keep_raw, reason, queued_at}]
        self._lock = threading.RLock()
        self._process_lock = threading.Lock()  # 硬保证 process_day 串行
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._current: dict | None = None
        self._progress: dict = {}
        self._last_scan_at: str | None = None
        self._last_scan_result: dict | None = None

    # ---------------------------------------------------------------- 引擎
    @property
    def engines(self) -> Engines:
        if self._engines is None:
            self._engines = get_engines(self.cfg)
        return self._engines

    @property
    def current_day(self) -> str | None:
        with self._lock:
            return self._current["day"] if self._current else None

    # ---------------------------------------------------------------- 状态恢复
    def recover_stalled(self) -> int:
        """把上次异常中断遗留的 processing 行复位为 imported（可被重新处理）。"""
        n = self.db.query_one(
            "SELECT COUNT(*) c FROM ingest_files WHERE status='processing'")["c"]
        if n:
            self.db.execute(
                "UPDATE ingest_files SET status='imported', error=NULL"
                " WHERE status='processing'")
            log.info("启动恢复：%d 个中断遗留的 processing 文件已复位为待处理", n)
            self._publish("pipeline.recovered", {"files": n})
        return n

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
        """兼容旧接口（事件订阅用）：自动补全入队。"""
        self.queue_day(day, reason="auto")

    def queue_day(self, day: str, keep_raw: bool | None = None,
                  reason: str = "auto") -> dict:
        """入队某日处理。返回 {status: queued|already_queued|already_processing, position}。"""
        with self._lock:
            if self._current and self._current["day"] == day:
                return {"status": "already_processing", "day": day}
            if any(item["day"] == day for item in self._queue):
                pos = next(i for i, x in enumerate(self._queue) if x["day"] == day) + 1
                return {"status": "already_queued", "day": day, "position": pos}
            self._queue.append({"day": day, "keep_raw": keep_raw,
                                "reason": reason, "queued_at": _now()})
            pos = len(self._queue)
        log.info("流水线入队：%s（来源 %s，队列位置 %d）", day, reason, pos)
        self._publish("pipeline.queued", {"day": day, "reason": reason, "position": pos})
        return {"status": "queued", "day": day, "position": pos}

    def queue_snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(x, position=i + 1) for i, x in enumerate(self._queue)]

    def _worker_loop(self) -> None:
        last_scan = 0.0
        while not self._stop.is_set():
            with self._lock:
                item = self._queue[0] if self._queue else None
                if item:
                    self._queue.pop(0)
                    self._current = item
            if item is None:
                if time.time() - last_scan >= 120:
                    last_scan = time.time()
                    try:
                        self.run_auto()
                    except Exception:
                        log.exception("流水线自动扫描异常")
                self._stop.wait(5.0)
                continue
            try:
                keep_raw = item.get("keep_raw")
                if keep_raw is None:
                    keep_raw = item["day"] < self.cfg.auto_transcribe_since
                self.process_day(item["day"], keep_raw=keep_raw, reason=item["reason"])
            except Exception:
                log.exception("流水线处理 %s 异常", item["day"])
                self._publish("pipeline.day_failed", {"day": item["day"]})
            finally:
                with self._lock:
                    self._current = None
                    self._progress = {}

    def _publish(self, topic: str, data: dict) -> None:
        if self.events:
            try:
                self.events.publish(topic, data)
            except Exception:
                log.exception("事件发布失败：%s", topic)

    # ---------------------------------------------------------------- 扫描
    def pending_days(self) -> list[str]:
        """>= 分界线、存在未完成文件的日期，按日期倒序（最新优先）。"""
        since = self.cfg.auto_transcribe_since
        rows = self.db.query(
            "SELECT recorded_at AS d, SUM(CASE WHEN status != 'done' THEN 1 ELSE 0 END)"
            " AS pending FROM ingest_files WHERE recorded_at >= ?"
            " GROUP BY recorded_at", (since,)
        )
        return sorted((r["d"] for r in rows if r["pending"] and r["pending"] > 0), reverse=True)

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

        scan_at = _now()
        with self._lock:
            self._last_scan_at = scan_at
        try:
            resolve_model(self.cfg, "sensevoice-int8.onnx")
        except FileNotFoundError:
            with self._lock:
                self._last_scan_result = {
                    "at": scan_at, "days": [], "queued": 0,
                    "note": "本地模型未就绪，已跳过（先运行 scripts/setup_models.py）",
                }
            return []
        days = self.pending_days()
        results = [self.queue_day(d, reason="auto") for d in days]
        queued = sum(1 for r in results if r["status"] == "queued")
        detail = {d: self.day_detail(d) for d in days}
        with self._lock:
            self._last_scan_result = {
                "at": scan_at, "days": days, "queued": queued,
                "already_queued": sum(1 for r in results if r["status"] == "already_queued"),
                "already_processing": sum(
                    1 for r in results if r["status"] == "already_processing"),
                "detail": detail,
                "note": (f"扫描完成：{len(days)} 天存在未处理文件（新入队 {queued} 天）"
                         if days else "扫描完成：全部日期均已处理，无待入队项目"),
            }
        if days:
            self._publish("pipeline.scan_done",
                          {"queued_days": days, "queued": queued, "detail": detail})
        return days

    def day_detail(self, day: str) -> dict:
        """某日的文件进度明细。"""
        row = self.db.query_one(
            "SELECT COUNT(*) files,"
            " SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) done,"
            " SUM(CASE WHEN status='processing' THEN 1 ELSE 0 END) processing,"
            " SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) failed"
            " FROM ingest_files WHERE recorded_at=?", (day,)
        )
        segs = self.db.query_one(
            "SELECT COUNT(*) c FROM audio_segments WHERE day=?", (day,))["c"]
        files = row["files"] or 0
        done = row["done"] or 0
        return {
            "day": day, "files": files, "done": done,
            "processing": row["processing"] or 0, "failed": row["failed"] or 0,
            "pending": files - done,
            "segments": segs,
            "historical": day < self.cfg.auto_transcribe_since,
            "percent": round(done / files * 100) if files else 100,
        }

    def days_detail(self, limit: int = 14) -> list[dict]:
        days = sorted({r["day"] for r in self.db.query(
            "SELECT DISTINCT recorded_at AS day FROM ingest_files")}, reverse=True)[:limit]
        return [self.day_detail(d) for d in days]

    def progress_snapshot(self) -> dict:
        with self._lock:
            p = dict(self._progress)
            cur = dict(self._current) if self._current else None
            queue = [dict(x, position=i + 1) for i, x in enumerate(self._queue)]
            return {"progress": p, "current": cur, "queue": queue,
                    "last_scan_at": self._last_scan_at,
                    "last_scan_result": self._last_scan_result}

    # ---------------------------------------------------------------- 进度
    def _stage(self, stage: str, *, file_index: int = 0, file_total: int = 0,
               file_name: str = "", extra: str = "") -> None:
        """更新阶段进度（单调递增的百分比）。"""
        start = 0
        for name, _label, weight in STAGES:
            if name == stage:
                break
            start += weight
        label = next(lb for n, lb, _w in STAGES if n == stage)
        weight = next(w for n, _lb, w in STAGES if n == stage)
        frac = (file_index / file_total) if file_total and stage == "transcribe" else 0.0
        if stage == "opus" and file_total:
            frac = file_index / file_total
        pct = int(start + weight * frac)
        with self._lock:
            self._progress = {
                "day": (self._current or {}).get("day"),
                "stage": stage, "stage_label": label,
                "percent": min(99, max(0, pct)),
                "file_index": file_index, "file_total": file_total,
                "file_name": file_name, "extra": extra,
                "updated_at": _now(),
                "started_at": (self._progress.get("started_at")
                               or (self._current or {}).get("queued_at") or _now()),
            }
        self._publish("pipeline.stage", {
            "day": (self._current or {}).get("day"), "stage": stage,
            "label": label, "percent": pct,
            "file": file_name, "file_index": file_index, "file_total": file_total,
        })

    # ---------------------------------------------------------------- 单日处理
    def process_day(self, day: str, keep_raw: bool | None = None,
                    reason: str = "auto") -> dict:
        if keep_raw is None:
            keep_raw = day < self.cfg.auto_transcribe_since
        if not self._process_lock.acquire(blocking=False):
            log.warning("%s 已有处理在运行，跳过重复调用", day)
            return {"ok": False, "note": "already running"}
        try:
            return self._process_day(day, keep_raw, reason)
        finally:
            self._process_lock.release()

    def _process_day(self, day: str, keep_raw: bool, reason: str) -> dict:
        rows = self.db.query(
            "SELECT id, raw_path, recorded_ts FROM ingest_files"
            " WHERE recorded_at=? AND status != 'done' ORDER BY id", (day,)
        )
        if not rows:
            return {"ok": True, "note": "no pending files"}
        # 找回原始 WAV：此前瘦身过（原始已删）的文件无法重转，跳过并说明
        runnable = []
        for r in rows:
            p = Path(r["raw_path"]) if r["raw_path"] else None
            if p and p.exists():
                runnable.append(r)
            else:
                log.info("%s 原始 WAV 不存在（可能已瘦身），跳过", r["raw_path"])
        if not runnable:
            self.db.execute(
                "UPDATE ingest_files SET status='done' WHERE recorded_at=?"
                " AND status='processing'", (day,))
            return {"ok": True, "note": "无可用原始 WAV（均已瘦身），已标记完成"}

        with self._lock:
            self._current = {"day": day, "keep_raw": keep_raw, "reason": reason,
                             "queued_at": self._current.get("queued_at") if self._current else _now()}
        self._publish("pipeline.day_start",
                      {"day": day, "files": len(runnable), "reason": reason})
        self._stage("prepare")

        speech_all: list[Segment] = []
        music_all: list[Segment] = []
        file_info: dict[int, tuple[Path, datetime, Path]] = {}
        failed_files: list[str] = []

        try:
            # ---- 1. 转码 + 转写（逐文件汇报进度）----
            total = len(runnable)
            for i, row in enumerate(runnable, 1):
                fid = row["id"]
                raw = Path(row["raw_path"])
                self.db.execute("UPDATE ingest_files SET status='processing' WHERE id=?", (fid,))
                start_dt = self._file_start(row, day)
                self._stage("transcribe", file_index=i - 1, file_total=total,
                            file_name=raw.name)
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
                        (str(e)[:300], fid))
                    self._publish("pipeline.file_failed",
                                  {"day": day, "file": raw.name, "error": str(e)[:200]})
                    failed_files.append(raw.name)
                self._stage("transcribe", file_index=i, file_total=total,
                            file_name=raw.name)

            if not file_info:
                self._publish("pipeline.day_done",
                              {"day": day, "segments": 0, "note": "all files failed"})
                return {"ok": False, "note": "all files failed",
                        "failed_files": failed_files}

            # ---- 2. 声纹 Embedding ----
            self._stage("embed")
            for fid, (wav16k, _s, _r) in file_info.items():
                pcm, rate = read_pcm16(wav16k)
                samples = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
                for seg in [s for s in speech_all if s.file_id == fid and s.duration >= 1.0]:
                    seg.embedding = self.engines.embedding(
                        samples[int(seg.start_s * rate):int(seg.end_s * rate)])

            # ---- 3. 当日全局聚类（同一天所有文件的片段一起聚类，编号唯一）----
            self._stage("cluster")
            fids = list(file_info)
            in_clause = ",".join(str(int(x)) for x in fids)
            # 已保留分段（本次不处理的文件）的最大编号：新编号顺延，避免冲突
            existing_max = self.db.query_one(
                f"SELECT MAX(cluster_id) m FROM audio_segments"
                f" WHERE day=? AND file_id NOT IN ({in_clause})", (day,))["m"] or 0
            groups = diarize.cluster_segments(
                speech_all, self.cfg.cluster_distance_threshold,
                start_id=(existing_max + 1) if existing_max else 1)
            if existing_max:
                log.info("%s 已有保留分段（最大聚类 %d），新聚类编号从 %d 起",
                         day, existing_max, existing_max + 1)

            speakers_view: dict[int, dict] = {}
            for cid in sorted(groups):
                name, score, status = None, None, "unmatched"
                centroid = diarize.cluster_centroid(speech_all, cid)
                if centroid is not None:
                    name, score, status = diarize.soft_match(
                        self.db, centroid,
                        high=self.cfg.speaker_match_high, mid=self.cfg.speaker_match_mid)
                speakers_view[cid] = {
                    "cluster_id": cid, "name": name, "status": status,
                    "score": round(score, 4) if score is not None else None,
                    "display": self._display_name(cid, name, status),
                    "segments": len(groups[cid]),
                }
            reps = {cid: diarize.pick_representative(speech_all, cid, count=3)
                    for cid in speakers_view}

            # ---- 4. 写库：仅替换本次处理文件的分段（已 done 文件的分段保留）----
            self._stage("write")
            self.db.execute(
                f"DELETE FROM audio_segments WHERE day=? AND file_id IN ({in_clause})", (day,))
            emb_store: dict[int, np.ndarray] = {}
            clips_dir = self.cfg.subdir("audio", day, "clips")
            # 清理本次涉及聚类的旧代表切片：重处理会产生新的优质切片，
            # 旧文件留盘会被 UI 当成有效切片播放（表现为「改名后试听错乱/播不出来」）
            stale_ids = set(speakers_view)
            for r in self.db.query(
                    "SELECT DISTINCT cluster_id FROM audio_segments WHERE day=?"
                    f" AND file_id IN ({in_clause}) AND cluster_id IS NOT NULL", (day,)):
                stale_ids.add(r["cluster_id"])
            for cid in stale_ids:
                if cid is None:
                    continue
                for p in clips_dir.glob(f"speaker{cid}_*.wav"):
                    p.unlink(missing_ok=True)
                # 已改名聚类按人名命名的切片同样失效，一并清理
                for p in clips_dir.glob(f"*_{cid}.wav"):
                    p.unlink(missing_ok=True)
            for fid, (wav16k, start_dt, _raw) in file_info.items():
                for seg in sorted([s for s in speech_all if s.file_id == fid],
                                  key=lambda s: s.start_s):
                    st = start_dt + timedelta(seconds=seg.start_s)
                    en = start_dt + timedelta(seconds=seg.end_s)
                    view = speakers_view.get(seg.cluster_id, {})
                    clip_path = None
                    if seg.cluster_id in reps and any(seg is r for r in reps[seg.cluster_id]):
                        idx = next(i for i, r in enumerate(reps[seg.cluster_id], 1) if r is seg)
                        cpath = clips_dir / f"speaker{seg.cluster_id}_{idx}.wav"
                        try:
                            slice_wav(str(wav16k), str(cpath), seg.start_s, seg.end_s)
                            clip_path = str(cpath)
                        except Exception:
                            log.warning("代表切片失败 cluster=%s seg=[%.1f-%.1f]",
                                        seg.cluster_id, seg.start_s, seg.end_s, exc_info=True)
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

            self._merge_embeddings(day, emb_store)

            # ---- 5. 音乐节点（仅替换本次文件，保留其他文件的音乐节点）----
            self._stage("music")
            self.db.execute(
                f"DELETE FROM music_tracks WHERE day=? AND file_id IN ({in_clause})", (day,))
            for fid, (wav16k, start_dt, _raw) in file_info.items():
                fid_music = [m for m in music_all if m.file_id == fid]
                for s, e, _tags in merge_music_candidates(fid_music):
                    self.db.execute(
                        "INSERT INTO music_tracks(file_id, day, start_ts, end_ts,"
                        " audio_path, status) VALUES(?,?,?,?,?,'detected')",
                        (fid, day, _hms(start_dt + timedelta(seconds=s)),
                         _hms(start_dt + timedelta(seconds=e)), str(wav16k)))

            # ---- 6. Opus 瘦身 + 标记完成 ----
            n_opus = len(file_info)
            for i, (fid, (wav16k, _s, raw)) in enumerate(file_info.items(), 1):
                self._stage("opus", file_index=i, file_total=n_opus, file_name=raw.name)
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
            with self._lock:
                self._current = None
                self._progress = {}

        detail = self.day_detail(day)
        self._publish("pipeline.day_done", {
            "day": day, "segments": len(speech_all),
            "speakers": len(speakers_view), "keep_raw": keep_raw,
            "failed_files": failed_files, "detail": detail,
        })
        try:  # 新片段纳入全文检索
            from .core.fts import rebuild_fts

            rebuild_fts(self.db)
        except Exception:
            log.exception("FTS 重建失败")
        return {"ok": True, "day": day, "segments": len(speech_all),
                "speakers": len(speakers_view), "failed_files": failed_files,
                "keep_raw": keep_raw, "detail": detail}

    def _merge_embeddings(self, day: str, emb_store: dict[int, np.ndarray]) -> None:
        """合并声纹 npz：保留仍存在于库中的旧向量 + 本次新增向量。"""
        npz = self.cfg.subdir("audio", day) / "embeddings.npz"
        valid = {int(r["id"]) for r in self.db.query(
            "SELECT id FROM audio_segments WHERE day=?", (day,))}
        ids: dict[int, np.ndarray] = {}
        if npz.exists():
            try:
                data = np.load(npz)
                for i, e in zip(data["ids"], data["embs"]):
                    if int(i) in valid:
                        ids[int(i)] = e
            except Exception:
                log.warning("旧 embeddings.npz 读取失败，按本次结果重建", exc_info=True)
        for i, e in emb_store.items():
            ids[i] = e
        if ids:
            order = sorted(ids)
            np.savez_compressed(npz, ids=np.asarray(order),
                                embs=np.stack([ids[i] for i in order]))

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
    def _refresh_clips_for_name(self, day: str, cluster_id: int, display: str) -> str | None:
        """改名/合并后把代表切片改成按「显示名」命名，保证改名后仍可试听。

        UI 按「日」聚合切片（clips 列表来自 DB segment_path），因此这里同时
        重命名磁盘文件与 DB 记录，二者必须一致，否则播放会 404。
        """
        clips_dir = self.cfg.subdir("audio", day, "clips")
        src_clips = sorted(clips_dir.glob(f"speaker{cluster_id}_*.wav"))
        if not src_clips:
            return None
        safe = _safe_name(display)
        mapping: dict[Path, Path] = {}
        for i, p in enumerate(src_clips, 1):
            dst = clips_dir / f"{safe}_{i}.wav"
            try:
                p.replace(dst)
                mapping[p] = dst
            except OSError:
                log.warning("改名切片失败：%s → %s", p, dst, exc_info=True)
        # 按 DB 里的精确旧路径重写，避免前缀 LIKE 误伤（speaker1_ 会匹配 speaker11_）
        for old, new in mapping.items():
            self.db.execute(
                "UPDATE audio_segments SET segment_path=? WHERE day=? AND segment_path=?",
                (str(new), day, str(old)))
        return str(next(iter(mapping.values()))) if mapping else None

    def rename_speaker(self, day: str, cluster_id: int, new_name: str) -> int:
        """改名即学：更新该日称呼 → 重建代表切片名 → 抽取 Embedding 更新声纹库。"""
        rows = self.db.query(
            "SELECT id FROM audio_segments WHERE day=? AND cluster_id=?", (day, cluster_id))
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
            (new_name, day, cluster_id))
        clip = self._refresh_clips_for_name(day, cluster_id, new_name)
        self._publish("speaker.renamed",
                      {"day": day, "cluster": cluster_id, "name": new_name,
                       "learned": learned, "clip": clip})
        return learned

    def merge_speakers(self, day: str, keep_id: int, gone_id: int,
                       name: str | None = None) -> int:
        """合并被拆散的两个聚类。

        命名优先级：显式 name > keep 侧已有人名 > gone 侧已有人名 > 保持中性编号。
        （此前按调用方传入的 keep_id 固定取名字，导致「合并后变成说话人 N」）
        """
        if name:
            final_name = name
        else:
            # 优先沿用已命名的一侧（人名优先于中性「说话人 X」）
            names: list[tuple[int, str]] = []
            for cid in (keep_id, gone_id):
                row = self.db.query_one(
                    "SELECT speaker_name FROM audio_segments WHERE day=? AND cluster_id=?"
                    " AND speaker_name IS NOT NULL AND speaker_name != ''"
                    " LIMIT 1", (day, cid))
                if row and row["speaker_name"]:
                    names.append((cid, row["speaker_name"]))
            human = [(cid, n) for cid, n in names if not n.startswith("说话人")]
            # 若两侧都已命名且不同，保留片段更多的那个
            if human:
                if len(human) == 2 and human[0][1] != human[1][1]:
                    counts = {cid: self.db.query_one(
                        "SELECT COUNT(*) c FROM audio_segments WHERE day=? AND cluster_id=?",
                        (day, cid))["c"] for cid, _ in human}
                    final_name = max(human, key=lambda x: counts[x[0]])[1]
                else:
                    final_name = human[0][1]
            else:
                final_name = names[0][1] if names else None

        moved = self.merge_speakers_rows(day, keep_id, gone_id, final_name)
        if final_name:
            self._refresh_clips_for_name(day, keep_id, final_name)
            # 合并后以最终名字再学一次声纹（等价于「改名即学」）
            try:
                self.rename_speaker(day, keep_id, final_name)
            except Exception:
                log.warning("合并后声纹学习失败", exc_info=True)
        self._publish("speaker.merged",
                      {"day": day, "keep": keep_id, "gone": gone_id,
                       "moved": moved, "name": final_name})
        return moved

    def merge_speakers_rows(self, day: str, keep_id: int, gone_id: int,
                            display: str | None) -> int:
        """把 gone_id 的分段并入 keep_id（含旧切片清理与称呼统一）。"""
        self.db.execute(
            "UPDATE audio_segments SET cluster_id=? WHERE day=? AND cluster_id=?",
            (keep_id, day, gone_id))
        if display:
            self.db.execute(
                "UPDATE audio_segments SET speaker_name=? WHERE day=? AND cluster_id=?",
                (display, day, keep_id))
        moved = self.db.query_one(
            "SELECT COUNT(*) c FROM audio_segments WHERE day=? AND cluster_id=?",
            (day, keep_id))["c"]
        # 清理被并聚类的切片文件（其内容已由 keep 侧代表）
        clips_dir = self.cfg.subdir("audio", day, "clips")
        for p in clips_dir.glob(f"speaker{gone_id}_*.wav"):
            p.unlink(missing_ok=True)
        return moved
