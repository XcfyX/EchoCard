"""FastAPI 应用装配：状态 / 设备绑定 / 配置 / 上传 API / 最小控制台页面。"""
from __future__ import annotations

import logging
from dataclasses import asdict, fields
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from ..core.config import APP_VERSION, AppConfig
from ..core.db import Database
from ..core.events import EventBus
from ..ingest.dji_import import DeviceImporter
from ..ingest.photo_watch import PhotoWatcher
from ..ingest.usb_monitor import UsbMonitor
from ..ingest.usb_volumes import list_usb_volumes
from ..ingest.device_binding import bind_volume_manually
from ..ingest.lan_upload import router as lan_router
from ..pipeline import Pipeline

log = logging.getLogger(__name__)

# 配置页允许修改的字段白名单（绝不放行任意键写入）
EDITABLE_FIELDS = {
    "data_root", "server_port", "delete_after_verify", "auto_bind_dji",
    "proxy_url", "watched_photo_dirs", "auto_music_recognition", "theme",
    "export_prompt", "auto_transcribe_since", "auto_pipeline",
    "pipeline_provider", "cluster_distance_threshold", "cleanup_after_summary",
}


class ConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 未知字段直接 422，防任意键写入

    data_root: str | None = None
    server_port: int | None = None
    delete_after_verify: bool | None = None
    auto_bind_dji: bool | None = None
    proxy_url: str | None = None
    watched_photo_dirs: list[str] | None = None
    auto_music_recognition: bool | None = None
    theme: str | None = None
    export_prompt: str | None = None
    auto_transcribe_since: str | None = None
    api_base_url: str | None = None
    api_key: str | None = None
    api_model: str | None = None
    auto_pipeline: bool | None = None
    pipeline_provider: str | None = None
    cluster_distance_threshold: float | None = None
    cleanup_after_summary: bool | None = None


class RenameBody(BaseModel):
    name: str


class MergeBody(BaseModel):
    keep_id: int
    gone_id: int
    name: str | None = None


class MusicRangeBody(BaseModel):
    day: str
    start: str
    end: str


class ExportBody(BaseModel):
    days: list[str]


class ReflowTextBody(BaseModel):
    text: str
    day: str | None = None


class TodoToggleBody(BaseModel):
    done: bool


class StarBody(BaseModel):
    starred: bool


class CleanupBody(BaseModel):
    day: str
    keep_unrecognized_music: bool = True


def create_app(
    db: Database,
    cfg: AppConfig,
    events: EventBus,
    importer: DeviceImporter,
    watcher: PhotoWatcher | None = None,
    monitor: UsbMonitor | None = None,
    pipeline: Pipeline | None = None,
) -> FastAPI:
    app = FastAPI(title="EchoCard", version=APP_VERSION, docs_url="/api/docs")
    app.state.db = db
    app.state.cfg = cfg
    app.state.events = events
    app.state.importer = importer
    app.state.watcher = watcher
    app.state.monitor = monitor
    app.state.pipeline = pipeline
    app.include_router(lan_router)

    # ---- 状态 ----
    @app.get("/api/v1/status")
    def status() -> dict:
        devices = [dict(r) for r in db.query("SELECT * FROM devices ORDER BY id DESC")]
        counts = {
            "audio_files": db.query_one("SELECT COUNT(*) c FROM ingest_files")["c"],
            "photos": db.query_one("SELECT COUNT(*) c FROM photos")["c"],
            "days": db.query_one("SELECT COUNT(DISTINCT recorded_at) c FROM ingest_files")["c"],
        }
        return {
            "version": APP_VERSION,
            "data_root": cfg.data_root,
            "delete_after_verify": cfg.delete_after_verify,
            "auto_bind_dji": cfg.auto_bind_dji,
            "devices": devices,
            "counts": counts,
        }

    @app.get("/api/v1/ingest/log")
    def ingest_log(limit: int = 100) -> dict:
        return {"log": events.recent(min(max(limit, 1), 500))}

    # ---- 配置 ----
    @app.get("/api/v1/config")
    def get_config() -> dict:
        data = asdict(cfg)
        data["api_key"] = "••• 已配置" if cfg.api_key else ""  # 绝不回传真实 Key
        return data

    @app.patch("/api/v1/config")
    def patch_config(patch: ConfigPatch) -> dict:
        changed = []
        field_types = {f.name: f.type for f in fields(AppConfig)}
        for key, value in patch.model_dump(exclude_none=True).items():
            if key not in EDITABLE_FIELDS:
                raise HTTPException(400, f"字段不可修改：{key}")
            if key == "delete_after_verify" and value is True:
                log.warning("⚠ 设备清空开关被打开（delete_after_verify=True）")
            setattr(cfg, key, value)
            changed.append(key)
        cfg.save()
        if "watched_photo_dirs" in changed and watcher:
            watcher.reschedule()
        return {"ok": True, "changed": changed}

    # ---- 设备绑定向导 ----
    @app.get("/api/v1/usb/scan")
    def usb_scan() -> dict:
        vols = list_usb_volumes()
        return {"volumes": [
            {"drive": f"{v.drive_letter}:", "label": v.volume_label, "vid": v.vid,
             "serial": v.usbstor_serial, "marker": v.marker_exists(),
             "dji_like": v.looks_like_dji()}
            for v in vols
        ]}

    @app.post("/api/v1/usb/{drive}/bind")
    def usb_bind(drive: str) -> dict:
        letter = drive.rstrip(":")
        vol = next((v for v in list_usb_volumes() if v.drive_letter.upper() == letter.upper()),
                   None)
        if vol is None:
            raise HTTPException(404, f"未找到可移动卷 {drive}")
        decision = bind_volume_manually(vol, db)
        if monitor:
            monitor.scan_once()
        return {"ok": decision.bound, "device_id": decision.device_id, "reason": decision.reason}

    @app.delete("/api/v1/devices/{device_id}")
    def device_unbind(device_id: int) -> dict:
        db.execute("DELETE FROM devices WHERE id=?", (device_id,))
        return {"ok": True}

    # ================= M2：流水线 / 时间轴 / 说话人校准 / 音乐 =================
    import re as _re
    import threading as _threading

    _DAY_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def _require_pipeline() -> Pipeline:
        if pipeline is None:
            raise HTTPException(503, "流水线未启用")
        return pipeline

    def _valid_day(day: str) -> str:
        if not _DAY_RE.match(day or ""):
            raise HTTPException(400, "日期格式应为 YYYY-MM-DD")
        return day

    @app.post("/api/v1/pipeline/scan")
    def pipeline_scan() -> dict:
        days = _require_pipeline().run_auto()
        return {"ok": True, "queued_days": days}

    @app.get("/api/v1/pipeline/status")
    def pipeline_status() -> dict:
        p = _require_pipeline()
        return {
            "pending_days": p.pending_days(),
            "historical_days": p.historical_days(),
            "current_day": p.current_day,
            "auto_transcribe_since": cfg.auto_transcribe_since,
            "provider": p.engines.provider_used,
        }

    @app.post("/api/v1/pipeline/run/{day}")
    def pipeline_run(day: str) -> dict:
        """手动触发某日清洗（历史日期自动保留原 WAV）。"""
        _valid_day(day)
        p = _require_pipeline()
        if p.current_day == day:
            return {"ok": True, "note": "该日期正在处理中"}
        _threading.Thread(target=p.process_day, args=(day,),
                          kwargs={"keep_raw": True}, daemon=True).start()
        return {"ok": True, "note": "已启动（keep_raw=True，历史归档保留原 WAV）"}

    @app.get("/api/v1/day/{day}/timeline")
    def day_timeline(day: str) -> dict:
        _valid_day(day)
        segments = [dict(r) for r in db.query(
            "SELECT * FROM audio_segments WHERE day=? ORDER BY start_offset", (day,))]
        music = [dict(r) for r in db.query(
            "SELECT * FROM music_tracks WHERE day=? ORDER BY start_ts", (day,))]
        photos = [dict(r) for r in db.query(
            "SELECT * FROM photos WHERE captured_at=? ORDER BY captured_ts", (day,))]
        files = [dict(r) for r in db.query(
            "SELECT id, status, size_bytes, compressed_path, raw_path, error, deleted_on_device"
            " FROM ingest_files WHERE recorded_at=?", (day,))]
        # 说话人视图（当日聚类去重 + 代表切片）
        speakers: dict[int, dict] = {}
        for seg in db.query(
            "SELECT cluster_id, speaker_name, speaker_match_score, segment_path"
            " FROM audio_segments WHERE day=? AND cluster_id IS NOT NULL", (day,)
        ):
            cid = seg["cluster_id"]
            view = speakers.setdefault(cid, {
                "cluster_id": cid, "display": seg["speaker_name"],
                "score": seg["speaker_match_score"], "clips": [], "segments": 0})
            view["segments"] += 1
            if seg["segment_path"] and seg["segment_path"] not in view["clips"]:
                view["clips"].append(seg["segment_path"])
        return {"day": day, "segments": segments, "music": music,
                "photos": photos, "files": files,
                "speakers": sorted(speakers.values(), key=lambda v: v["cluster_id"]),
                "summary": None}

    @app.post("/api/v1/day/{day}/speakers/{cluster_id}/rename")
    def speaker_rename(day: str, cluster_id: int, body: RenameBody) -> dict:
        _valid_day(day)
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "新名字不能为空")
        learned = _require_pipeline().rename_speaker(day, cluster_id, name)
        return {"ok": True, "learned_samples": learned}

    @app.post("/api/v1/day/{day}/speakers/merge")
    def speaker_merge(day: str, body: MergeBody) -> dict:
        _valid_day(day)
        moved = _require_pipeline().merge_speakers(day, body.keep_id, body.gone_id, body.name)
        return {"ok": True, "moved": moved}

    @app.get("/api/v1/audio/segment/{segment_id}")
    def audio_segment(segment_id: int) -> FileResponse:
        seg = db.query_one(
            "SELECT s.*, f.raw_path AS f_raw, f.compressed_path AS f_opus"
            " FROM audio_segments s JOIN ingest_files f ON f.id = s.file_id"
            " WHERE s.id=?", (segment_id,))
        if seg is None:
            raise HTTPException(404, "片段不存在")
        src = seg["f_raw"] if seg["f_raw"] else seg["f_opus"]
        if not src:
            raise HTTPException(404, "音频源文件不存在（可能已被瘦身，请试听代表切片）")
        from ..audio.transcode import slice_wav

        import os as _os
        import tempfile as _tempfile

        tmp = Path(_tempfile.gettempdir()) / f"echocard-seg-{segment_id}-{_os.getpid()}.wav"
        slice_wav(src, tmp, seg["start_offset"], seg["end_offset"])
        from starlette.background import BackgroundTask

        return FileResponse(tmp, media_type="audio/wav",
                            background=BackgroundTask(lambda p=str(tmp): _os.unlink(p)))

    @app.get("/api/v1/music/day/{day}")
    def music_day(day: str) -> dict:
        _valid_day(day)
        return {"music": [dict(r) for r in db.query(
            "SELECT * FROM music_tracks WHERE day=? ORDER BY start_ts", (day,))]}

    @app.post("/api/v1/music/track/{track_id}/recognize")
    def music_track_recognize(track_id: int) -> dict:
        row = db.query_one("SELECT * FROM music_tracks WHERE id=?", (track_id,))
        if row is None:
            raise HTTPException(404, "音乐节点不存在")
        from ..audio.music_id import recognize_time_range

        return recognize_time_range(db, cfg, row["day"], row["start_ts"], row["end_ts"])

    @app.post("/api/v1/music/range")
    def music_range(body: MusicRangeBody) -> dict:
        _valid_day(body.day)
        from ..audio.music_id import recognize_time_range

        return recognize_time_range(db, cfg, body.day, body.start, body.end)

    # ================= M3：导出投喂包 / 回流 / 检索 / 清理 / API 总结 =================
    @app.post("/api/v1/export")
    def export_feedpack(body: ExportBody) -> dict:
        from ..export.feedpack import export_days

        for d in body.days:
            _valid_day(d)
        return export_days(db, cfg, body.days)

    @app.get("/api/v1/export/preview")
    def export_preview(days: str) -> dict:
        """days=2026-11-22,2026-11-23 → 返回投喂 Markdown 预览与一键复制文本。"""
        from ..export.feedpack import build_markdown

        day_list = [d.strip() for d in days.split(",") if d.strip()]
        for d in day_list:
            _valid_day(d)
        md, _photos = build_markdown(db, cfg, day_list)
        return {"markdown": md, "days": day_list}

    @app.post("/api/v1/reflow/zip")
    async def reflow_zip(request: Request) -> dict:
        from ..export.reflow import import_from_zip
        from ..export.cleanup import auto_cleanup_if_enabled

        body = await request.body()
        try:
            result = import_from_zip(db, cfg, body)
        except Exception as e:
            raise HTTPException(400, f"回流解析失败：{e}")
        auto_cleanup_if_enabled(db, cfg, result["day"])
        return result

    @app.post("/api/v1/reflow/md")
    def reflow_md_file(request: Request) -> dict:
        from ..export.reflow import import_summary
        from ..export.cleanup import auto_cleanup_if_enabled

        text = (request.query_params.get("text")
                or request.headers.get("X-Reflow-Text") or "")
        if not text:
            raise HTTPException(400, "缺少 Markdown 文本")
        result = import_summary(db, cfg, text, source="reflow_md")
        auto_cleanup_if_enabled(db, cfg, result["day"])
        return result

    @app.post("/api/v1/reflow/text")
    def reflow_text(body: ReflowTextBody) -> dict:
        """通道 3：Ctrl+V 粘贴 AI 回复文本。"""
        from ..export.reflow import import_summary
        from ..export.cleanup import auto_cleanup_if_enabled

        if not body.text.strip():
            raise HTTPException(400, "粘贴内容为空")
        text = body.text
        if body.day:
            _valid_day(body.day)
            if "date:" not in text:
                text = f"---\ndate: {body.day}\n---\n" + text
        result = import_summary(db, cfg, text, source="reflow_paste")
        auto_cleanup_if_enabled(db, cfg, result["day"])
        return result

    @app.get("/api/v1/summaries")
    def summaries_list() -> dict:
        return {"summaries": [dict(r) for r in db.query(
            "SELECT id, day_range, tags, todos_json, facts, music, source, imported_at"
            " FROM daily_summaries ORDER BY day_range DESC")]}

    @app.get("/api/v1/summary/{summary_id}")
    def summary_detail(summary_id: int) -> dict:
        row = db.query_one("SELECT * FROM daily_summaries WHERE id=?", (summary_id,))
        if row is None:
            raise HTTPException(404, "总结不存在")
        data = dict(row)
        data["todos"] = [dict(t) for t in db.query(
            "SELECT * FROM todos WHERE summary_id=? ORDER BY id", (summary_id,))]
        return data

    @app.post("/api/v1/todos/{todo_id}/toggle")
    def todo_toggle(todo_id: int, body: TodoToggleBody) -> dict:
        db.execute("UPDATE todos SET done=? WHERE id=?",
                   (1 if body.done else 0, todo_id))
        return {"ok": True}

    @app.get("/api/v1/search")
    def global_search(q: str, limit: int = 30) -> dict:
        from ..core.fts import search as fts_search

        return {"results": fts_search(db, q, limit)}

    @app.post("/api/v1/segments/{segment_id}/star")
    def segment_star(segment_id: int, body: StarBody) -> dict:
        db.execute("UPDATE audio_segments SET starred=? WHERE id=?",
                   (1 if body.starred else 0, segment_id))
        return {"ok": True}

    @app.post("/api/v1/cleanup")
    def cleanup_day_api(body: CleanupBody) -> dict:
        _valid_day(body.day)
        from ..export.cleanup import cleanup_day

        return cleanup_day(db, cfg, body.day, body.keep_unrecognized_music)

    @app.get("/api/v1/llm/status")
    def llm_status() -> dict:
        from ..llm_client import api_ready

        return {"available": api_ready(cfg), "model": cfg.api_model,
                "base_url": cfg.api_base_url}

    @app.post("/api/v1/llm/summary/{day}")
    def llm_summary(day: str) -> dict:
        """可选 API 一键总结：发送当日投喂内容 + 压缩照片 → 自动回流入库。"""
        _valid_day(day)
        from ..export.feedpack import build_markdown
        from ..export.cleanup import auto_cleanup_if_enabled
        from ..export.reflow import import_summary
        from ..llm_client import chat, api_ready
        import io as _io
        from PIL import Image as _Image

        if not api_ready(cfg):
            raise HTTPException(400, "API Key 未配置（设置页填写 Base URL / Key / 模型）")
        md, _refs = build_markdown(db, cfg, [day])
        photos = [Path(r["compressed_path"]) for r in db.query(
            "SELECT compressed_path FROM photos WHERE captured_at=?"
            " AND include_in_export=1 ORDER BY captured_ts", (day,))
            if Path(r["compressed_path"]).exists()]
        reply = chat(cfg, md, images=photos)
        result = import_summary(db, cfg, reply, source="api")
        auto_cleanup_if_enabled(db, cfg, day)
        return {"ok": True, "day": day, "summary_id": result["summary_id"],
                "reply_chars": len(reply), "photos_sent": len(photos[:12])}

    # ---- 最小控制台（M4 替换为完整 UI）----
    webui = Path(__file__).resolve().parent.parent / "webui"
    app.mount("/", StaticFiles(directory=str(webui), html=True), name="webui")
    return app
