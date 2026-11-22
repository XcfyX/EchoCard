"""FastAPI 应用装配：状态 / 设备绑定 / 配置 / 上传 API / 最小控制台页面。"""
from __future__ import annotations

import logging
from dataclasses import asdict, fields
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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

log = logging.getLogger(__name__)

# 配置页允许修改的字段白名单（绝不放行任意键写入）
EDITABLE_FIELDS = {
    "data_root", "server_port", "delete_after_verify", "auto_bind_dji",
    "proxy_url", "watched_photo_dirs", "auto_music_recognition", "theme",
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


def create_app(
    db: Database,
    cfg: AppConfig,
    events: EventBus,
    importer: DeviceImporter,
    watcher: PhotoWatcher | None = None,
    monitor: UsbMonitor | None = None,
) -> FastAPI:
    app = FastAPI(title="EchoCard", version=APP_VERSION, docs_url="/api/docs")
    app.state.db = db
    app.state.cfg = cfg
    app.state.events = events
    app.state.importer = importer
    app.state.watcher = watcher
    app.state.monitor = monitor
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
        return asdict(cfg)

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

    # ---- 触发后台清洗流水线（M1 记录事件，M2 起真正执行）----
    @app.post("/api/v1/pipeline/run/{day}")
    def pipeline_run(day: str) -> dict:
        events.publish("pipeline.day_ready", {"day": day, "file_ids": [
            r["id"] for r in db.query(
                "SELECT id FROM ingest_files WHERE recorded_at=?", (day,))]})
        return {"ok": True, "note": "M1 仅记录触发事件；M2 接入本地音频流水线"}

    # ---- 最小控制台（M4 替换为完整 UI）----
    webui = Path(__file__).resolve().parent.parent / "webui"
    app.mount("/", StaticFiles(directory=str(webui), html=True), name="webui")
    return app
