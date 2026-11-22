"""局域网扫码传图页 + 通用硬件接入 REST API。

通道 1（手机扫码）：GET /upload 简洁上传页，批量勾选、自动识别拍摄日期。
通道 2（未来 ESP32-S3 等硬件）：规范 REST 接口零改动接入：
  POST /api/v1/ingest/audio   （multipart：file + captured_at/device_id/bookmark）
  POST /api/v1/ingest/photo   （multipart：file + captured_at/device_id/bookmark）
"""
from __future__ import annotations

import io
import logging
import socket
from typing import Optional

import qrcode
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ..core.utils import now_iso, parse_dt, today_str
from .photo_ingest import ingest_photo_bytes

log = logging.getLogger(__name__)

router = APIRouter()


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("223.5.5.5", 80))  # 仅取本机出口 IP，不发包
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def _get_or_create_api_device(request: Request, device_id: str | None) -> int:
    db = request.app.state.db
    serial = (device_id or "default-device").strip() or "default-device"
    row = db.query_one("SELECT id FROM devices WHERE kind='lan_api' AND serial=?", (serial,))
    if row:
        return row["id"]
    cur = db.execute(
        "INSERT INTO devices(kind, serial, bound_at, last_seen, note)"
        " VALUES('lan_api', ?, ?, ?, ?)",
        (serial, now_iso(), now_iso(), "REST API 接入设备"),
    )
    return cur.lastrowid


@router.get("/api/v1/lan/qr")
def lan_qr(request: Request) -> Response:
    url = f"http://{_lan_ip()}:{request.app.state.cfg.server_port}/upload"
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@router.get("/upload")
def upload_page() -> HTMLResponse:
    return HTMLResponse(UPLOAD_PAGE)


@router.post("/api/v1/ingest/photo")
async def ingest_photo(
    request: Request,
    file: UploadFile = File(...),
    captured_at: Optional[str] = Form(default=None),
    device_id: Optional[str] = Form(default=None),
    bookmark: Optional[str] = Form(default=None),
) -> JSONResponse:
    cfg = request.app.state.cfg
    db = request.app.state.db
    data = await file.read()
    dev_id = _get_or_create_api_device(request, device_id)
    res = ingest_photo_bytes(
        db, cfg, data, source="lan", original_name=file.filename or "photo.jpg",
        captured_at=captured_at, device_id=dev_id, bookmark=bookmark,
    )
    payload = {"ok": res.ok, "duplicate": res.duplicate, "day": res.day,
               "id": res.photo_id, "reason": res.reason}
    request.app.state.events.publish("ingest.photo_done",
                                     {"name": file.filename, **payload})
    return JSONResponse(payload)


@router.post("/api/v1/ingest/audio")
async def ingest_audio(
    request: Request,
    file: UploadFile = File(...),
    captured_at: Optional[str] = Form(default=None),
    device_id: Optional[str] = Form(default=None),
    bookmark: Optional[str] = Form(default=None),
) -> JSONResponse:
    import hashlib

    cfg = request.app.state.cfg
    db = request.app.state.db
    name = file.filename or "audio.wav"
    if not name.lower().endswith(".wav"):
        return JSONResponse({"ok": False, "reason": "音频仅支持 .wav（16kHz 单声道最优）"},
                            status_code=400)
    data = await file.read()
    if not data:
        return JSONResponse({"ok": False, "reason": "空文件"}, status_code=400)
    sha = hashlib.sha256(data).hexdigest()
    if db.query_one("SELECT 1 FROM ingest_files WHERE src_sha256=?", (sha,)):
        return JSONResponse({"ok": True, "duplicate": True, "reason": "内容哈希已存在"})

    dt = parse_dt(captured_at)
    fallback = dt is None or dt.year < 2023
    day = dt.date().isoformat() if dt and not fallback else today_str()
    dev_id = _get_or_create_api_device(request, device_id)
    raw_dir = cfg.subdir("audio", day, "raw")
    dst = raw_dir / name
    i = 1
    while dst.exists():
        dst = raw_dir / f"{dst.stem}-{i}{dst.suffix}"
        i += 1
    dst.write_bytes(data)
    cur = db.execute(
        "INSERT INTO ingest_files(device_id, source, src_path, src_sha256, size_bytes,"
        " recorded_at, recorded_ts, date_fallback, raw_path, status, bookmark, imported_at)"
        " VALUES(?,?,?,?,?,?,?,? ,?,'imported',?,?)",
        (dev_id, "lan_api", str(name), sha, len(data), day,
         dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None,
         1 if fallback else 0, str(dst), bookmark, now_iso()),
    )
    payload = {"ok": True, "duplicate": False, "day": day, "id": cur.lastrowid}
    request.app.state.events.publish("ingest.audio_done", {"name": name, **payload})
    request.app.state.events.publish("pipeline.day_ready", {"day": day, "file_ids": [cur.lastrowid]})
    return JSONResponse(payload)


UPLOAD_PAGE = """<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EchoCard 传图</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:#0f1115; color:#e6e8ee;
         display:flex; justify-content:center; padding:24px; }
  .card { width:100%; max-width:560px; background:#171a21; border:1px solid #262b36;
          border-radius:16px; padding:24px; box-shadow:0 8px 30px rgba(0,0,0,.4); }
  h1 { font-size:20px; margin:0 0 4px; } p.sub { color:#8b93a5; font-size:13px; margin:0 0 18px; }
  #drop { border:2px dashed #3a4152; border-radius:12px; padding:28px; text-align:center;
          color:#8b93a5; transition:.15s; }
  #drop.hot { border-color:#6c8cff; color:#aebbff; background:rgba(108,140,255,.06); }
  input[type=file] { display:none; }
  label.date { display:block; margin:14px 0 6px; font-size:13px; color:#8b93a5; }
  input[type=date] { background:#10131a; color:#e6e8ee; border:1px solid #2a3040;
                     border-radius:8px; padding:8px 10px; width:100%; box-sizing:border-box; }
  ul { list-style:none; padding:0; margin:14px 0 0; font-size:13px; }
  li { padding:6px 8px; border-radius:8px; margin-bottom:6px; background:#10131a; }
  li.ok { color:#5dd39e; } li.dup { color:#d9b44a; } li.err { color:#e5645d; }
  button { margin-top:14px; width:100%; padding:12px; border:0; border-radius:10px;
           background:#5b7cfa; color:#fff; font-size:15px; }
  button:disabled { opacity:.5; }
</style></head>
<body><div class="card">
  <h1>EchoCard 传图</h1>
  <p class="sub">照片将按拍摄时间自动归档到对应日期 · 仅限局域网</p>
  <div id="drop">点击选择或拖入照片（可多选）</div>
  <input id="files" type="file" multiple accept="image/*">
  <label class="date">指定拍摄日期（留空则自动读取 EXIF / 文件时间）</label>
  <input id="cdate" type="date">
  <ul id="list"></ul>
  <button id="go">上传</button>
</div>
<script>
const drop=document.getElementById('drop'),fi=document.getElementById('files'),
      btn=document.getElementById('go'),list=document.getElementById('list');
drop.onclick=()=>fi.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hot')};
drop.ondragleave=()=>drop.classList.remove('hot');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hot');
  fi.files=e.dataTransfer.files;};
btn.onclick=async()=>{
  btn.disabled=true;
  const date=document.getElementById('cdate').value;
  for(const f of fi.files){
    const li=document.createElement('li');li.textContent=f.name+' … 上传中';list.appendChild(li);
    const fd=new FormData();fd.append('file',f);
    if(date)fd.append('captured_at',date);
    try{
      const r=await fetch('/api/v1/ingest/photo',{method:'POST',body:fd});
      const j=await r.json();
      if(j.ok&&j.duplicate) {li.className='dup';li.textContent=f.name+' ↷ 重复已跳过';}
      else if(j.ok){li.className='ok';li.textContent=f.name+' ✓ 已归档至 '+j.day;}
      else{li.className='err';li.textContent=f.name+' ✕ '+(j.reason||'失败');}
    }catch(e){li.className='err';li.textContent=f.name+' ✕ 网络错误';}
  }
  btn.disabled=false;
};
</script></body></html>"""
