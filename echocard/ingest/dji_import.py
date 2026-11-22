"""DJI Mic 2（及任意已绑定录音设备）单文件原子导入状态机。

数据零丢失红线流程（对每个 .wav 文件）：
  认亲已过 → 递归扫描（仅 *.wav，跳过隐藏/系统目录）
  → 时间归类（文件名内嵌时间戳 > mtime；早于 %d 年兜底归入当日，按文件序保序）
  → 边拷贝边算 SHA-256（落盘目录内同卷临时文件 + fsync + os.replace 原子落位）
  → 大小一致性校验 → SQLite 落库（内容哈希唯一约束去重）
  → 仅当 校验/落库全部成功 且 delete_after_verify=True 才删除设备上的该单个源文件

任何一步失败/中断：清理临时文件、保留源文件、绝不误删。
绝不格式化整盘、绝不触碰设备上的隐藏或系统文件。
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from ..core.config import AppConfig
from ..core.db import Database
from ..core.events import EventBus
from ..core.utils import now_iso
from .usb_volumes import VolumeInfo

log = logging.getLogger(__name__)

# DJI 录音文件命名（实测）：DJI_42_20261122_212830.WAV —— 内嵌权威录制时间戳
WAV_TIME_PAT = re.compile(r"DJI_\d+_(\d{8})_(\d{6})", re.IGNORECASE)
# 设备时间漂移判定下限（规格书：早于 2023 年视为漂移）
MIN_SANE_YEAR = 2023
CHUNK = 1024 * 1024
# 设备上的系统/回收目录：扫描时直接跳过，绝不进入
SKIP_DIRS = {"system volume information", "$recycle.bin", "recycler", "recycled"}


@dataclass
class ImportReport:
    imported: int = 0
    skipped_dedup: int = 0
    failed: int = 0
    deleted_on_device: int = 0
    errors: list[str] = field(default_factory=list)
    day_files: dict[str, list[int]] = field(default_factory=dict)  # 日期 -> ingest_file ids

    def summary(self) -> str:
        return (
            f"导入 {self.imported} / 去重跳过 {self.skipped_dedup} / 失败 {self.failed} / "
            f"已清空源文件 {self.deleted_on_device}"
        )


class DeviceImporter:
    def __init__(self, db: Database, cfg: AppConfig, events: EventBus | None = None):
        self.db = db
        self.cfg = cfg
        self.events = events

    # ------------------------------------------------------------------ 入口
    def import_volume(self, vol: VolumeInfo) -> ImportReport | None:
        """导入一个已挂载卷。未通过认亲 → 返回 None，绝不产生任何读写。"""
        from .device_binding import recognize_volume  # 局部导入避免环

        decision = recognize_volume(vol, self.db, self.cfg)
        self._publish("usb.device_seen", {
            "drive": f"{vol.drive_letter}:", "bound": decision.bound,
            "reason": decision.reason, "serial": vol.usbstor_serial,
        })
        if not decision.bound:
            log.info("忽略未认亲可移动卷 %s:（%s）—— 绝不触碰",
                     vol.drive_letter, decision.reason)
            return None

        report = ImportReport()
        root = Path(vol.root)
        try:
            wavs = self._scan_wavs(root)
        except OSError as e:
            report.errors.append(f"扫描失败：{e}")
            return report
        if not wavs:
            log.info("%s: 上未发现新的 WAV 文件", vol.drive_letter)
            self._publish("ingest.volume_done", {"drive": vol.drive_letter, "summary": "无新增 WAV"})
            return report

        log.info("开始导入 %s: 上 %d 个 WAV 文件（设备清空开关=%s）",
                 vol.drive_letter, len(wavs), self.cfg.delete_after_verify)
        for src in wavs:
            try:
                self._import_one(src, decision.device_id, report)
            except OSError as e:
                report.failed += 1
                report.errors.append(f"{src.name}: {e}")
                if not root.exists():
                    report.errors.append("检测到设备断开，终止本次导入（已导入部分安全落库，"
                                         "未完成文件保留在设备上）")
                    break
        self._publish("ingest.volume_done", {
            "drive": vol.drive_letter, "summary": report.summary(),
            "imported": report.imported, "skipped": report.skipped_dedup,
            "failed": report.failed, "deleted": report.deleted_on_device,
            "errors": report.errors,
        })
        for day in report.day_files:
            self._publish("pipeline.day_ready", {"day": day, "file_ids": report.day_files[day]})
        return report

    # ------------------------------------------------------------------ 扫描
    def _scan_wavs(self, root: Path) -> list[Path]:
        out: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in SKIP_DIRS and not d.startswith(".")
            ]
            for fn in filenames:
                if fn.startswith("."):  # 隐藏文件
                    continue
                if fn.lower().endswith(".wav"):
                    out.append(Path(dirpath) / fn)
        # 文件名内嵌时间优先；无时间戳的按文件名序保序
        out.sort(key=lambda p: (self._parse_name_time(p) or datetime.min, p.name))
        return out

    @staticmethod
    def _parse_name_time(p: Path) -> datetime | None:
        m = WAV_TIME_PAT.search(p.name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None

    def _recorded_at(self, src: Path) -> tuple[str, str | None, bool]:
        """返回 (归属日期, 原始时间戳, 是否兜底)。"""
        named = self._parse_name_time(src)
        if named and named.year >= MIN_SANE_YEAR:
            return named.date().isoformat(), named.strftime("%Y-%m-%d %H:%M:%S"), False
        try:
            mt = datetime.fromtimestamp(src.stat().st_mtime)
        except OSError:
            mt = None
        if mt and mt.year >= MIN_SANE_YEAR:
            return mt.date().isoformat(), mt.strftime("%Y-%m-%d %H:%M:%S"), False
        return date.today().isoformat(), None, True  # 时间漂移兜底归入当日

    # ------------------------------------------------------------------ 单文件
    def _import_one(self, src: Path, device_id: int | None, report: ImportReport) -> None:
        day, ts, fallback = self._recorded_at(src)
        try:
            src_size = src.stat().st_size
        except OSError as e:
            report.failed += 1
            report.errors.append(f"{src.name}: 无法读取文件大小（{e}）")
            return
        if src_size == 0:
            report.failed += 1
            report.errors.append(f"{src.name}: 零字节损坏文件，保留源文件不删除")
            return

        raw_dir = self.cfg.subdir("audio", day, "raw")
        dst = self._dest_path(raw_dir, src.name)
        tmp = dst.parent / f".tmp-{os.getpid()}-{src.name}"
        try:
            sha, copied = self._copy_with_hash(src, tmp)
            if copied != src_size:
                report.failed += 1
                report.errors.append(
                    f"{src.name}: 大小校验不一致（源 {src_size}B / 拷贝 {copied}B），保留源文件"
                )
                return
            if self.db.query_one("SELECT 1 FROM ingest_files WHERE src_sha256=?", (sha,)):
                report.skipped_dedup += 1
                log.info("%s 内容哈希已存在，跳过重复导入", src.name)
                return
            os.replace(tmp, dst)  # 同卷原子落位
            try:
                cur = self.db.execute(
                    "INSERT INTO ingest_files(device_id, source, src_path, src_sha256,"
                    " size_bytes, recorded_at, recorded_ts, date_fallback, raw_path,"
                    " status, imported_at) VALUES(?,?,?,?,?,?,?,?,?,'imported',?)",
                    (device_id, "usb", str(src), sha, src_size, day, ts,
                     1 if fallback else 0, str(dst), now_iso()),
                )
            except Exception:
                dst.unlink(missing_ok=True)  # 落库失败则回滚文件落位
                raise
            fid = cur.lastrowid
            report.day_files.setdefault(day, []).append(fid)
            report.imported += 1
            log.info("已导入：%s → %s（SHA-256 前 8 位 %s）", src.name, dst, sha[:8])
        finally:
            tmp.unlink(missing_ok=True)  # 任何未落位路径都清理临时

        # ---- 删除闸门：校验 + 落库全部成功 + 红线开关开启 ----
        if self.cfg.delete_after_verify:
            self._safe_delete(src, fid, report)
        else:
            log.info("%s 已导入（设备源文件保留：清空开关关闭）", src.name)

    @staticmethod
    def _dest_path(raw_dir: Path, name: str) -> Path:
        dst = raw_dir / name
        if dst.exists():  # 内容不同但同名的极端情况：追加序号，绝不覆盖
            stem, suffix = dst.stem, dst.suffix
            i = 1
            while True:
                dst = raw_dir / f"{stem}-{i}{suffix}"
                if not dst.exists():
                    return dst
                i += 1
        return dst

    @staticmethod
    def _copy_with_hash(src: Path, dst_tmp: Path) -> tuple[str, int]:
        h = hashlib.sha256()
        copied = 0
        with open(src, "rb") as fsrc, open(dst_tmp, "wb") as fdst:
            while True:
                chunk = fsrc.read(CHUNK)
                if not chunk:
                    break
                fdst.write(chunk)
                h.update(chunk)
                copied += len(chunk)
            fdst.flush()
            os.fsync(fdst.fileno())
        return h.hexdigest(), copied

    # ------------------------------------------------------------------ 删除闸门
    def _safe_delete(self, src: Path, file_id: int, report: ImportReport) -> None:
        row = self.db.query_one(
            "SELECT size_bytes, src_sha256, deleted_on_device FROM ingest_files WHERE id=?",
            (file_id,),
        )
        if row is None or row["deleted_on_device"]:
            report.errors.append(f"{src.name}: 删除闸门拒绝——落库记录缺失")
            return
        try:
            if src.stat().st_size != row["size_bytes"]:
                report.errors.append(f"{src.name}: 删除前大小复核失败，保留源文件")
                return
            src.unlink()
        except OSError as e:
            report.errors.append(f"{src.name}: 删除失败（{e}），保留源文件")
            return
        self.db.execute("UPDATE ingest_files SET deleted_on_device=1 WHERE id=?", (file_id,))
        report.deleted_on_device += 1
        log.info("%s 校验+落库成功，已安全清空该单个源文件", src.name)

    def _publish(self, topic: str, data: dict) -> None:
        if self.events:
            self.events.publish(topic, data)
