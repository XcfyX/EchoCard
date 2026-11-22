"""USB 可移动卷枚举与认亲信息采集（全程只读，绝不写入未绑定设备）。

实测锚点（DJI Mic 2 TX2，序列号 0123456798AB，2026-11-22）：
  USBSTOR\\DISK&VEN___DJI&PROD_WIRELESS_MIC_TX2&REV_0.01\\0123456798AB&0
  PnP 父设备：USB\\VID_10D6&PID_B00B\\0123456798AB
  → 真实 VID 为 Actions 半导体 0x10D6（与规格书的 0x2CA3 不一致），
    故认亲采用多信号联合判断：标记文件 ∨ VID ∨ USBSTOR 厂商/产品串 ∨ 已绑定序列号。
"""
from __future__ import annotations

import ctypes
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..core.config import DEVICE_MARKER_NAME, DJI_VID_CANDIDATES

log = logging.getLogger(__name__)

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
DRIVE_REMOVABLE = 2

_PNP_USB_RE = re.compile(r"^USB\\VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})\\(.+)$")


@dataclass
class VolumeInfo:
    """一个可移动卷的认亲信息快照。"""

    drive_letter: str            # 'E'
    root: str                    # 'E:\\' 或测试注入的模拟目录
    volume_label: str = ""
    drive_type: int = DRIVE_REMOVABLE
    is_usb: bool = True
    vid: str | None = None       # 小写 4 位 hex
    pid: str | None = None
    usbstor_serial: str | None = None
    vendor_desc: str | None = None   # USBSTOR 串中的 VEN_xxx（DJI 显示为 ___DJI）
    product_desc: str | None = None  # USBSTOR 串中的 PROD_xxx
    marker_path: str | None = None   # 设备根目录 .echocard_source 的路径

    def marker_exists(self) -> bool:
        return bool(self.marker_path) and Path(self.marker_path).exists()

    def looks_like_dji(self) -> bool:
        if self.marker_exists():
            return True
        if self.vid and self.vid.lower() in DJI_VID_CANDIDATES:
            return True
        vendor = (self.vendor_desc or "").upper().replace("_", "")
        product = (self.product_desc or "").upper().replace("_", "")
        return "DJI" in vendor or "DJI" in product

    def is_dji_vid(self) -> bool:
        """仅凭硬件 VID 判定（用于设备 kind 归类）。"""
        return bool(self.vid and self.vid.lower() in DJI_VID_CANDIDATES) or "DJI" in (
            (self.vendor_desc or "") + (self.product_desc or "")
        ).upper()

    def stable_key(self) -> str:
        """跨插拔的稳定标识：优先 USBSTOR 序列号。"""
        return self.usbstor_serial or (f"{self.vid}:{self.pid}" if self.vid else self.drive_letter)


def get_drive_type(letter: str) -> int:
    return kernel32.GetDriveTypeW(f"{letter}:\\")


def _volume_label(letter: str) -> str:
    buf = ctypes.create_unicode_buffer(261)
    fs = ctypes.create_unicode_buffer(261)
    ok = kernel32.GetVolumeInformationW(
        f"{letter}:\\", buf, 261, None, None, None, fs, 261
    )
    return buf.value if ok else ""


def _ref_value(wmi_path: str) -> str:
    r"""从 WMI 关联路径串中提取 DeviceID 引用值。

    注意：关联路径（Win32_DiskDriveToDiskPartition 等）中的对象路径采用 WMI 转义——
    反斜杠翻倍（DeviceID="\\.\PHYSICALDRIVE2" 的每个反斜杠写成两个），
    而实例属性 d.DeviceID 是未转义的原始值，必须统一反转义后才能相互匹配。
    """
    m = re.search(r'"(.*)"', wmi_path or "")
    if m:
        return m.group(1).replace("\\\\", "\\")
    tail = (wmi_path or "").rsplit(":", 1)[-1]
    return tail.replace("\\\\", "\\")


def _map_disk_to_letters(wmi) -> dict[str, list[str]]:
    """物理磁盘 DeviceID -> 逻辑盘符列表（Win32_DiskDriveToDiskPartition 级联）。"""
    disk_by_partition: dict[str, str] = {}
    for a in wmi.ExecQuery(
        "SELECT Antecedent,Dependent FROM Win32_DiskDriveToDiskPartition"
    ):
        disk_by_partition[_ref_value(a.Dependent)] = _ref_value(a.Antecedent)
    out: dict[str, list[str]] = {}
    for a in wmi.ExecQuery("SELECT Antecedent,Dependent FROM Win32_LogicalDiskToPartition"):
        disk = disk_by_partition.get(_ref_value(a.Antecedent))
        letter = _ref_value(a.Dependent)  # 形如 'E:'
        if disk and letter:
            out.setdefault(disk, []).append(letter.rstrip(":"))
    return out


def _find_usb_vids(wmi, serials: list[str | None]) -> dict[str, tuple[str, str]]:
    """在 Win32_PnPEntity 中按序列号反查 USB 桥接设备的 VID/PID。"""
    out: dict[str, tuple[str, str]] = {}
    serial_set = {s for s in serials if s}
    if not serial_set:
        return out
    try:
        rows = wmi.ExecQuery("SELECT PNPDeviceID FROM Win32_PnPEntity")
    except Exception:  # pragma: no cover - WMI 异常时降级
        return out
    for r in rows:
        m = _PNP_USB_RE.match(r.PNPDeviceID or "")
        if not m:
            continue
        vid, pid, tail = m.groups()
        serial = tail.split("&")[0]
        if serial in serial_set:
            out[serial] = (vid.lower(), pid.lower())
    return out


def list_usb_volumes(marker_name: str = DEVICE_MARKER_NAME) -> list[VolumeInfo]:
    """枚举当前挂载的 USB 可移动卷。WMI 失败时降级为盘符类型扫描（标记文件仍可认亲）。"""
    disks: list[dict] = []
    letter_by_disk: dict[str, list[str]] = {}
    vid_by_serial: dict[str, tuple[str, str]] = {}
    try:
        import pythoncom
        from win32com.client import GetObject

        pythoncom.CoInitialize()
        wmi = GetObject(r"winmgmts:\\.\root\cimv2")
        for d in wmi.ExecQuery("SELECT * FROM Win32_DiskDrive WHERE InterfaceType='USB'"):
            pnp = d.PNPDeviceID or ""
            disks.append(
                {
                    "device_id": d.DeviceID,
                    "pnp": pnp,
                    "serial": pnp.split("\\")[-1].split("&")[0],
                    "vendor": (re.search(r"VEN_([^&]+)", pnp) or [None, ""])[1] if "VEN_" in pnp else "",
                    "product": (re.search(r"PROD_([^&]+)", pnp) or [None, ""])[1] if "PROD_" in pnp else "",
                }
            )
        letter_by_disk = _map_disk_to_letters(wmi)
        vid_by_serial = _find_usb_vids(wmi, [x["serial"] for x in disks])
    except Exception:
        log.warning("WMI 枚举失败，降级为盘符类型扫描", exc_info=True)

    vols: list[VolumeInfo] = []
    for d in disks:
        for letter in letter_by_disk.get(d["device_id"], []):
            vols.append(
                VolumeInfo(
                    drive_letter=letter,
                    root=f"{letter}:\\",
                    volume_label=_volume_label(letter),
                    drive_type=get_drive_type(letter),
                    is_usb=True,
                    vid=vid_by_serial.get(d["serial"], (None, None))[0],
                    pid=vid_by_serial.get(d["serial"], (None, None))[1],
                    usbstor_serial=d["serial"],
                    vendor_desc=d["vendor"],
                    product_desc=d["product"],
                    marker_path=_marker_if_exists(letter, marker_name),
                )
            )
    if vols:
        return vols

    # 降级路径：逐盘符扫描可移动盘（无 VID/序列号信息，靠标记文件/卷标认亲）
    for code in range(ord("A"), ord("Z") + 1):
        letter = chr(code)
        if get_drive_type(letter) == DRIVE_REMOVABLE:
            vols.append(
                VolumeInfo(
                    drive_letter=letter,
                    root=f"{letter}:\\",
                    volume_label=_volume_label(letter),
                    drive_type=DRIVE_REMOVABLE,
                    is_usb=True,
                    marker_path=_marker_if_exists(letter, marker_name),
                )
            )
    return vols


def _marker_if_exists(letter: str, marker_name: str) -> str | None:
    p = Path(f"{letter}:\\") / marker_name
    return str(p) if p.exists() else None
