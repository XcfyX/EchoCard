"""只读探测当前挂载的 USB 可移动卷与 DJI 设备目录结构。

绝不写入、绝不删除 —— 仅用于人工核对认亲锚点（卷标 / VID / 序列号 / 目录树）。
用法：.venv\\Scripts\\python scripts\\probe_dji.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echocard.core.config import DEVICE_MARKER_NAME, DJI_VID_CANDIDATES
from echocard.ingest.device_binding import recognize_volume
from echocard.core.config import AppConfig
from echocard.core.db import Database  # noqa: F401  (仅在传入临时库时使用)
from echocard.ingest.usb_volumes import list_usb_volumes


def main() -> None:
    vols = list_usb_volumes()
    print(f"发现 {len(vols)} 个 USB 可移动卷：\n")
    for v in vols:
        print(f"  盘符      : {v.drive_letter}:")
        print(f"  根目录    : {v.root}")
        print(f"  卷标      : {v.volume_label!r}")
        print(f"  VID:PID   : {v.vid}:{v.pid}")
        print(f"  序列号    : {v.usbstor_serial}")
        print(f"  厂商/产品 : {v.vendor_desc!r} / {v.product_desc!r}")
        print(f"  标记文件  : {v.marker_path or '无'}")
        print(f"  DJI 特征  : {v.looks_like_dji()}（VID 候选 {sorted(DJI_VID_CANDIDATES)}）")
        if v.looks_like_dji():
            root = Path(v.root)
            print("  目录树（只读展示，最多 60 项）：")
            shown = 0
            for p in sorted(root.rglob("*")):
                if shown >= 60:
                    print("    …")
                    break
                if "$recycle" in p.parts[-1].lower() or p.name.startswith("System Volume"):
                    continue
                size = f"{p.stat().st_size:>12,}B" if p.is_file() else "         dir"
                print(f"    {size}  {p.relative_to(root)}")
                shown += 1
        print()
    # 认亲演练：使用内存数据库，不落盘
    from echocard.core.db import Database

    db = Database(":memory:")
    for v in vols:
        d = recognize_volume(v, db, AppConfig())
        print(f"认亲演练 {v.drive_letter}: → bound={d.bound} reason={d.reason}")
    print(f"\n说明：绑定标记文件为设备根目录的 {DEVICE_MARKER_NAME}（由界面向导写入）。")


if __name__ == "__main__":
    main()
