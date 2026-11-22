"""EchoCard 全局配置。

隐私红线：
- 运行时生成的 config.json 位于 %APPDATA%/EchoCard/（含代理地址等本地私密配置），
  在公开仓库之外，绝不入库；
- 数据根目录默认 D:\\EchoCardData，录音/照片/数据库/导出包全部收敛于此。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "EchoCard"
APP_VERSION = "0.1.0"

# DJI 系设备 USB VID 候选：
#   0x2CA3 —— 规格书给出的 DJI VID；
#   0x10D6 —— 实测 DJI Mic 2 TX2（Actions 半导体 USB 桥接芯片，
#             PnP 父设备 USB\VID_10D6&PID_B00B\0123456798AB）。
DJI_VID_CANDIDATES = {"2ca3", "10d6"}

# 设备绑定标记文件（放在录音设备根目录，用于向导式一键绑定）
DEVICE_MARKER_NAME = ".echocard_source"


def default_config_path() -> Path:
    base = os.environ.get("ECHOCARD_CONFIG_DIR") or (
        Path(os.environ.get("APPDATA", str(Path.home()))) / APP_NAME
    )
    return Path(base) / "config.json"


@dataclass
class AppConfig:
    data_root: str = r"D:\EchoCardData"
    server_port: int = 8765

    # ---- 安全红线开关：默认一律关闭；单元测试全部通过并经用户二次确认后才可打开 ----
    delete_after_verify: bool = False  # 校验+落库成功后是否删除设备上的单个源文件
    auto_bind_dji: bool = True         # 自动绑定 DJI 系设备（VID/产品串识别）

    # ---- 网络（识曲 / 可选 API 总结走代理；仅保存在本地配置，绝不入库）----
    proxy_url: str = "http://127.0.0.1:7897"

    # ---- 采集 ----
    watched_photo_dirs: list[str] = field(default_factory=list)
    auto_music_recognition: bool = False  # 后台自动识曲，默认关闭

    # ---- 流水线（M2）----
    # 历史归档分界线：早于该日期的录音只归档保留原 WAV，不自动转写（可手动触发）
    auto_transcribe_since: str = "2026-11-22"
    auto_pipeline: bool = True            # 自动补全所有未处理日期（>= 分界线）
    pipeline_provider: str = "auto"       # auto(尝试CUDA，失败回退CPU) / cpu / cuda
    cluster_distance_threshold: float = 0.55  # CAM++ cosine 距离聚类阈值
    cleanup_after_summary: bool = False   # AI 总结归档后清理未星标压缩音频（默认关）

    # ---- UI（M4 使用）----
    theme: str = "dark"  # dark / light

    # ---- 导出（M3）：投喂 Prompt 模板（留空使用内置默认，可自由编辑）----
    export_prompt: str = ""

    # ---- 可选 API 一键总结（M3）：留空 API Key 时完全隐藏/静默 ----
    api_base_url: str = "https://open.bigmodel.cn/api/paas/v4"  # 智谱 GLM（OpenAI 兼容）占位
    api_key: str = ""        # 仅存本地 config（仓库外），绝不入库
    api_model: str = "glm-4.5v"

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        path = path or default_config_path()
        cfg = cls()
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = {}
            valid = {f.name for f in fields(cls)}
            for key, value in raw.items():
                if key in valid and value is not None:
                    setattr(cfg, key, value)
        cfg._path = path  # type: ignore[attr-defined]
        return cfg

    def save(self, path: Path | None = None) -> None:
        path = path or getattr(self, "_path", None) or default_config_path()  # type: ignore[attr-defined]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        self._path = path  # type: ignore[attr-defined]

    # ---- 数据目录助手（全部个人数据收敛于 data_root，被 .gitignore 拦截）----
    def data_root_path(self) -> Path:
        return Path(self.data_root)

    def subdir(self, *parts: str) -> Path:
        p = self.data_root_path().joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p
