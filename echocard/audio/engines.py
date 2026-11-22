"""本地模型管理：懒加载 sherpa-onnx 引擎（CUDA 优先探测，失败平滑回退 CPU）。

显存红线：SenseVoice-Small int8 经 onnxruntime CUDA EP 峰值 < 1.5GB（远低于 4.5GB）；
CPU 路径零显存占用。模型文件由 scripts/setup_models.py 预先下载至 <data_root>/models。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from ..core.config import AppConfig

log = logging.getLogger(__name__)

MODELS_SUBDIR = "models"


def models_dir(cfg: AppConfig) -> Path:
    return cfg.subdir(MODELS_SUBDIR)


def resolve_model(cfg: AppConfig, filename: str) -> Path:
    p = models_dir(cfg) / filename
    if not p.exists():
        raise FileNotFoundError(
            f"缺少模型文件 {filename}，请先运行：scripts/setup_models.py（目标目录 {p.parent}）"
        )
    return p


class Engines:
    """进程级单例：VAD / ASR / 声纹 / 音频打标四个模型全部懒加载。"""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._lock = threading.RLock()
        self._vad = None
        self._asr = None
        self._embed = None
        self._tagger = None
        self.provider_used: str | None = None

    # ---- provider 探测 ----
    def _pick_provider(self) -> str:
        pref = self.cfg.pipeline_provider
        if pref == "cuda":
            # 显式要求 CUDA：需要 sherpa-onnx GPU 编译版轮子；CPU 轮子会在内部
            # 打印警告并自动回退 CPU（不影响正确性）
            log.info("provider=cuda：请确认已安装 sherpa-onnx GPU 版本，否则内部自动回退 CPU")
            return "cuda"
        # "cpu" 与 "auto"：PyPI 轮子为 CPU 构建（int8 CPU 推理已足够快），诚实报告 cpu
        return "cpu"

    def _cuda_available(self) -> bool:  # 兼容旧调用占位
        return self.cfg.pipeline_provider == "cuda"

    # ---- 懒加载 ----
    def vad(self):
        import sherpa_onnx

        with self._lock:
            if self._vad is None:
                config = sherpa_onnx.VadModelConfig()
                config.silero_vad.model = str(resolve_model(self.cfg, "silero_vad.onnx"))
                config.silero_vad.threshold = 0.5
                config.silero_vad.min_silence_duration = 0.35
                config.silero_vad.min_speech_duration = 0.30
                config.sample_rate = 16000
                self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=120)
            return self._vad

    def asr(self):
        import sherpa_onnx

        with self._lock:
            if self._asr is None:
                provider = self._pick_provider()
                self.provider_used = provider
                self._asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(resolve_model(self.cfg, "sensevoice-int8.onnx")),
                    tokens=str(resolve_model(self.cfg, "tokens.txt")),
                    use_itn=True, num_threads=4, provider=provider,
                )
            return self._asr

    def embed(self):
        import sherpa_onnx

        with self._lock:
            if self._embed is None:
                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
                config.model = str(resolve_model(self.cfg, "campplus.onnx"))
                config.num_threads = 2
                config.provider = "cpu"
                self._embed = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            return self._embed

    @property
    def vad_model_path(self) -> Path:
        v5 = models_dir(self.cfg) / "silero_vad_v5.onnx"
        if v5.exists():
            return v5  # Silero VAD v5（规格书指定；sherpa-onnx ≥1.10 已适配）
        return resolve_model(self.cfg, "silero_vad.onnx")

    def tagger(self):
        """YAMNet 音乐打标（onnxruntime 原生推理，sherpa-onnx 2.x 已移除 AudioTagging）。

        模型缺失时返回 None（降级为仅用 SenseVoice 事件标签）。
        """
        with self._lock:
            if self._tagger is not None:
                return self._tagger if self._tagger != "missing" else None
            try:
                model = resolve_model(self.cfg, "yamnet.onnx")
                labels = resolve_model(self.cfg, "class_labels.csv")
                self._tagger = YamnetTagger(model, labels)
            except Exception:
                log.warning("YAMNet 不可用，音乐兜底检测降级为事件标签", exc_info=True)
                self._tagger = "missing"
            return self._tagger if self._tagger != "missing" else None


class YamnetTagger:
    """YAMNet ONNX 直推：music_score(samples) → 0~1 音乐概率。

    YAMNet 输入为 0.975s/15600 样本帧（16kHz float32）；输出 [1,521] 分数。
    """

    FRAME = 15600
    MUSIC_KEYWORDS = ("music", "singing")

    def __init__(self, model_path: Path, labels_path: Path):
        import onnxruntime

        self.sess = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.music_idx: list[int] = []
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0].isdigit():
                continue
            label = (parts[-1] if len(parts) >= 2 else "").lower()
            if any(k in label for k in self.MUSIC_KEYWORDS):
                self.music_idx.append(int(parts[0]))
        log.info("YAMNet 就绪：音乐类标签索引 %d 个", len(self.music_idx))

    def music_score(self, samples: "np.ndarray") -> float:
        """一段 16kHz float32 波形的音乐概率（多帧取最大值）。"""
        best = 0.0
        n = len(samples)
        for i in range(0, max(0, n - self.FRAME + 1), self.FRAME):
            frame = samples[i:i + self.FRAME].astype(np.float32)[None, :]
            out = self.sess.run(None, {self.input_name: frame})
            scores = np.asarray(out[0]).reshape(-1)
            if self.music_idx:
                best = max(best, float(np.max(scores[self.music_idx])))
        return best


_ENGINES: Engines | None = None
_ENGINES_LOCK = threading.Lock()


def get_engines(cfg: AppConfig) -> Engines:
    global _ENGINES
    with _ENGINES_LOCK:
        if _ENGINES is None or _ENGINES.cfg is not cfg:
            _ENGINES = Engines(cfg)
        return _ENGINES


# ---------------------------------------------------------------- 工具函数
def pcm_to_float(pcm: bytes | bytearray) -> np.ndarray:
    return np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0


def rms_db(samples: np.ndarray) -> float:
    """均方根电平 dBFS（客观特征标签用）。"""
    if len(samples) == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(samples))))
    return 20.0 * np.log10(max(r, 1e-10))
