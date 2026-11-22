"""EchoCard 本地模型一键下载（ModelScope 优先 → hf-mirror → GitHub 兜底）。

下载目标（<data_root>/models/，默认 D:\\EchoCardData\\models）：
  silero_vad.onnx          ~0.6MB  Silero VAD（sherpa-onnx 兼容版）
  silero_vad_v5.onnx       ~2.3MB  Silero VAD v5（存在时优先使用）
  sensevoice-int8.onnx     ~160MB  SenseVoice-Small int8（zh/en/ja/ko/yue + 事件标签）
  tokens.txt               ~1.3MB  SenseVoice 词表
  campplus.onnx            ~28MB   3D-Speaker CAM++ 声纹 Embedding
  yamnet.onnx              ~16MB   YAMNet 音乐/事件打标兜底
  class_labels.csv         ~14KB   YAMNet 521 类标签

用法：
  .venv\\Scripts\\python scripts\\setup_models.py           # 全量
  .venv\\Scripts\\python scripts\\setup_models.py --only sensevoice
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echocard.core.config import AppConfig  # noqa: E402

PROXY = None  # 运行时从 config 读取

GITHUB = "https://github.com"
GHPROXY = "https://gh-proxy.com"  # GitHub 加速镜像（实测可用）
HFM = "https://hf-mirror.com"
HF = "https://huggingface.co"
MS = "https://modelscope.cn"

# ModelScope 快速镜像（实测满速；官方 HF 仓 csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17）
MS_SV = "mariolux/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
MS_VAD = "xiaowangge/sherpa-onnx-sense-voice-small"

def _ms(repo: str, path: str) -> str:
    return f"{MS}/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"

# 目标文件 -> (候选 URL 列表, 归档内提取名)
MODELS: dict[str, tuple[list[str], str | None]] = {
    "silero_vad_v5.onnx": ([
        _ms(MS_VAD, "silero-vad/model.onnx"),                       # 实测：满速
        f"{GITHUB}/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    ], None),
    "sensevoice-int8.onnx": ([
        _ms(MS_SV, "model.int8.onnx"),                              # 实测：满速
        f"{GHPROXY}/{GITHUB}/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
        f"{GITHUB}/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
    ], "model.int8.onnx"),
    "tokens.txt": ([
        _ms(MS_SV, "tokens.txt"),
        f"{HFM}/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt",
    ], None),
    "campplus.onnx": ([
        f"{GHPROXY}/{GITHUB}/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        f"{GITHUB}/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
        "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
    ], None),
    "yamnet.onnx": ([
        f"{HFM}/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx",
        f"{HF}/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx",
    ], None),
    "class_labels.csv": ([
        f"{HFM}/zeropointnine/yamnet-onnx/resolve/main/yamnet_class_map.csv",
        f"{HF}/zeropointnine/yamnet-onnx/resolve/main/yamnet_class_map.csv",
    ], None),
}

MIN_SIZE = {"sensevoice-int8.onnx": 100_000_000, "campplus.onnx": 10_000_000,
            "yamnet.onnx": 5_000_000, "silero_vad_v5.onnx": 1_000_000,
            "tokens.txt": 500_000, "class_labels.csv": 5_000,
            "silero_vad.onnx": 100_000}


def _download(url: str, dst: Path) -> None:
    import urllib.request

    handlers = []
    if PROXY:
        handlers.append(urllib.request.ProxyHandler(
            {"http": PROXY, "https": PROXY}))
    opener = urllib.request.build_opener(*handlers)
    opener.addheaders = [("User-Agent", "EchoCard/0.1")]
    req = opener.open(url, timeout=90)
    total = int(req.headers.get("Content-Length") or 0)
    done = 0
    with open(dst, "wb") as f:
        while True:
            chunk = req.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r    {done/1e6:8.1f}/{total/1e6:.1f}MB ({pct:3d}%)", end="", flush=True)
    print()


def fetch(name: str, out_dir: Path, force: bool = False) -> bool:
    urls, member = MODELS[name]
    dst = out_dir / name
    if dst.exists() and not force and dst.stat().st_size >= MIN_SIZE.get(name, 10_000):
        print(f"[skip] {name} 已存在")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="echocard-models-") as tmp:
        for url in urls:
            src = Path(tmp) / (url.rsplit("/", 1)[-1].split("?")[0] or "dl")
            print(f"[down] {name} ← {url.split('/')[2]}")
            try:
                _download(url, src)
            except Exception as e:
                print(f"    失败：{str(e)[:160]}，尝试下一候选…")
                continue
            try:
                if member:  # 归档内提取指定成员
                    opener = tarfile if src.suffix in (".bz2", ".bz") or src.name.endswith(".tar.bz2") else zipfile
                    if src.name.endswith(".tar.bz2"):
                        with tarfile.open(src, "r:bz2") as tf:
                            hit = None
                            for m in tf.getmembers():
                                if m.name.endswith(member):
                                    hit = m
                                    break
                            if hit is None:
                                raise FileNotFoundError(f"归档内未找到 {member}")
                            with tf.extractfile(hit) as fsrc, open(dst, "wb") as fdst:
                                fdst.write(fsrc.read())
                            # tar 包内附带 tokens.txt 时顺手保存
                            if name == "sensevoice-int8.onnx":
                                tok = dst.parent / "tokens.txt"
                                if not tok.exists() or tok.stat().st_size < MIN_SIZE["tokens.txt"]:
                                    for m in tf.getmembers():
                                        if m.name.endswith("tokens.txt"):
                                            with tf.extractfile(m) as fsrc2, open(tok, "wb") as fdst2:
                                                fdst2.write(fsrc2.read())
                                            break
                    else:
                        with zipfile.ZipFile(src) as zf:
                            hit = next(n for n in zf.namelist() if n.endswith(member))
                            zf.extract(hit, tmp)
                            (Path(tmp) / hit).replace(dst)
                else:
                    src.replace(dst)
            except Exception as e:
                print(f"    解包失败：{str(e)[:160]}，尝试下一候选…")
                continue
            if dst.exists() and dst.stat().st_size >= MIN_SIZE.get(name, 10_000):
                print(f"[ok]   {name}  {dst.stat().st_size/1e6:.1f}MB")
                return True
            dst.unlink(missing_ok=True)
    print(f"[FAIL] {name} 全部候选源失败")
    return False


def main() -> None:
    global PROXY
    parser = argparse.ArgumentParser(description="下载 EchoCard 本地模型")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--only", help="只下载指定模型（如 sensevoice）")
    args = parser.parse_args()

    cfg = AppConfig.load(Path(args.config) if args.config else None)
    PROXY = cfg.proxy_url
    out = Path(cfg.data_root) / "models"
    print(f"模型目录：{out}（代理：{PROXY or '无'}）")

    targets = ([f"{args.only}"] if args.only in MODELS
               else [k for k in MODELS if args.only and args.only.split(".")[0] in k]
               if args.only else list(MODELS))
    ok = all(fetch(t, out, args.force) for t in targets)
    print("\n全部就绪 ✅" if ok else "\n存在失败项 ❌（见上方日志）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
