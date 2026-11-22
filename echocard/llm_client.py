"""轻量 OpenAI 兼容多模态 HTTP 客户端（约 80 行，绝不引入 Agent 框架）。

- 留空 API Key 时功能静默隐藏（服务端判定 available=False）；
- 支持多模态（文本 + 压缩照片 base64）；
- 默认占位指向智谱 GLM（OpenAI 兼容格式），可改任意兼容服务。
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import httpx

from .core.config import AppConfig

log = logging.getLogger(__name__)

DEFAULT_BASE = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4.5v"


def api_ready(cfg: AppConfig) -> bool:
    return bool(cfg.api_key.strip()) if cfg.api_key else False


def chat(cfg: AppConfig, prompt: str, images: list[Path] | None = None,
         system: str | None = None, timeout: float = 300.0,
         max_images: int = 12) -> str:
    """发送一次 chat/completions；返回助手回复文本。"""
    if not api_ready(cfg):
        raise RuntimeError("API Key 未配置（设置页填写后可用）")
    base = (cfg.api_base_url or DEFAULT_BASE).rstrip("/")
    model = cfg.api_model or DEFAULT_MODEL
    proxy = cfg.proxy_url or None

    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in (images or [])[:max_images]:
        b64 = base64.b64encode(Path(img).read_bytes()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    body = {"model": model, "messages": messages, "temperature": 0.7}
    headers = {"Authorization": f"Bearer {cfg.api_key}",
               "Content-Type": "application/json"}

    with httpx.Client(timeout=timeout, proxy=proxy, trust_env=True) as client:
        resp = client.post(f"{base}/chat/completions", json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"API 响应格式异常: {json.dumps(data)[:300]}") from e
