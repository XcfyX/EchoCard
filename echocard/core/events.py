"""线程安全事件总线 + 流水线环形日志。

供 UI 轮询（/api/v1/ingest/log）与后续流水线触发（pipeline.day_ready）使用；
订阅者异常绝不影响发布者。
"""
from __future__ import annotations

import threading
from collections import deque

from .utils import now_iso


class EventBus:
    def __init__(self, log_capacity: int = 500):
        self._subs: dict[str, list] = {}
        self._lock = threading.RLock()
        self._log: deque[dict] = deque(maxlen=log_capacity)

    def subscribe(self, topic: str, fn) -> None:
        with self._lock:
            self._subs.setdefault(topic, []).append(fn)

    def publish(self, topic: str, data: dict | None = None) -> None:
        entry = {"ts": now_iso(), "topic": topic, "data": data or {}}
        with self._lock:
            self._log.append(entry)
            handlers = list(self._subs.get(topic, []))
        for fn in handlers:
            try:
                fn(entry["data"])
            except Exception:  # 订阅者故障不阻断流水线
                import logging

                logging.getLogger(__name__).exception("事件订阅者异常 topic=%s", topic)

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            items = list(self._log)
        return items[-limit:]
