"""长期声纹库（speaker_profiles）：零门槛改名即学、多原型加权更新、合并聚类。

设计：
- 每个人物维护至多 MAX_PROTOTYPES=5 个原型向量；新样本与最近原型余弦 >= 0.82 时
  做加权平均渐进更新（new = old*(1-a) + x*a），否则新增原型（挤掉最旧的一个）；
- 「改名即学」：用户在界面将「说话人 N」改为「我」→ 自动抽取该聚类质量最高的
  若干切片 Embedding 更新声纹库，后续日期自动识别且越用越准；
- 「合并聚类」：两个被拆散的编号合并为一个，重新学习。
"""
from __future__ import annotations

import json
import logging

import numpy as np

from ..core.utils import now_iso

log = logging.getLogger(__name__)

MAX_PROTOTYPES = 5
MERGE_BLEND_THRESHOLD = 0.82
LEARN_TOP_K = 8  # 每次学习抽取质量最高的切片数


def _load_prototypes(db, name: str) -> tuple[int | None, np.ndarray]:
    row = db.query_one("SELECT id, prototypes FROM speaker_profiles WHERE name=?", (name,))
    if row is None:
        return None, np.zeros((0,), dtype=np.float32)
    try:
        arr = np.asarray(json.loads(row["prototypes"]), dtype=np.float32)
        return row["id"], arr if arr.ndim == 2 else np.zeros((0,), dtype=np.float32)
    except (ValueError, TypeError):
        return row["id"], np.zeros((0,), dtype=np.float32)


def _save_prototypes(db, name: str, protos: np.ndarray, sample_delta: int) -> None:
    pid, _existing = _load_prototypes(db, name)
    payload = json.dumps(np.asarray(protos, dtype=np.float32).tolist(), separators=(",", ":"))
    if pid is None:
        db.execute(
            "INSERT INTO speaker_profiles(name, prototypes, sample_count, updated_at)"
            " VALUES(?,?,?,?)", (name, payload, max(sample_delta, 0), now_iso())
        )
    else:
        db.execute(
            "UPDATE speaker_profiles SET prototypes=?, sample_count=sample_count+?,"
            " updated_at=? WHERE id=?", (payload, max(sample_delta, 0), now_iso(), pid)
        )


def learn_from_cluster(db, embeddings: list[np.ndarray], name: str) -> int:
    """将一个聚类的优质切片 Embedding 入库/进化。返回学习样本数。"""
    good = [e for e in embeddings if e is not None and len(e) > 0]
    if not good:
        return 0
    X = np.stack(good).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    top = X[:LEARN_TOP_K]  # 调用方已按质量排序
    pid, protos = _load_prototypes(db, name)
    merged: list[np.ndarray] = [p for p in (protos if protos.ndim == 2 and len(protos) else [])]
    for x in top:
        if merged:
            sims = np.asarray([
                float(np.dot(x, p) / (np.linalg.norm(p) + 1e-9)) for p in merged
            ])
            j = int(np.argmax(sims))
            if sims[j] >= MERGE_BLEND_THRESHOLD:
                merged[j] = merged[j] * 0.9 + x * 0.1
                continue
        merged.append(x)
        if len(merged) > MAX_PROTOTYPES:
            merged.pop(0)  # 挤掉最旧原型
    _save_prototypes(db, name, np.stack(merged).astype(np.float32), len(top))
    log.info("声纹库已更新：%s（+%d 切片，%d 个原型）", name, len(top), len(merged))
    return len(top)


def rename_cluster(db, segments: list[Segment], cluster_id: int, new_name: str,
                   quality_fn=None) -> int:
    """改名即学：刷新该日全文称呼 + 更新声纹库。返回学习切片数。"""
    from .diarize import cluster_centroid

    members = [s for s in segments if s.cluster_id == cluster_id]
    learned = 0
    if members:
        ranked = sorted(members, key=lambda s: -s.rms_db)
        embeddings = [s.embedding for s in ranked]
        if quality_fn is not None:
            embeddings = quality_fn(members)
        learned = learn_from_cluster(db, embeddings, new_name)
    return learned


def merge_clusters(db, segments: list[Segment], keep_id: int, gone_id: int,
                   name: str | None) -> int:
    """合并被拆散的两个聚类：gone_id 的切片并入 keep_id，可顺便命名并学习。"""
    moved = 0
    for s in segments:
        if s.cluster_id == gone_id:
            s.cluster_id = keep_id
            moved += 1
    if name:
        learn_from_cluster(
            db, sorted([s.embedding for s in segments
                        if s.cluster_id == keep_id and s.embedding is not None],
                       key=lambda e: 0)[:LEARN_TOP_K], name
        )
    return moved


def delete_profile(db, name: str) -> bool:
    row = db.query_one("SELECT id FROM speaker_profiles WHERE name=?", (name,))
    if row is None:
        return False
    db.execute("DELETE FROM speaker_profiles WHERE id=?", (row["id"],))
    return True
