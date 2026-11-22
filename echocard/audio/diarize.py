"""无监督说话人聚类 + 与长期声纹库软匹配。

流程（规格书 B2）：
  1. 当日全部人声切片提取 CAM++ Embedding → 余弦距离层次聚类为中性编号「说话人 N」
  2. 聚类中心与 speaker_profiles 声纹库软匹配：
     >= 0.75 高置信度 → 显示人名；0.55~0.75 → 「疑似-XXX (说话人 N)」；< 0.55 → 保持中性
  3. 改名/合并后由 speakers.learn_from_cluster 被动更新声纹库，越用越准
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .vad_asr import Segment

log = logging.getLogger(__name__)


def cluster_segments(segments: list[Segment], distance_threshold: float = 0.55) -> None:
    """就地聚类：为带 embedding 的语音片段分配 cluster_id（从 1 开始，按时长降序编号）。"""
    voiced = [s for s in segments if s.embedding is not None and s.duration >= 1.0]
    if not voiced:
        return
    if len(voiced) == 1:
        voiced[0].cluster_id = 1
        return
    X = np.stack([s.embedding for s in voiced]).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    D = pdist(X, metric="cosine")
    Z = linkage(D, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # 按总时长降序重编号：说话人 1 = 说话最多的人
    totals: dict[int, float] = {}
    for s, lb in zip(voiced, labels):
        totals[lb] = totals.get(lb, 0.0) + s.duration
    order = sorted(totals, key=lambda k: -totals[k])
    remap = {old: i + 1 for i, old in enumerate(order)}
    for s, lb in zip(voiced, labels):
        s.cluster_id = remap[lb]
    log.info("聚类完成：%d 个切片 → %d 个说话人（距离阈值 %.2f）",
             len(voiced), len(totals), distance_threshold)


def cluster_centroid(segments: list[Segment], cluster_id: int) -> np.ndarray | None:
    embs = [s.embedding for s in segments
            if s.cluster_id == cluster_id and s.embedding is not None]
    if not embs:
        return None
    c = np.mean(np.stack(embs).astype(np.float32), axis=0)
    n = np.linalg.norm(c)
    return c / (n + 1e-9)


def pick_representative(segments: list[Segment], cluster_id: int, count: int = 3,
                        min_s: float = 3.0, max_s: float = 8.0) -> list[Segment]:
    """挑选 2~3 段信噪比最高（RMS 最大）、时长 3~8 秒的代表性原声。"""
    pool = [s for s in segments if s.cluster_id == cluster_id and s.embedding is not None]
    if not pool:
        return []
    scored = sorted(pool, key=lambda s: -s.rms_db)
    chosen: list[Segment] = []
    for s in scored:
        if len(chosen) >= count:
            break
        if s.duration >= min_s and s.duration <= max_s:
            chosen.append(s)
    if not chosen:  # 没有理想时长时退回 RMS 最高的短切片
        chosen = scored[:count]
    return chosen


def soft_match(db, centroid: np.ndarray, high: float = 0.75,
               mid: float = 0.55) -> tuple[str | None, float, str]:
    """聚类中心 vs 声纹库软匹配 → (显示名, 最高相似度, 匹配状态)。

    高置信度 → 人名；中置信度 → 「疑似-XXX (说话人 N)」（由调用方拼装）；
    低置信度 → None（保持中性编号）。
    """
    import json

    best_name, best_score = None, 0.0
    for row in db.query("SELECT name, prototypes FROM speaker_profiles"):
        try:
            protos = np.asarray(json.loads(row["prototypes"]), dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if protos.size == 0:
            continue
        protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9)
        score = float(np.max(protos @ centroid))
        if score > best_score:
            best_name, best_score = row["name"], score
    if best_score >= high:
        return best_name, best_score, "matched"
    if best_score >= mid:
        return best_name, best_score, "suspected"
    return None, best_score, "unmatched"
