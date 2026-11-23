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


def cluster_segments(segments: list[Segment], distance_threshold: float = 0.55,
                     start_id: int = 1) -> dict[int, list[Segment]]:
    """就地聚类：为带 embedding 的语音片段分配 cluster_id。

    **必须对整个「日」的所有片段统一调用**（此前按文件分别调用，导致同一天
    每个文件都从 1 开始编号，「说话人 1」实际是几十个不同的人）。

    - start_id：新编号起始值（重处理时顺延既有编号，避免与保留分段冲突）
    - 返回 {cluster_id: [segments...]}
    """
    voiced = [s for s in segments if s.embedding is not None and s.duration >= 1.0]
    if not voiced:
        return {}
    if len(voiced) == 1:
        voiced[0].cluster_id = start_id
        return {start_id: [voiced[0]]}
    X = np.stack([s.embedding for s in voiced]).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    D = pdist(X, metric="cosine")
    Z = linkage(D, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # 按总时长降序重编号（起始编号可由调用方指定）：编号 1 对应说话最多的人
    totals: dict[int, float] = {}
    for s, lb in zip(voiced, labels):
        totals[lb] = totals.get(lb, 0.0) + s.duration
    order = sorted(totals, key=lambda k: -totals[k])
    remap = {old: start_id + i for i, old in enumerate(order)}
    groups: dict[int, list[Segment]] = {}
    for s, lb in zip(voiced, labels):
        s.cluster_id = remap[lb]
        groups.setdefault(s.cluster_id, []).append(s)
    log.info("聚类完成：%d 个切片 → %d 个说话人（距离阈值 %.2f，起始编号 %d）",
             len(voiced), len(totals), distance_threshold, start_id)
    return groups


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
    """挑选 2~3 段信噪比最高（RMS 最大）、时长 3~8 秒的代表性原声。

    同一文件内优先；若该聚类只存在于某些文件，则按片段归属自然分组。
    """
    pool = [s for s in segments if s.cluster_id == cluster_id and s.embedding is not None]
    if not pool:
        return []
    scored = sorted(pool, key=lambda s: -s.rms_db)
    chosen: list[Segment] = []
    seen_files: set = set()
    # 先按「不同文件各取一段」保证覆盖，再补足
    for s in scored:
        if len(chosen) >= count:
            break
        if min_s <= s.duration <= max_s and s.file_id not in seen_files:
            chosen.append(s)
            seen_files.add(s.file_id)
    for s in scored:
        if len(chosen) >= count:
            break
        if s not in chosen and min_s <= s.duration <= max_s:
            chosen.append(s)
    if not chosen:  # 没有理想时长时退回 RMS 最高的短切片
        chosen = scored[:count]
    return chosen


def soft_match(db, centroid: np.ndarray, high: float = 0.75,
               mid: float = 0.55) -> tuple[str | None, float, str]:
    """聚类中心 vs 声纹库软匹配 → (显示名, 最高相似度, 匹配状态)。

    匹配策略：
    - 每个档案取「多原型中最相似的一个」再横向比较（原型会随佩戴/环境影响漂移，
      单原型均值化会互相抵消，因此用 max 而非 mean）；
    - 相似度做温和校准：CAM++ 跨天跨佩戴场景同人常见 0.45~0.60，
      故 mid 阈值默认 0.42、high 默认 0.62（可在配置里覆盖）。
    """
    import json

    best_name, best_score = None, 0.0
    for row in db.query("SELECT name, prototypes FROM speaker_profiles"):
        try:
            protos = np.asarray(json.loads(row["prototypes"]), dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if protos.ndim != 2 or protos.size == 0:
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
