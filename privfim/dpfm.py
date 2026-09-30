from __future__ import annotations

import hashlib
import itertools
import math
import os
from dataclasses import dataclass

import numpy as np

from .types import AlphaReport


UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """可复现的 64 位混合函数，用于为对齐用户 ID 生成公共哈希秩。"""
    with np.errstate(over="ignore"):
        values = (values + np.uint64(0x9E3779B97F4A7C15)) & UINT64_MASK
        values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & UINT64_MASK
        values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & UINT64_MASK
        return values ^ (values >> np.uint64(31))


def _build_cuda_ranks(
    n_rows: int,
    m: int,
    gamma: float,
    seed: int,
    block_size: int,
) -> np.ndarray:
    """Generate the same deterministic ranks on CUDA, returning host storage."""
    import cupy as cp

    ranks = np.empty((n_rows, m), dtype=np.uint16)
    row_ids = cp.arange(n_rows, dtype=cp.uint64)[:, None]
    denominator = math.log1p(gamma)
    cuda_block_size = max(block_size, 256)
    seed_value = cp.uint64(seed & ((1 << 64) - 1))

    def splitmix64(values):
        values = values + cp.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> cp.uint64(30))) * cp.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> cp.uint64(27))) * cp.uint64(
            0x94D049BB133111EB
        )
        return values ^ (values >> cp.uint64(31))

    for start in range(0, m, cuda_block_size):
        stop = min(m, start + cuda_block_size)
        sketch_ids = cp.arange(start, stop, dtype=cp.uint64)[None, :]
        mixed_input = row_ids ^ splitmix64(sketch_ids + seed_value)
        hashed = splitmix64(mixed_input)
        uniform = ((hashed >> cp.uint64(11)).astype(cp.float64) + 0.5) / float(
            1 << 53
        )
        geometric = cp.ceil(-cp.log1p(-uniform) / denominator)
        ranks[:, start:stop] = cp.asnumpy(
            cp.clip(geometric, 1, np.iinfo(np.uint16).max).astype(cp.uint16)
        )
    return ranks


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class RankOracle:
    """所有客户端共享的用户 ID 几何哈希秩，保证跨方 α 可以组合。

    秩矩阵按 ``(user, coordinate)`` 存储，使集合成员的整行访问保持连续。
    """

    ranks: np.ndarray
    gamma: float

    @classmethod
    def build(
        cls,
        n_rows: int,
        m: int,
        gamma: float,
        seed: int,
        block_size: int = 64,
    ) -> "RankOracle":
        if os.environ.get("PRIVFIM_CUDA_RANKS") == "1":
            return cls(
                ranks=_build_cuda_ranks(
                    n_rows=n_rows,
                    m=m,
                    gamma=gamma,
                    seed=seed,
                    block_size=block_size,
                ),
                gamma=gamma,
            )
        ranks = np.empty((n_rows, m), dtype=np.uint16)
        row_ids = np.arange(n_rows, dtype=np.uint64)[:, None]
        denominator = math.log1p(gamma)

        for start in range(0, m, block_size):
            stop = min(m, start + block_size)
            sketch_ids = np.arange(start, stop, dtype=np.uint64)[None, :]
            seed_value = np.uint64(seed & ((1 << 64) - 1))
            mixed_input = row_ids ^ _splitmix64(sketch_ids + seed_value)
            hashed = _splitmix64(mixed_input)
            uniform = ((hashed >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)
            geometric = np.ceil(-np.log1p(-uniform) / denominator)
            ranks[:, start:stop] = np.clip(
                geometric, 1, np.iinfo(np.uint16).max
            ).astype(np.uint16)

        return cls(ranks=ranks, gamma=gamma)

    @property
    def m(self) -> int:
        return int(self.ranks.shape[1])

    def private_alpha(
        self,
        membership: np.ndarray,
        epsilon: float,
        delta: float,
        random_seed: int,
    ) -> tuple[np.ndarray, float]:
        row_indexes = np.flatnonzero(membership)
        if len(row_indexes) == 0:
            clean_alpha = np.zeros(self.m, dtype=np.float64)
        else:
            # NumPy advanced indexing materializes the selected m-by-|D| matrix.
            # Bound that temporary so large sketches fit without changing ranks.
            target_bytes = 256 * 2**20
            rows_per_chunk = max(
                1, target_bytes // (self.m * self.ranks.dtype.itemsize)
            )
            clean_max = np.zeros(self.m, dtype=self.ranks.dtype)
            for start in range(0, len(row_indexes), rows_per_chunk):
                selected = row_indexes[start : start + rows_per_chunk]
                chunk_max = self.ranks[selected].max(axis=0)
                np.maximum(clean_max, chunk_max, out=clean_max)
            clean_alpha = clean_max.astype(np.float64)

        phantom_count, alpha_min = dpfm_parameters(
            epsilon=epsilon,
            delta=delta,
            m=self.m,
            gamma=self.gamma,
        )
        rng = np.random.default_rng(random_seed)
        phantom_alpha = sample_geometric_max(
            rng=rng,
            sample_count=self.m,
            set_size=phantom_count,
            gamma=self.gamma,
        )
        private = np.maximum.reduce(
            [clean_alpha, phantom_alpha, np.full(self.m, alpha_min)]
        )
        return private.astype(np.float64), phantom_count


def dpfm_parameters(
    epsilon: float,
    delta: float,
    m: int,
    gamma: float,
) -> tuple[float, float]:
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("DPFM 的 epsilon/delta 不合法")
    eps1 = epsilon / (4.0 * math.sqrt(m * math.log(1.0 / delta)))
    phantom_count = math.ceil(1.0 / math.expm1(eps1))
    alpha_min = math.ceil(
        -math.log(-math.expm1(-eps1)) / math.log1p(gamma)
    )
    return float(phantom_count), float(alpha_min)


def sample_geometric_max(
    rng: np.random.Generator,
    sample_count: int,
    set_size: float,
    gamma: float,
) -> np.ndarray:
    """直接采样 K 个几何变量的最大值，避免真的构造 K 个幻影元素。"""
    uniform = np.clip(rng.random(sample_count), np.finfo(float).tiny, 1.0)
    one_minus_root = -np.expm1(np.log(uniform) / set_size)
    return np.ceil(-np.log(one_minus_root) / math.log1p(gamma))


def fm_cardinality(alpha: np.ndarray, gamma: float, phantom_count: float) -> float:
    m = len(alpha)
    debias = 0.7213 / (1.0 + 1.079 / m)
    denominator = np.sum(np.power(1.0 + gamma, -alpha))
    if denominator <= 0:
        return float("inf")
    return float(m / denominator * debias - phantom_count)


def _candidate_grid(upper: int, map_step: int, max_map_points: int) -> np.ndarray:
    """构造包含 0 和上界的整数搜索网格。"""
    if upper <= 0:
        return np.asarray([0.0])
    step = max(map_step, math.ceil(upper / max_map_points))
    values = np.arange(0, upper + 1, step, dtype=np.float64)
    if values[-1] != upper:
        values = np.append(values, float(upper))
    return values


def _bounded_candidate_grid(
    lower: int,
    upper: int,
    map_step: int,
    max_map_points: int,
) -> np.ndarray:
    """在闭区间内构造包含两个端点的整数 MAP 搜索网格。"""
    if lower < 0 or upper < lower:
        raise ValueError(f"MAP 搜索区间不合法: [{lower}, {upper}]")
    if lower == upper:
        return np.asarray([float(lower)])
    step = max(map_step, math.ceil((upper - lower) / max_map_points))
    values = np.arange(lower, upper + 1, step, dtype=np.float64)
    if values[-1] != upper:
        values = np.append(values, float(upper))
    return values


def _geometric_log_cdf(value: float, gamma: float) -> float:
    """单个几何秩在 value 处的 log-CDF。"""
    if value <= 0:
        return float("-inf")
    cdf = -math.expm1(-value * math.log1p(gamma))
    return math.log(cdf)


def _geometric_log_cdf_array(values: np.ndarray, gamma: float) -> np.ndarray:
    """向量化几何 CDF 的对数；输入值必须为正。"""
    return np.log(-np.expm1(-values * math.log1p(gamma)))


def _alpha_min_for_report(report: AlphaReport, gamma: float) -> float:
    _, alpha_min = dpfm_parameters(
        epsilon=report.epsilon,
        delta=report.delta,
        m=len(report.alpha),
        gamma=gamma,
    )
    return alpha_min


def private_cardinality_map_estimate(
    report: AlphaReport,
    n_rows: int,
    gamma: float,
    map_step: int,
    max_map_points: int,
) -> float:
    """使用完整私有 Alpha 向量估计单个集合大小。"""
    candidates = _candidate_grid(n_rows, map_step, max_map_points)
    total_sizes = candidates + report.phantom_count
    alpha_min = _alpha_min_for_report(report, gamma)
    observations, occurrences = np.unique(report.alpha, return_counts=True)
    log_likelihood = np.zeros(len(candidates), dtype=np.float64)

    for observed, occurrence in zip(observations, occurrences, strict=True):
        log_high = total_sizes * _geometric_log_cdf(observed, gamma)
        if observed <= alpha_min:
            log_probability = log_high
        else:
            log_low = total_sizes * _geometric_log_cdf(observed - 1.0, gamma)
            ratio = np.exp(np.minimum(0.0, log_low - log_high))
            log_probability = log_high + np.log1p(-np.minimum(ratio, 1.0 - 1e-15))
        log_likelihood += float(occurrence) * log_probability

    return float(candidates[int(np.argmax(log_likelihood))])


def intersection_bounds(
    block_counts: np.ndarray,
    n_rows: int,
) -> tuple[int, int]:
    """由边缘集合大小给出交集的 Frechet 集合论上下界。"""
    counts = np.clip(np.asarray(block_counts, dtype=np.float64), 0.0, n_rows)
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("交集边界至少需要一个边缘集合大小")
    lower = math.ceil(max(0.0, float(np.sum(counts)) - (len(counts) - 1) * n_rows))
    upper = math.floor(float(np.min(counts)))
    return min(lower, upper), upper


def _private_cardinality_map_values(
    alpha: np.ndarray,
    phantom_count: float,
    alpha_min: float,
    n_rows: int,
    gamma: float,
    map_step: int,
    max_map_points: int,
) -> float:
    """已知幻影数量和截断下限时，对私有 Alpha 执行联合 MAP。"""
    candidates = _candidate_grid(n_rows, map_step, max_map_points)
    total_sizes = candidates + phantom_count
    observations, occurrences = np.unique(alpha, return_counts=True)
    log_likelihood = np.zeros(len(candidates), dtype=np.float64)
    for observed, occurrence in zip(observations, occurrences, strict=True):
        log_high = total_sizes * _geometric_log_cdf(observed, gamma)
        if observed <= alpha_min:
            log_probability = log_high
        else:
            log_low = total_sizes * _geometric_log_cdf(observed - 1.0, gamma)
            ratio = np.exp(np.minimum(0.0, log_low - log_high))
            log_probability = log_high + np.log1p(
                -np.minimum(ratio, 1.0 - 1e-15)
            )
        log_likelihood += float(occurrence) * log_probability
    return float(candidates[int(np.argmax(log_likelihood))])


def union_complement_estimate(
    reports: list[AlphaReport],
    n_rows: int,
    gamma: float,
    map_step: int,
    max_map_points: int,
) -> float:
    """通过补集取并估计交集频数；该分布对任意块数均精确。"""
    if not reports:
        return 0.0
    if not all(report.is_complement for report in reports):
        raise ValueError("并补估计要求客户端上传补集 Alpha")
    alpha_lengths = {len(report.alpha) for report in reports}
    if len(alpha_lengths) != 1:
        raise ValueError("参与并补的 Alpha 长度必须一致")

    combined_alpha = np.max(np.stack([report.alpha for report in reports]), axis=0)
    phantom_count = sum(report.phantom_count for report in reports)
    alpha_min = max(_alpha_min_for_report(report, gamma) for report in reports)
    union_count = _private_cardinality_map_values(
        alpha=combined_alpha,
        phantom_count=phantom_count,
        alpha_min=alpha_min,
        n_rows=n_rows,
        gamma=gamma,
        map_step=map_step,
        max_map_points=max_map_points,
    )
    return float(np.clip(n_rows - union_count, 0, n_rows))


def union_complement_fm_estimate(
    reports: list[AlphaReport],
    n_rows: int,
    gamma: float,
) -> float:
    """传统 FM 反演消融：合并补集 Alpha 后直接做偏差校正反演。"""
    if not reports:
        return 0.0
    if not all(report.is_complement for report in reports):
        raise ValueError("并补估计要求客户端上传补集 Alpha")
    alpha_lengths = {len(report.alpha) for report in reports}
    if len(alpha_lengths) != 1:
        raise ValueError("参与并补的 Alpha 长度必须一致")

    combined_alpha = np.max(np.stack([report.alpha for report in reports]), axis=0)
    union_count = fm_cardinality(
        alpha=combined_alpha,
        gamma=gamma,
        phantom_count=sum(report.phantom_count for report in reports),
    )
    return float(np.clip(n_rows - union_count, 0, n_rows))


def fm_category_intersection_estimate(
    reports: list[AlphaReport],
    n_rows: int,
    gamma: float,
) -> float:
    """用正向类别桶的 FM Alpha 取并，得到目标交集的补集大小。"""
    if not reports:
        return float(n_rows)
    if any(report.is_complement for report in reports):
        raise ValueError("类别 FM 需要正向取值桶 Alpha")
    alpha_lengths = {len(report.alpha) for report in reports}
    if len(alpha_lengths) != 1:
        raise ValueError("参与类别 FM 取并的 Alpha 长度必须一致")

    combined_alpha = np.max(np.stack([report.alpha for report in reports]), axis=0)
    union_count = fm_cardinality(
        alpha=combined_alpha,
        gamma=gamma,
        phantom_count=sum(report.phantom_count for report in reports),
    )
    return float(np.clip(n_rows - union_count, 0, n_rows))


def _joint_log_cdf(
    limits: np.ndarray,
    candidates: np.ndarray,
    block_counts: np.ndarray,
    phantom_counts: np.ndarray,
    alpha_mins: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """私有 Alpha 联合 CDF；两块时精确，多块时采用公共交集核模型。"""
    if np.any(limits < alpha_mins):
        return np.full(len(candidates), float("-inf"), dtype=np.float64)

    log_cdfs = np.asarray(
        [_geometric_log_cdf(float(value), gamma) for value in limits],
        dtype=np.float64,
    )
    log_common = _geometric_log_cdf(float(np.min(limits)), gamma)
    base = np.sum((block_counts + phantom_counts) * log_cdfs)
    coefficient = log_common - np.sum(log_cdfs)
    return base + candidates * coefficient


def _joint_log_probability(
    observation: np.ndarray,
    candidates: np.ndarray,
    block_counts: np.ndarray,
    phantom_counts: np.ndarray,
    alpha_mins: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """由联合 CDF 的有限差分计算一个 Alpha 向量的 log-PMF。"""
    return _joint_log_probability_batch(
        observations=np.asarray(observation, dtype=np.float64)[None, :],
        candidates=candidates,
        block_counts=block_counts,
        phantom_counts=phantom_counts,
        alpha_mins=alpha_mins,
        gamma=gamma,
    )[0]


def _joint_log_probability_batch(
    observations: np.ndarray,
    candidates: np.ndarray,
    block_counts: np.ndarray,
    phantom_counts: np.ndarray,
    alpha_mins: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """批量计算联合 PMF，避免对每个 Alpha 坐标重复 Python 循环。"""
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError("联合 Alpha 观测必须是二维数组")
    log_terms = []
    signs = []
    for predecessors in itertools.product((0.0, 1.0), repeat=observations.shape[1]):
        predecessor_array = np.asarray(predecessors, dtype=np.float64)
        limits = observations - predecessor_array
        valid = np.all(limits >= alpha_mins, axis=1)
        safe_limits = np.maximum(limits, 1.0)
        log_cdfs = _geometric_log_cdf_array(safe_limits, gamma)
        log_common = _geometric_log_cdf_array(
            np.min(safe_limits, axis=1), gamma
        )
        base = np.sum(
            log_cdfs * (block_counts + phantom_counts)[None, :], axis=1
        )
        coefficient = log_common - np.sum(log_cdfs, axis=1)
        log_term = base[:, None] + coefficient[:, None] * candidates[None, :]
        log_terms.append(np.where(valid[:, None], log_term, -np.inf))
        signs.append(-1.0 if int(predecessor_array.sum()) % 2 else 1.0)

    stacked = np.stack(log_terms, axis=0)
    maximum = np.max(stacked, axis=0)
    scaled = np.sum(
        np.asarray(signs, dtype=np.float64)[:, None, None]
        * np.exp(stacked - maximum[None, :, :]),
        axis=0,
    )
    return maximum + np.log(np.clip(scaled, np.finfo(float).tiny, None))


def _joint_log_likelihood(
    observations: np.ndarray,
    occurrences: np.ndarray,
    candidates: np.ndarray,
    block_counts: np.ndarray,
    phantom_counts: np.ndarray,
    alpha_mins: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """计算一组候选交集基数的联合对数似然。"""
    log_likelihood = np.zeros(len(candidates), dtype=np.float64)
    chunk_size = 128
    for start in range(0, len(observations), chunk_size):
        stop = min(start + chunk_size, len(observations))
        probabilities = _joint_log_probability_batch(
            observations=observations[start:stop],
            candidates=candidates,
            block_counts=block_counts,
            phantom_counts=phantom_counts,
            alpha_mins=alpha_mins,
            gamma=gamma,
        )
        log_likelihood += np.sum(
            probabilities * occurrences[start:stop, None], axis=0
        )
    return log_likelihood


def _adaptive_joint_map(
    candidates: np.ndarray,
    observations: np.ndarray,
    occurrences: np.ndarray,
    block_counts: np.ndarray,
    phantom_counts: np.ndarray,
    alpha_mins: np.ndarray,
    gamma: float,
    coarse_points: int = 64,
) -> float:
    """先定位似然峰，再在原始候选网格上局部精搜。

    联合似然随交集基数平滑变化。完整候选域实验会重复求解数千次，若每次都
    对最多 1500 个网格点求值，代价会被无意义的远离峰值区域主导。这里先在
    原网格的等距子集上定位，再把相邻两个粗网格区间按原分辨率完整求值。
    """
    if len(candidates) == 1:
        return float(candidates[0])
    if len(candidates) <= coarse_points:
        likelihood = _joint_log_likelihood(
            observations,
            occurrences,
            candidates,
            block_counts,
            phantom_counts,
            alpha_mins,
            gamma,
        )
        return float(candidates[int(np.argmax(likelihood))])

    stride = math.ceil((len(candidates) - 1) / (coarse_points - 1))
    coarse_indexes = np.arange(0, len(candidates), stride, dtype=np.int64)
    if coarse_indexes[-1] != len(candidates) - 1:
        coarse_indexes = np.append(coarse_indexes, len(candidates) - 1)
    coarse_likelihood = _joint_log_likelihood(
        observations,
        occurrences,
        candidates[coarse_indexes],
        block_counts,
        phantom_counts,
        alpha_mins,
        gamma,
    )
    peak = int(np.argmax(coarse_likelihood))
    start = int(coarse_indexes[max(0, peak - 1)])
    stop = int(coarse_indexes[min(len(coarse_indexes) - 1, peak + 1)]) + 1
    local_candidates = candidates[start:stop]
    local_likelihood = _joint_log_likelihood(
        observations,
        occurrences,
        local_candidates,
        block_counts,
        phantom_counts,
        alpha_mins,
        gamma,
    )
    return float(local_candidates[int(np.argmax(local_likelihood))])


def map_intersection_estimate(
    reports: list[AlphaReport],
    n_rows: int,
    gamma: float,
    map_step: int,
    max_map_points: int,
    block_counts: np.ndarray | None = None,
    use_frechet_lower_bound: bool = True,
) -> float:
    """使用全部私有 Alpha 坐标的联合似然估计全局交集频数。

    两个本地块时联合分布是精确的。三个及以上本地块缺少各阶部分交集
    统计量，因此使用“所有块只共享最终公共交集”的公共交集核近似。
    """
    if not reports:
        return 0.0
    if any(report.is_complement for report in reports):
        raise ValueError("直接交集 MAP 要求上传目标集合的正向 Alpha")

    alpha_lengths = {len(report.alpha) for report in reports}
    if len(alpha_lengths) != 1:
        raise ValueError("参与组合的 Alpha 长度必须一致")

    if block_counts is None:
        block_counts = np.asarray(
            [
                private_cardinality_map_estimate(
                    report=report,
                    n_rows=n_rows,
                    gamma=gamma,
                    map_step=map_step,
                    max_map_points=max_map_points,
                )
                for report in reports
            ],
            dtype=np.float64,
        )
    else:
        block_counts = np.asarray(block_counts, dtype=np.float64)
        if block_counts.shape != (len(reports),):
            raise ValueError("边缘基数数量必须与 Alpha 报告数量一致")
        block_counts = np.clip(block_counts, 0.0, n_rows)
    if len(reports) == 1:
        return float(block_counts[0])

    lower, upper = intersection_bounds(block_counts, n_rows)
    if not use_frechet_lower_bound:
        # 消融时仅移除由边缘基数导出的 Frechet 下界；上界 min_j |B_j|
        # 仍是交集属于每个局部块的必然约束，而非额外的先验假设。
        lower = 0
    candidates = _bounded_candidate_grid(
        lower, upper, map_step, max_map_points
    )
    phantom_counts = np.asarray(
        [report.phantom_count for report in reports], dtype=np.float64
    )
    alpha_mins = np.asarray(
        [_alpha_min_for_report(report, gamma) for report in reports],
        dtype=np.float64,
    )
    observations = np.stack([report.alpha for report in reports], axis=1)
    unique_observations, occurrences = np.unique(
        observations, axis=0, return_counts=True
    )
    return _adaptive_joint_map(
        candidates=candidates,
        observations=unique_observations,
        occurrences=occurrences,
        block_counts=block_counts,
        phantom_counts=phantom_counts,
        alpha_mins=alpha_mins,
        gamma=gamma,
    )


def _nested_pair_log_probability_batch(
    observations: np.ndarray,
    candidates: np.ndarray,
    left_count: float,
    right_count: float,
    phantom_counts: np.ndarray,
    alpha_mins: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """计算 A、B、A∩B 三份 Alpha 的精确嵌套集合 PMF。

    联合键不是替换单项键：对每个坐标，它们共享同一用户秩。设 d=|A∩B|，
    则 clean CDF 为 F(t_A)^(|A|-d) F(t_B)^(|B|-d)
    F(min(t_A,t_B,t_AB))^d；各 Alpha 的幻影元素保持独立。
    """
    observations = np.asarray(observations, dtype=np.float64)
    log_terms = []
    signs = []
    for predecessors in itertools.product((0.0, 1.0), repeat=3):
        predecessor_array = np.asarray(predecessors, dtype=np.float64)
        limits = observations - predecessor_array
        valid = np.all(limits >= alpha_mins, axis=1)
        safe_limits = np.maximum(limits, 1.0)
        log_cdfs = _geometric_log_cdf_array(safe_limits, gamma)
        log_common = _geometric_log_cdf_array(
            np.min(safe_limits, axis=1), gamma
        )
        # 幻影元素只属于各自报告；A\\B、B\\A 与 A∩B 的真实元素
        # 分别贡献 F(t_A), F(t_B), F(min(t_A,t_B,t_AB))。
        base = (
            (left_count + phantom_counts[0]) * log_cdfs[:, 0]
            + (right_count + phantom_counts[1]) * log_cdfs[:, 1]
            + phantom_counts[2] * log_cdfs[:, 2]
        )
        coefficient = (
            log_common - log_cdfs[:, 0] - log_cdfs[:, 1]
        )
        log_term = base[:, None] + coefficient[:, None] * candidates[None, :]
        log_terms.append(np.where(valid[:, None], log_term, -np.inf))
        signs.append(-1.0 if int(predecessor_array.sum()) % 2 else 1.0)

    stacked = np.stack(log_terms, axis=0)
    maximum = np.max(stacked, axis=0)
    scaled = np.sum(
        np.asarray(signs, dtype=np.float64)[:, None, None]
        * np.exp(stacked - maximum[None, :, :]),
        axis=0,
    )
    return maximum + np.log(np.clip(scaled, np.finfo(float).tiny, None))


def _nested_pair_log_likelihood(
    observations: np.ndarray,
    occurrences: np.ndarray,
    candidates: np.ndarray,
    left_count: float,
    right_count: float,
    phantom_counts: np.ndarray,
    alpha_mins: np.ndarray,
    gamma: float,
) -> np.ndarray:
    likelihood = np.zeros(len(candidates), dtype=np.float64)
    for start in range(0, len(observations), 128):
        stop = min(start + 128, len(observations))
        likelihood += np.sum(
            _nested_pair_log_probability_batch(
                observations[start:stop],
                candidates,
                left_count,
                right_count,
                phantom_counts,
                alpha_mins,
                gamma,
            )
            * occurrences[start:stop, None],
            axis=0,
        )
    return likelihood


def map_nested_pair_cardinality_estimate(
    left_report: AlphaReport,
    right_report: AlphaReport,
    joint_report: AlphaReport,
    n_rows: int,
    gamma: float,
    map_step: int,
    max_map_points: int,
    left_count: float | None = None,
    right_count: float | None = None,
) -> float:
    """融合 `{a}`、`{b}` 与本地 `{a,b}` Alpha 的精确 MAP 频率估计。"""
    reports = (left_report, right_report, joint_report)
    if any(report.is_complement for report in reports):
        raise ValueError("嵌套 MAP 需要正向 Alpha")
    if len({len(report.alpha) for report in reports}) != 1:
        raise ValueError("嵌套 MAP 的 Alpha 长度必须一致")
    if left_count is None:
        left_count = private_cardinality_map_estimate(
            left_report, n_rows, gamma, map_step, max_map_points
        )
    if right_count is None:
        right_count = private_cardinality_map_estimate(
            right_report, n_rows, gamma, map_step, max_map_points
        )
    lower, upper = intersection_bounds(
        np.asarray([left_count, right_count], dtype=np.float64), n_rows
    )
    candidates = _bounded_candidate_grid(lower, upper, map_step, max_map_points)
    observations, occurrences = np.unique(
        np.stack([report.alpha for report in reports], axis=1),
        axis=0,
        return_counts=True,
    )
    phantom_counts = np.asarray(
        [report.phantom_count for report in reports], dtype=np.float64
    )
    alpha_mins = np.asarray(
        [_alpha_min_for_report(report, gamma) for report in reports],
        dtype=np.float64,
    )
    likelihood = _nested_pair_log_likelihood(
        observations,
        occurrences,
        candidates,
        float(np.clip(left_count, 0.0, n_rows)),
        float(np.clip(right_count, 0.0, n_rows)),
        phantom_counts,
        alpha_mins,
        gamma,
    )
    return float(candidates[int(np.argmax(likelihood))])
