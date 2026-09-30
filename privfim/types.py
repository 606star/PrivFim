from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeAlias

import numpy as np


# 进入实验缓存键；协议语义改变时必须更新，防止混用旧结果。
PROTOCOL_IMPLEMENTATION_VERSION = "positive-alpha-map-v22-plan-budget"


# Item 使用 (属性编号, 属性取值) 表示，避免不同属性中同名取值发生冲突。
Item: TypeAlias = tuple[int, int]
Itemset: TypeAlias = tuple[Item, ...]

# FM-Other 用这个保留值表示一个属性中所有未被候选 S 点名的取值。
# 当前数据载入为 int64，因此选择 int64 最小值并在客户端拒绝真实数据冲突。
FM_OTHER_VALUE = -(1 << 63)

MAP_ESTIMATOR = "map"
MAP_NO_FRECHET_ESTIMATOR = "map_no_frechet"
MAP_BOUNDS_ONLY_ESTIMATOR = "map_bounds_only"
FM_INVERSE_ESTIMATOR = "fm_inverse"
FM_FULL_ESTIMATOR = "fm_full"
FM_OTHER_ESTIMATOR = "fm_other"
ESTIMATORS = (
    MAP_ESTIMATOR,
    MAP_NO_FRECHET_ESTIMATOR,
    MAP_BOUNDS_ONLY_ESTIMATOR,
    FM_INVERSE_ESTIMATOR,
    FM_FULL_ESTIMATOR,
    FM_OTHER_ESTIMATOR,
)
ESTIMATOR_LABELS = {
    MAP_ESTIMATOR: "MAP-Direct",
    MAP_NO_FRECHET_ESTIMATOR: "MAP-NoLowerBound",
    MAP_BOUNDS_ONLY_ESTIMATOR: "MAP-BoundsOnly",
    FM_INVERSE_ESTIMATOR: "FM-Inverse",
    FM_FULL_ESTIMATOR: "FM-Full",
    FM_OTHER_ESTIMATOR: "FM-Other",
}

SINGLETON_ALPHA = "singleton_alpha"
LOCAL_ITEMSET_ALPHA = "local_itemset_alpha"
MIXED_ITEMSET_ALPHA = "mixed_itemset_alpha"
# VertiMRF 风格的高维值域压缩消融；精确 MAP-M 保持独立模式。
MIXED_BINNED_ITEMSET_ALPHA = "mixed_binned_itemset_alpha"
# 保护式分箱：候选高价值 value 单独成桶，剩余长尾再压缩。
MIXED_PROTECTED_BINNED_ITEMSET_ALPHA = "mixed_protected_binned_itemset_alpha"
MIXED_COVER_ITEMSET_ALPHA = "mixed_cover_itemset_alpha"
# 仅用于消融：生成与 MAP-M 相同的联合 Alpha 并消耗预算，但服务端不使用联合键。
MIXED_DUMMY_ITEMSET_ALPHA = "mixed_dummy_itemset_alpha"
# MAP-L 的预算受控变体：保留候选相关单项，并只加入最有价值的本地联合键。
LOCAL_TOP_ITEMSET_ALPHA = "local_top_itemset_alpha"
# 频率猜测消融：不使用 SVSM 乘积，仅按本地原始频数选择键。
LOCAL_TOP_SINGLETON_ALPHA = "local_top_singleton_alpha"
LOCAL_TOP_ITEMSET_COMPONENT_ALPHA = "local_top_itemset_component_alpha"
DIRECT_UNION_COMPLEMENT = "direct_union_complement"
REPORT_MODES = (
    DIRECT_UNION_COMPLEMENT,
    SINGLETON_ALPHA,
    LOCAL_ITEMSET_ALPHA,
    MIXED_ITEMSET_ALPHA,
    MIXED_BINNED_ITEMSET_ALPHA,
    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
    MIXED_COVER_ITEMSET_ALPHA,
    MIXED_DUMMY_ITEMSET_ALPHA,
    LOCAL_TOP_ITEMSET_ALPHA,
    LOCAL_TOP_SINGLETON_ALPHA,
    LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
)
REPORT_MODE_LABELS = {
    # 保留旧代码标识以兼容配置；该模式现在表示不经 Top-2k 筛选的 MAP-S。
    DIRECT_UNION_COMPLEMENT: "MAP-S-All",
    SINGLETON_ALPHA: "MAP-S",
    LOCAL_ITEMSET_ALPHA: "MAP-L",
    MIXED_ITEMSET_ALPHA: "MAP-M",
    MIXED_BINNED_ITEMSET_ALPHA: "MAP-M-Bin",
    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA: "MAP-M-BinP",
    MIXED_COVER_ITEMSET_ALPHA: "MAP-M-Cover",
    MIXED_DUMMY_ITEMSET_ALPHA: "MAP-M-Cover-Dummy",
    LOCAL_TOP_ITEMSET_ALPHA: "MAP-L-Top",
    LOCAL_TOP_SINGLETON_ALPHA: "MAP-NoGuess-Items",
    LOCAL_TOP_ITEMSET_COMPONENT_ALPHA: "MAP-NoGuess-ItemsetComponents",
}


def canonical_itemset(items: Iterable[Item]) -> Itemset:
    """生成稳定、可作为字典键的项集表示。"""
    result = tuple(sorted(set(items), key=lambda item: (item[0], item[1])))
    attrs = [item[0] for item in result]
    if len(attrs) != len(set(attrs)):
        raise ValueError(f"同一项集中不能包含同一属性的多个取值: {result}")
    return result


def itemset_text(itemset: Itemset) -> str:
    parts = (
        f"a{attr}=OTHER" if value == FM_OTHER_VALUE else f"a{attr}={value}"
        for attr, value in itemset
    )
    return "{" + ", ".join(parts) + "}"


@dataclass(frozen=True)
class NoisyCountReport:
    client_id: str
    counts: dict[Item, float]
    epsilon: float
    noisy_n: float | None = None
    n_epsilon: float = 0.0


@dataclass(frozen=True)
class Candidate:
    itemset: Itemset
    score: float
    guessed_count: float


@dataclass(frozen=True)
class AlphaReport:
    client_id: str
    key: Itemset
    alpha: np.ndarray
    epsilon: float
    delta: float
    phantom_count: float
    is_complement: bool = False
    coordinate_epsilon: float | None = None
    # Dummy 消融仍然发布并计入预算，但服务端故意不将该报告纳入 MAP。
    used_in_estimation: bool = True


@dataclass(frozen=True)
class CandidateEstimate:
    itemset: Itemset
    estimated_count: float
    guessed_count: float
    local_blocks: tuple[Itemset, ...]
