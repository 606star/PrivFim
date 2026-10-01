from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeAlias

import numpy as np


# Included in experiment cache keys; update when protocol semantics change.
PROTOCOL_IMPLEMENTATION_VERSION = "positive-alpha-map-v22-plan-budget"


# Represent items as (attribute ID, value) to distinguish values across attributes.
Item: TypeAlias = tuple[int, int]
Itemset: TypeAlias = tuple[Item, ...]

# FM-Other uses this sentinel for attribute values not targeted by candidates S.
# Data use int64, so reserve its minimum value and reject collisions on the client.
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
# VertiMRF-style domain-compression ablation; exact MAP-M remains a separate mode.
MIXED_BINNED_ITEMSET_ALPHA = "mixed_binned_itemset_alpha"
# Protected binning: isolate high-value candidate values and compress the long tail.
MIXED_PROTECTED_BINNED_ITEMSET_ALPHA = "mixed_protected_binned_itemset_alpha"
MIXED_COVER_ITEMSET_ALPHA = "mixed_cover_itemset_alpha"
# Ablation only: generate and budget MAP-M joint Alpha, but ignore joint keys at the server.
MIXED_DUMMY_ITEMSET_ALPHA = "mixed_dummy_itemset_alpha"
# Budget-controlled MAP-L: retain relevant items and add only the most valuable local joint keys.
LOCAL_TOP_ITEMSET_ALPHA = "local_top_itemset_alpha"
# Frequency-guessing ablation: select keys by raw local counts, without SVSM products.
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
    # Keep the legacy identifier for compatibility; this is MAP-S without top-2k filtering.
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
    """Return a stable itemset representation suitable for dictionary keys."""
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
    # The dummy ablation publishes and budgets this report but excludes it from MAP.
    used_in_estimation: bool = True


@dataclass(frozen=True)
class CandidateEstimate:
    itemset: Itemset
    estimated_count: float
    guessed_count: float
    local_blocks: tuple[Itemset, ...]
