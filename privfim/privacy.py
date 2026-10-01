from __future__ import annotations

import math
from dataclasses import dataclass

from .types import Itemset


NOMINAL_ACCOUNTING_MODEL = "report_allocation_with_overlap_rdp_audit"
JOINT_DP_STATUS = "overlap_rdp_accounted_secret_prf_required"


MeasurementGroup = tuple[int, ...]


@dataclass(frozen=True)
class PerKeyBudget:
    epsilon: float
    delta: float


@dataclass(frozen=True)
class GroupedBudgetAllocation:
    """Budgets and public group structure for attribute-projection grouping.

    Keys within a group share attributes but differ in values, so a record matches
    at most one key. Under add/remove adjacency, compose keys in parallel within
    each group and compose groups sequentially.
    """

    groups: dict[MeasurementGroup, tuple[Itemset, ...]]
    budget_by_key: dict[Itemset, PerKeyBudget]


@dataclass(frozen=True)
class OverlapRdpAccount:
    weighted_overlap: float
    rdp_coefficient: float
    rdp_order: float
    rdp_epsilon: float
    target_delta: float

    def to_dict(self) -> dict[str, float]:
        return {
            "weighted_overlap": self.weighted_overlap,
            "rdp_coefficient": self.rdp_coefficient,
            "rdp_order": self.rdp_order,
            "rdp_epsilon": self.rdp_epsilon,
            "target_delta": self.target_delta,
        }


def split_budget(epsilon: float, delta: float, key_count: int) -> PerKeyBudget:
    """Split budgets equally across keys or measurement groups; audit tuples with RDP."""
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon/delta 不合法")
    if key_count <= 0:
        raise ValueError("key_count 必须大于 0")
    return PerKeyBudget(epsilon=epsilon / key_count, delta=delta / key_count)


def split_weighted_budget(
    epsilon: float, delta: float, weights: list[float]
) -> list[PerKeyBudget]:
    """Allocate budgets by public key-type weights, preserving total epsilon/delta."""
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon/delta 不合法")
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("预算权重必须全部为正数")
    total = float(sum(weights))
    return [
        PerKeyBudget(epsilon=epsilon * weight / total, delta=delta * weight / total)
        for weight in weights
    ]


def measurement_group(key: Itemset) -> MeasurementGroup:
    """Return a key's local attribute projection, ignoring its values."""
    if not key:
        raise ValueError("报告键不能为空")
    attributes = tuple(attr for attr, _ in key)
    if len(attributes) != len(set(attributes)):
        raise ValueError("同一报告键不能重复包含属性")
    return attributes


def measurement_groups(
    keys: tuple[Itemset, ...],
) -> dict[MeasurementGroup, tuple[Itemset, ...]]:
    """Group public report keys by local projection into mutually exclusive value bins."""
    grouped: dict[MeasurementGroup, list[Itemset]] = {}
    for key in sorted(keys):
        grouped.setdefault(measurement_group(key), []).append(key)
    return {
        group: tuple(grouped[group])
        for group in sorted(grouped)
    }


def split_grouped_budget(
    epsilon: float,
    delta: float,
    keys: tuple[Itemset, ...],
) -> GroupedBudgetAllocation:
    """Split budgets equally across projection groups and reuse them within disjoint bins.

    For example, `{a=0}` and `{a=1}` share group `(a,)`, while `{a=0,b=1}`
    and `{a=1,b=1}` share `(a,b)`. Each group receives `epsilon / G` and
    `delta / G`, without further division by its bin count. A record can match
    multiple groups, so `public_measurement_group_overlap_bound` composes them sequentially.
    """
    groups = measurement_groups(keys)
    group_budget = split_budget(epsilon, delta, len(groups))
    budget_by_key = {
        key: group_budget
        for group_keys in groups.values()
        for key in group_keys
    }
    return GroupedBudgetAllocation(groups=groups, budget_by_key=budget_by_key)


def dpfm_coordinate_epsilon(
    epsilon: float,
    delta: float,
    repetitions: int,
) -> float:
    """Convert a DPFM vector budget to the pure-DP parameter for one hash coordinate."""
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon/delta 不合法")
    if repetitions <= 0:
        raise ValueError("repetitions 必须大于 0")
    return epsilon / (4.0 * math.sqrt(repetitions * math.log(1.0 / delta)))


def overlap_rdp_account(
    repetitions: int,
    target_delta: float,
    overlap_bounds: dict[str, int],
    coordinate_epsilons: dict[str, float],
) -> OverlapRdpAccount:
    """Bound approximate DP from RDP for overlapping-key tuples with shared hashes.

    For client j, one record matches at most q_j keys, with scalar DPFM
    parameter eta_j per coordinate. Extending the VertiMRF conditional-chain argument gives
    rho(lambda) <= 2*m*lambda*sum_j(q_j*eta_j^2)。
    """
    if repetitions <= 0:
        raise ValueError("repetitions 必须大于 0")
    if not 0 < target_delta < 1:
        raise ValueError("target_delta 必须在 (0, 1) 内")
    if set(overlap_bounds) != set(coordinate_epsilons):
        raise ValueError("重叠界与坐标预算的客户端集合必须一致")
    if any(value < 0 for value in overlap_bounds.values()):
        raise ValueError("重叠界不能为负数")
    if any(value < 0 for value in coordinate_epsilons.values()):
        raise ValueError("坐标 epsilon 不能为负数")

    weighted_overlap = sum(
        overlap_bounds[client_id] * coordinate_epsilons[client_id] ** 2
        for client_id in overlap_bounds
    )
    if weighted_overlap == 0:
        return OverlapRdpAccount(
            weighted_overlap=0.0,
            rdp_coefficient=0.0,
            rdp_order=2.0,
            rdp_epsilon=0.0,
            target_delta=target_delta,
        )

    coefficient = 2.0 * repetitions * weighted_overlap
    log_delta = math.log(1.0 / target_delta)
    optimal_order = 1.0 + math.sqrt(log_delta / coefficient)
    rdp_order = max(2.0, optimal_order)
    epsilon = coefficient * rdp_order + log_delta / (rdp_order - 1.0)
    return OverlapRdpAccount(
        weighted_overlap=weighted_overlap,
        rdp_coefficient=coefficient,
        rdp_order=rdp_order,
        rdp_epsilon=epsilon,
        target_delta=target_delta,
    )


def public_overlap_bound(keys: tuple[Itemset, ...]) -> int:
    """Conservatively bound the number of public report keys one record can match."""
    singleton_attributes = {key[0][0] for key in keys if len(key) == 1}
    multi_itemset_count = sum(len(key) > 1 for key in keys)
    return len(singleton_attributes) + multi_itemset_count


def public_measurement_group_overlap_bound(keys: tuple[Itemset, ...]) -> int:
    """Bound the number of measurement groups one record can match.

    Distinct keys with the same projection differ in a value, allowing at most one
    match per group. Different groups may all match, so the group count is a safe public bound.
    """
    return len(measurement_groups(keys))
