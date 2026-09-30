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
    """按属性投影分组后的预算和公开分组结构。

    同一组中的键有相同属性集合、不同属性取值，因此一条记录至多命中其中
    一个键。这里采用 add/remove 邻接关系：组内键可并行组合，组间再顺序组合。
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
    """按报告原语（键或测量组）均分预算；tuple 另由 RDP 审计。"""
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon/delta 不合法")
    if key_count <= 0:
        raise ValueError("key_count 必须大于 0")
    return PerKeyBudget(epsilon=epsilon / key_count, delta=delta / key_count)


def split_weighted_budget(
    epsilon: float, delta: float, weights: list[float]
) -> list[PerKeyBudget]:
    """按公开键类型权重分配预算；总 epsilon/delta 不变。"""
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
    """返回键对应的本地属性投影，忽略具体属性取值。"""
    if not key:
        raise ValueError("报告键不能为空")
    attributes = tuple(attr for attr, _ in key)
    if len(attributes) != len(set(attributes)):
        raise ValueError("同一报告键不能重复包含属性")
    return attributes


def measurement_groups(
    keys: tuple[Itemset, ...],
) -> dict[MeasurementGroup, tuple[Itemset, ...]]:
    """按本地投影归并公开报告键，组内键是互斥值桶。"""
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
    """将预算均分到属性投影组，并复用于组内互斥的值桶。

    例如 `{a=0}`、`{a=1}` 同属 `(a,)` 组；`{a=0,b=1}`、
    `{a=1,b=1}` 同属 `(a,b)` 组。每个组获得 `epsilon / G`、
    `delta / G`，而不是再按组内桶数量稀释。组间可能同时命中，故在
    `public_measurement_group_overlap_bound` 中按组顺序组合。
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
    """将一个 DPFM 向量预算转换成单个哈希坐标的纯 DP 参数。"""
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
    """计算共享哈希重叠键 tuple 的保守 RDP 到近似 DP 界。

    对客户端 j，一条记录至多命中 q_j 个键，且每个坐标的标量
    DPFM 参数为 eta_j。VertiMRF 条件链证明推广给出
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
    """由公开报告键给出单条记录可同时命中键数的保守上界。"""
    singleton_attributes = {key[0][0] for key in keys if len(key) == 1}
    multi_itemset_count = sum(len(key) > 1 for key in keys)
    return len(singleton_attributes) + multi_itemset_count


def public_measurement_group_overlap_bound(keys: tuple[Itemset, ...]) -> int:
    """分组发布中一条记录可命中的测量组数上界。

    同一属性投影的两个键只可能在某个属性取值不同，因此组内最多命中一个；
    不同投影组保守地允许同时命中，故组数本身是公开的安全上界。
    """
    return len(measurement_groups(keys))
