from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .types import (
    DIRECT_UNION_COMPLEMENT,
    ESTIMATORS,
    LOCAL_ITEMSET_ALPHA,
    LOCAL_TOP_ITEMSET_ALPHA,
    MIXED_BINNED_ITEMSET_ALPHA,
    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
    MAP_ESTIMATOR,
    MIXED_COVER_ITEMSET_ALPHA,
    MIXED_ITEMSET_ALPHA,
    REPORT_MODES,
    SINGLETON_ALPHA,
    LOCAL_TOP_SINGLETON_ALPHA,
    LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
)


@dataclass(frozen=True)
class DataConfig:
    csv_path: str = "data/Retail.csv"
    num_clients: int = 4
    max_rows: int | None = None
    # 以完整数据集行数为基准的抽样比例；与 max_rows 二选一。
    sample_ratio: float | None = None
    # 实验 5/6 的表示变换比例。变换在垂直划分前完成。
    feature_ratio: float | None = None
    domain_ratio: float | None = None
    transform_seed: int | None = None
    attribute_ratios: tuple[float, ...] | None = None
    partition_seed: int | None = None
    # auto 保持旧行为：给定 partition_seed 时随机打散，否则按原列序。
    partition_strategy: str = "auto"
    row_sampling_seed: int | None = 2026


@dataclass(frozen=True)
class ProtocolConfig:
    k: int = 15
    candidate_multiplier: float = 2.0
    # 第二阶段每个客户端的单项/联合键总上限；None 沿用 k，0 表示不限。
    second_stage_upload_limit: int | None = None
    # 第一阶段保留的频繁单项数；None 表示沿用最终输出 k。
    first_stage_k: int | None = None
    # 每个客户端第一轮最多上传的加噪单项数；先对完整公开值域加噪，
    # 再按 noisy count 选择 Top-P，因此键选择是 DP 输出的后处理。
    first_stage_report_limit: int | None = None
    # singleton 保持当前“先筛单项、再组合”的 SVSM；itemset 先在同一
    # 第一轮 DP 单项直方图上枚举并优先保留联合项集，用于候选优先级消融。
    first_stage_mode: str = "singleton"
    # topk 为当前候选剪枝；none 保留值域内的全部合法候选。
    candidate_pruning: str = "topk"
    min_itemset_size: int = 1
    max_itemset_size: int = 4

    epsilon: float = 1.0
    # Fraction of the first-round budget reserved for the independently released N.
    # Thus eps_N = noisy_n_ratio * phase1_ratio * epsilon.
    noisy_n_ratio: float = 0.1
    phase1_ratio: float = 0.5
    delta: float = 1e-5

    m: int = 2048
    gamma: float = 1.0
    map_step: int = 5
    max_map_points: int = 1500
    hash_block_size: int = 64
    seed: int = 2026
    estimator: str = MAP_ESTIMATOR
    local_key_policy: str = "required"
    local_projection_limit: int | None = None
    local_joint_budget_weight: float = 1.0
    # 正值启用 MAP-M 的属性投影组预算；数值大小仅为兼容旧配置保留。
    # 设为 0 时关闭分组复用，对全部上传键逐键均分预算。
    mixed_joint_budget_weight: float = 4.0
    # 可选 MAP-M-Bin：由第一轮私有直方图将属性压缩为至多该数目的连续桶。
    # None 保持精确 MAP-M；桶内恢复只读取第一轮已经私有化的直方图。
    map_m_bin_count: int | None = None

    modes: tuple[str, ...] = field(
        default_factory=lambda: (
            DIRECT_UNION_COMPLEMENT,
            SINGLETON_ALPHA,
            LOCAL_ITEMSET_ALPHA,
            MIXED_ITEMSET_ALPHA,
            LOCAL_TOP_SINGLETON_ALPHA,
            LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
        )
    )


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    output_dir: str = "results"

    def validate(self) -> None:
        p = self.protocol
        if self.data.num_clients < 1:
            raise ValueError("num_clients 必须大于 0")
        if self.data.max_rows is not None and self.data.sample_ratio is not None:
            raise ValueError("max_rows 与 sample_ratio 不能同时设置")
        for name, value in (
            ("sample_ratio", self.data.sample_ratio),
            ("feature_ratio", self.data.feature_ratio),
            ("domain_ratio", self.data.domain_ratio),
        ):
            if value is not None and not 0 < value <= 1:
                raise ValueError(f"{name} 必须在 (0, 1] 内")
        if self.data.attribute_ratios is not None:
            if len(self.data.attribute_ratios) != self.data.num_clients:
                raise ValueError("attribute_ratios 长度必须等于 num_clients")
            if any(value <= 0 for value in self.data.attribute_ratios):
                raise ValueError("attribute_ratios 必须全部为正数")
        if self.data.partition_strategy not in {"auto", "order", "random", "affinity"}:
            raise ValueError("partition_strategy 必须是 auto、order、random 或 affinity")
        if p.k < 2:
            raise ValueError("k 至少为 2")
        if p.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier 必须大于 0")
        if not math.isfinite(p.candidate_multiplier):
            raise ValueError("candidate_multiplier 必须是有限数")
        if p.second_stage_upload_limit is not None and (
            type(p.second_stage_upload_limit) is not int
            or p.second_stage_upload_limit < 0
        ):
            raise ValueError("second_stage_upload_limit 必须为非负整数或 None")
        if p.first_stage_k is not None and p.first_stage_k < 1:
            raise ValueError("first_stage_k 必须大于 0")
        if p.first_stage_report_limit is not None and p.first_stage_report_limit < 1:
            raise ValueError("first_stage_report_limit 必须大于 0")
        if p.first_stage_mode not in {"singleton", "itemset"}:
            raise ValueError("first_stage_mode 必须是 singleton 或 itemset")
        if p.candidate_pruning not in {"topk", "none"}:
            raise ValueError("candidate_pruning 必须是 topk 或 none")
        if not 0 < p.epsilon or not 0 < p.delta < 1:
            raise ValueError("epsilon 必须大于 0，delta 必须在 (0, 1) 内")
        if not 0 < p.noisy_n_ratio < 1:
            raise ValueError("noisy_n_ratio 必须在 (0, 1) 内")
        if not 0 < p.phase1_ratio < 1:
            raise ValueError("phase1_ratio 必须在 (0, 1) 内")
        if p.m < 8 or p.gamma <= 0:
            raise ValueError("m 至少为 8，gamma 必须大于 0")
        if p.min_itemset_size < 1 or p.max_itemset_size < p.min_itemset_size:
            raise ValueError("项集大小范围不合法")
        if p.estimator not in {*ESTIMATORS, "fm"}:
            raise ValueError(
            "estimator 必须是 map、map_no_frechet、map_bounds_only、"
                "fm_inverse、fm_full 或 fm_other"
            )
        if p.local_key_policy not in {"top_k", "all", "required", "capped"}:
            raise ValueError(
                "local_key_policy 必须是 top_k、all、required 或 capped"
            )
        if p.local_projection_limit is not None and p.local_projection_limit < 0:
            raise ValueError("local_projection_limit 不能为负数")
        if p.local_joint_budget_weight <= 0:
            raise ValueError("local_joint_budget_weight 必须大于 0")
        if p.mixed_joint_budget_weight < 0:
            raise ValueError("mixed_joint_budget_weight 不能小于 0")
        if p.map_m_bin_count is not None and p.map_m_bin_count < 1:
            raise ValueError("map_m_bin_count 必须大于 0")
        binned_modes = {
            MIXED_BINNED_ITEMSET_ALPHA,
            MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
        }
        if set(p.modes) & binned_modes and p.map_m_bin_count is None:
            raise ValueError("MAP-M-Bin/BinP 必须设置 map_m_bin_count")
        if set(p.modes) & binned_modes and p.estimator != MAP_ESTIMATOR:
            raise ValueError("MAP-M-Bin/BinP 只支持 map 估计器")
        if p.local_key_policy == "capped" and p.local_projection_limit is None:
            raise ValueError("capped 策略必须设置 local_projection_limit")
        unknown_modes = set(p.modes) - set(REPORT_MODES)
        if unknown_modes:
            raise ValueError(f"未知第二轮方案: {sorted(unknown_modes)}")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    data_raw = dict(raw.get("data", {}))
    if data_raw.get("attribute_ratios") is not None:
        data_raw["attribute_ratios"] = tuple(data_raw["attribute_ratios"])
    csv_path = Path(data_raw.get("csv_path", DataConfig.csv_path))
    if not csv_path.is_absolute():
        # 数据路径相对于配置文件所在目录解析。
        data_raw["csv_path"] = str((config_path.parent / csv_path).resolve())

    protocol_raw = dict(raw.get("protocol", {}))
    if "modes" in protocol_raw:
        protocol_raw["modes"] = tuple(protocol_raw["modes"])

    output_dir = Path(raw.get("output_dir", "results"))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()

    config = ExperimentConfig(
        data=DataConfig(**data_raw),
        protocol=ProtocolConfig(**protocol_raw),
        output_dir=str(output_dir),
    )
    config.validate()
    return config
