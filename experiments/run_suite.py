from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from privfim.config import DataConfig, ExperimentConfig, ProtocolConfig
from privfim.dpfm import RankOracle
from privfim.pipeline import run_experiment
from privfim.types import (
    DIRECT_UNION_COMPLEMENT,
    FM_FULL_ESTIMATOR,
    FM_OTHER_ESTIMATOR,
    LOCAL_ITEMSET_ALPHA,
    MAP_ESTIMATOR,
    MIXED_BINNED_ITEMSET_ALPHA,
    MIXED_ITEMSET_ALPHA,
    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
    PROTOCOL_IMPLEMENTATION_VERSION,
    SINGLETON_ALPHA,
    LOCAL_TOP_SINGLETON_ALPHA,
    LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (2026, 2027, 2028, 2029, 2030)
DEFAULT_AXES = (
    "dataset",
    "k",
    "epsilon",
    "clients",
    "attribute_ratio",
    "m",
    "correlation",
    "placement",
    "tail",
    "domain",
    "scale",
    "sample_ratio",
    "feature_ratio",
    "domain_ratio",
    "candidate_pool",
    "first_stage_pool",
    "second_stage_pool",
    "first_stage_source",
    "pruning",
    "budget_split",
)
DENSE_ONLY_AXES = ("k", "epsilon", "m", "candidate_pool", "budget_split")
BASE_SWEEP_VALUES = {
    "k": (5, 10, 15, 20, 25),
    "epsilon": (0.25, 0.5, 1.0, 2.0, 4.0),
    "m": (256, 512, 1024, 2048),
    "candidate_pool": (1.0, 1.5, 2.0, 2.5, 5.0),
    "first_stage_pool": (0.5, 0.75, 1.0, 1.5, 2.0),
    "second_stage_pool": (0.5, 0.75, 1.0, 1.5, 2.0),
    "first_stage_source": ("singleton", "itemset"),
    "pruning": ("topk", "none"),
    "budget_split": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    "sample_ratio": (0.2, 0.4, 0.6, 0.8, 1.0),
    "feature_ratio": (0.2, 0.4, 0.6, 0.8, 1.0),
    "domain_ratio": (0.2, 0.4, 0.6, 0.8, 1.0),
}
DENSE_SWEEP_VALUES = {
    "k": (3, 5, 8, 10, 12, 15, 18, 20, 25, 30),
    "epsilon": (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
    "m": (128, 256, 512, 1024, 2048, 4096),
    "candidate_pool": (1, 1.5, 2, 3, 4, 5, 6, 7),
    "budget_split": (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9),
}
METRIC_NAMES = (
    "precision",
    "recall",
    "f1",
    "jaccard",
    "ncr",
    "mae",
    "rmse",
    "nmae",
    "nrmse",
    "candidate_recall",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    kind: str
    correlation: float | None = None
    layout: str | None = None
    domain_size: int | None = None
    zipf_exponent: float | None = None


@dataclass(frozen=True)
class ExperimentCase:
    dataset: DatasetSpec
    seed: int
    axes: tuple[str, ...]
    k: int = 15
    epsilon: float = 1.0
    num_clients: int = 4
    attribute_ratios: tuple[float, ...] | None = None
    m: int = 2048
    max_rows: int | None = None
    sample_ratio: float | None = None
    feature_ratio: float | None = None
    domain_ratio: float | None = None
    candidate_multiplier: float = 2.0
    first_stage_k: int | None = None
    first_stage_report_limit: int | None = None
    first_stage_multiplier: float | None = None
    second_stage_multiplier: float | None = None
    first_stage_mode: str = "singleton"
    candidate_pruning: str = "topk"
    phase1_ratio: float = 0.5
    # 允许论文复现实验显式固定纵向属性放置；旧实验默认按原列序。
    partition_strategy: str = "order"
    partition_seed: int | None = None

    def identity(self) -> tuple:
        return (
            self.dataset.name,
            self.seed,
            self.k,
            self.epsilon,
            self.num_clients,
            self.attribute_ratios,
            self.m,
            self.max_rows,
            self.sample_ratio,
            self.feature_ratio,
            self.domain_ratio,
            self.candidate_multiplier,
            self.first_stage_k,
            self.first_stage_report_limit,
            self.first_stage_multiplier,
            self.second_stage_multiplier,
            self.first_stage_mode,
            self.candidate_pruning,
            self.phase1_ratio,
            self.partition_strategy,
            self.partition_seed,
        )


def _dataset_specs() -> dict[str, DatasetSpec]:
    real_dir = PROJECT_ROOT / "data" / "real"
    synthetic_dir = PROJECT_ROOT / "data" / "synthetic"
    specs = {
        "Retail": DatasetSpec("Retail", real_dir / "Retail.csv", "real"),
        "Adult": DatasetSpec("Adult", real_dir / "Adult.csv", "real"),
        "Bank": DatasetSpec("Bank", real_dir / "Bank.csv", "real"),
        "Diabetic": DatasetSpec(
            "Diabetic", real_dir / "Diabetic.csv", "real"
        ),
        # 官网重建版：47 个 UCI 特征，ID 与 readmitted 标签已分离。
        "DiabeticUCI": DatasetSpec(
            "DiabeticUCI", real_dir / "DiabeticUCI.csv", "real"
        ),
        "Mushroom": DatasetSpec(
            "Mushroom", real_dir / "Mushroom.csv", "real"
        ),
        "CensusIncomeKDD": DatasetSpec(
            "CensusIncomeKDD", real_dir / "CensusIncomeKDD.csv", "real"
        ),
        "LetterRecognition": DatasetSpec(
            "LetterRecognition", real_dir / "LetterRecognition.csv", "real"
        ),
        "DefaultCredit": DatasetSpec(
            "DefaultCredit", real_dir / "DefaultCredit.csv", "real"
        ),
        "Covertype": DatasetSpec(
            "Covertype", real_dir / "Covertype.csv", "real"
        ),
        "KDDCup99_10pct": DatasetSpec(
            "KDDCup99_10pct", real_dir / "KDDCup99_10pct.csv", "real"
        ),
        "MiniBooNE": DatasetSpec(
            "MiniBooNE", real_dir / "MiniBooNE.csv", "real"
        ),
        "PokerHand": DatasetSpec(
            "PokerHand", real_dir / "PokerHand.csv", "real"
        ),
    }
    synthetic = (
        ("independent_uniform", 0.0, "grouped", 8, 0.0),
        ("weakcorr_uniform", 0.25, "grouped", 8, 0.0),
        ("mediumcorr_uniform", 0.5, "grouped", 8, 0.0),
        ("correlated_uniform", 0.75, "grouped", 8, 0.0),
        ("highcorr_uniform", 0.9, "grouped", 8, 0.0),
        ("correlated_zipf_grouped", 0.75, "grouped", 8, 1.2),
        ("correlated_zipf_interleaved", 0.75, "interleaved", 8, 1.2),
        ("highcorr_zipf_grouped", 0.9, "grouped", 8, 1.5),
        ("tail_zipf_0", 0.75, "grouped", 8, 0.0),
        ("tail_zipf_0_6", 0.75, "grouped", 8, 0.6),
        ("tail_zipf_1_2", 0.75, "grouped", 8, 1.2),
        ("tail_zipf_1_8", 0.75, "grouped", 8, 1.8),
        ("domain_4", 0.75, "grouped", 4, 1.2),
        ("domain_8", 0.75, "grouped", 8, 1.2),
        ("domain_16", 0.75, "grouped", 16, 1.2),
        ("domain_32", 0.75, "grouped", 32, 1.2),
    )
    for name, correlation, layout, domain_size, zipf_exponent in synthetic:
        specs[name] = DatasetSpec(
            name=name,
            csv_path=synthetic_dir / f"{name}.csv",
            kind="synthetic",
            correlation=correlation,
            layout=layout,
            domain_size=domain_size,
            zipf_exponent=zipf_exponent,
        )
    return specs


def build_cases(
    seeds: Iterable[int], selected_axes: set[str], dense_only: bool = False,
    sweep_datasets: Iterable[str] = ("Retail",),
    ks: Iterable[int] | None = None,
) -> list[ExperimentCase]:
    if dense_only and not selected_axes <= set(DENSE_ONLY_AXES):
        raise ValueError(
            "dense-only 仅支持 k、epsilon、m、candidate_pool 和 budget_split 轴"
        )
    specs = _dataset_specs()
    seeds = tuple(seeds)
    sweep_datasets = tuple(sweep_datasets)
    cases: dict[tuple, ExperimentCase] = {}
    custom_ks = None if ks is None else tuple(dict.fromkeys(ks))
    if custom_ks is not None and (not custom_ks or any(k < 2 for k in custom_ks)):
        raise ValueError("ks 至少包含一个大于等于 2 的整数")

    def sweep_values(axis: str) -> tuple[int | float, ...]:
        if axis == "k" and custom_ks is not None:
            return custom_ks
        values = DENSE_SWEEP_VALUES[axis] if dense_only else BASE_SWEEP_VALUES[axis]
        if not dense_only:
            return values
        base = set(BASE_SWEEP_VALUES[axis])
        return tuple(value for value in values if value not in base)

    def add(axis: str, dataset_name: str | None = None, **changes: object) -> None:
        if axis not in selected_axes:
            return
        # 参数消融默认仍只用 Retail；显式指定时复用相同设置覆盖多个数据集。
        if dataset_name is None:
            for name in sweep_datasets:
                add(axis, name, **changes)
            return
        for seed in seeds:
            case = ExperimentCase(
                dataset=specs[dataset_name],
                seed=seed,
                axes=(axis,),
                **changes,
            )
            key = case.identity()
            if key in cases:
                axes = tuple(sorted(set(cases[key].axes) | {axis}))
                cases[key] = replace(cases[key], axes=axes)
            else:
                cases[key] = case

    for name in (
        "Retail",
        "Adult",
        "Bank",
        "Diabetic",
        "DiabeticUCI",
        "Mushroom",
        "CensusIncomeKDD",
        "Covertype",
        "KDDCup99_10pct",
        "MiniBooNE",
        "PokerHand",
        "independent_uniform",
        "correlated_uniform",
        "correlated_zipf_grouped",
        "highcorr_zipf_grouped",
    ):
        add("dataset", name)

    for value in sweep_values("k"):
        add("k", k=value)
    for value in sweep_values("epsilon"):
        add("epsilon", epsilon=value)
    for value in (2, 4, 8):
        add("clients", num_clients=value)
    for value in (
        None,
        (1.4, 1.0, 1.0, 1.0),
        (1.8, 1.0, 1.0, 1.0),
        (2.4, 1.0, 1.0, 1.0),
        (3.0, 1.0, 1.0, 1.0),
        (4.0, 1.0, 1.0, 1.0),
        (5.0, 1.0, 1.0, 1.0),
        (7.0, 1.0, 1.0, 1.0),
        (9.0, 1.0, 1.0, 1.0),
        (13.0, 1.0, 1.0, 1.0),
    ):
        add("attribute_ratio", attribute_ratios=value)
    for value in sweep_values("m"):
        add("m", m=value)

    for name in (
        "independent_uniform",
        "weakcorr_uniform",
        "mediumcorr_uniform",
        "correlated_uniform",
        "highcorr_uniform",
    ):
        add("correlation", name)
    for name in (
        "correlated_zipf_grouped",
        "correlated_zipf_interleaved",
    ):
        add("placement", name)
    for name in (
        "tail_zipf_0",
        "tail_zipf_0_6",
        "tail_zipf_1_2",
        "tail_zipf_1_8",
    ):
        add("tail", name)
    for name in ("domain_4", "domain_8", "domain_16", "domain_32"):
        add("domain", name)
    for rows in (5_000, 10_000, 20_000, 50_000):
        add("scale", "correlated_zipf_grouped", max_rows=rows)
    for value in BASE_SWEEP_VALUES["sample_ratio"]:
        add("sample_ratio", sample_ratio=value)
    for value in BASE_SWEEP_VALUES["feature_ratio"]:
        add("feature_ratio", feature_ratio=value)
    for value in BASE_SWEEP_VALUES["domain_ratio"]:
        add("domain_ratio", domain_ratio=value)
    for value in sweep_values("candidate_pool"):
        add("candidate_pool", candidate_multiplier=value)
    for value in BASE_SWEEP_VALUES["first_stage_pool"]:
        add(
            "first_stage_pool",
            first_stage_report_limit=max(1, round(15 * value)),
            first_stage_multiplier=value,
        )
    for value in BASE_SWEEP_VALUES["second_stage_pool"]:
        add(
            "second_stage_pool",
            second_stage_multiplier=value,
        )
    for value in ("singleton", "itemset"):
        add("first_stage_source", first_stage_mode=value)
    for value in ("topk", "none"):
        add("pruning", candidate_pruning=value)
    for value in sweep_values("budget_split"):
        add("budget_split", phase1_ratio=value)

    return sorted(cases.values(), key=lambda case: str(case.identity()))


def _ratio_text(ratios: tuple[float, ...] | None, clients: int) -> str:
    values = ratios if ratios is not None else (1.0,) * clients
    return ":".join(f"{value:g}" for value in values)


def _run_id(
    case: ExperimentCase,
    estimator: str = MAP_ESTIMATOR,
    map_m_bin_count: int | None = None,
    mixed_joint_weight: float | None = None,
) -> str:
    identity = (*case.identity(), PROTOCOL_IMPLEMENTATION_VERSION)
    if map_m_bin_count is not None:
        identity = (*identity, "map_m_bin_count", map_m_bin_count)
    if mixed_joint_weight is not None:
        identity = (*identity, "mixed_joint_budget_weight", mixed_joint_weight)
    if estimator != MAP_ESTIMATOR:
        # 协议版本进入缓存键，避免复用早期语义不同的 FM 结果。
        identity = (*identity, estimator, "category-buckets-v1")
    identity = json.dumps(identity, default=str, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    prefix = "" if estimator == MAP_ESTIMATOR else f"{estimator}_"
    return f"{prefix}{case.dataset.name}_s{case.seed}_{digest}"


def _config_for(
    case: ExperimentCase,
    output_dir: Path,
    estimator: str = MAP_ESTIMATOR,
    map_modes: tuple[str, ...] | None = None,
    mixed_joint_weight: float | None = None,
    map_m_bin_count: int | None = None,
) -> ExperimentConfig:
    modes = (
        (SINGLETON_ALPHA,)
        if estimator in {FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR}
        else map_modes if map_modes is not None else ProtocolConfig().modes
    )
    return ExperimentConfig(
        data=DataConfig(
            csv_path=str(case.dataset.csv_path),
            num_clients=case.num_clients,
            max_rows=case.max_rows,
            sample_ratio=case.sample_ratio,
            feature_ratio=case.feature_ratio,
            domain_ratio=case.domain_ratio,
            transform_seed=case.seed,
            attribute_ratios=case.attribute_ratios,
            partition_seed=case.partition_seed,
            partition_strategy=case.partition_strategy,
            row_sampling_seed=case.seed,
        ),
        protocol=ProtocolConfig(
            k=case.k,
            candidate_multiplier=case.candidate_multiplier,
            first_stage_k=case.first_stage_k,
            first_stage_report_limit=(
                case.k
                if case.first_stage_report_limit is None
                else case.first_stage_report_limit
            ),
            first_stage_mode=case.first_stage_mode,
            second_stage_upload_limit=(
                None
                if case.second_stage_multiplier is None
                else max(1, round(case.k * case.second_stage_multiplier))
            ),
            candidate_pruning=case.candidate_pruning,
            min_itemset_size=1,
            max_itemset_size=4,
            epsilon=case.epsilon,
            phase1_ratio=case.phase1_ratio,
            delta=1e-5,
            m=case.m,
            gamma=1.0,
            map_step=5,
            max_map_points=1500,
            hash_block_size=64,
            seed=case.seed,
            estimator=estimator,
            mixed_joint_budget_weight=(
                ProtocolConfig().mixed_joint_budget_weight
                if mixed_joint_weight is None
                else mixed_joint_weight
            ),
            map_m_bin_count=map_m_bin_count,
            modes=modes,
        ),
        output_dir=str(output_dir),
    )


def _flatten(
    run_id: str,
    case: ExperimentCase,
    estimator: str,
    summary: dict,
) -> list[dict]:
    rows = []
    for mode, result in summary["modes"].items():
        communication = result.get("communication")
        if communication is None:
            # 兼容通信统计加入之前生成的 MAP 正式缓存。
            communication = {
                "round1_uplink_bytes": math.nan,
                "candidate_downlink_bytes": math.nan,
                "binning_downlink_bytes": math.nan,
                "round2_uplink_bytes": math.nan,
                "total_bytes": math.nan,
            }
        per_key_epsilons = [
            float(value)
            for value in result["epsilon_per_key"].values()
            if value > 0
        ]
        coordinate_epsilons = [
            float(value)
            for value in result["coordinate_epsilon"].values()
            if value > 0
        ]
        row = {
            "run_id": run_id,
            "implementation_version": summary["implementation_version"],
            "axes": "+".join(case.axes),
            "dataset": case.dataset.name,
            "dataset_kind": case.dataset.kind,
            "correlation": case.dataset.correlation,
            "layout": case.dataset.layout,
            "domain_size": case.dataset.domain_size,
            "zipf_exponent": case.dataset.zipf_exponent,
            "seed": case.seed,
            "rows": summary["dataset"]["rows"],
            "noisy_rows": summary["dataset"].get("noisy_rows", ""),
            "attributes": summary["dataset"]["attributes"],
            "domain_sizes": ":".join(
                str(value) for value in summary["dataset"].get("domain_sizes", [])
            ),
            "max_rows": case.max_rows,
            "sample_ratio": case.sample_ratio,
            "feature_ratio": case.feature_ratio,
            "domain_ratio": case.domain_ratio,
            "k": case.k,
            "epsilon": case.epsilon,
            "noisy_n_ratio": summary["config"]["protocol"].get(
                "noisy_n_ratio", ""
            ),
            "num_clients": case.num_clients,
            "attribute_ratio": _ratio_text(
                case.attribute_ratios, case.num_clients
            ),
            "attribute_counts": ":".join(
                str(len(partition))
                for partition in summary["dataset"]["partitions"]
            ),
            "m": case.m,
            "candidate_multiplier": case.candidate_multiplier,
            "first_stage_k": summary["config"]["protocol"].get(
                "first_stage_k", ""
            ),
            "first_stage_report_limit": summary["config"]["protocol"].get(
                "first_stage_report_limit", ""
            ),
            "first_stage_multiplier": case.first_stage_multiplier,
            "second_stage_multiplier": case.second_stage_multiplier,
            "second_stage_upload_limit": summary["config"]["protocol"].get(
                "second_stage_upload_limit", ""
            ),
            "first_stage_mode": summary["config"]["protocol"].get(
                "first_stage_mode", "singleton"
            ),
            "candidate_pruning": summary["config"]["protocol"].get(
                "candidate_pruning", "topk"
            ),
            "phase1_ratio": case.phase1_ratio,
            "mixed_joint_budget_weight": summary["config"]["protocol"].get(
                "mixed_joint_budget_weight", ""
            ),
            "estimator": estimator,
            "mode": mode,
            "scheme": result["scheme_label"],
            "candidate_count": result["candidate_count"],
            "candidate_count_target": summary.get("candidate_count_target", ""),
            "estimated_candidate_count": result["estimated_candidate_count"],
            "estimable_candidate_ratio": result["estimable_candidate_ratio"],
            "direct_candidate_count": summary["direct_candidate_count"],
            "report_keys": result["total_report_keys"],
            "max_client_report_keys": result["max_report_key_count"],
            "total_public_overlap_bound": result["total_public_overlap_bound"],
            "max_public_overlap_bound": result["max_public_overlap_bound"],
            "accounting_model": summary["privacy"]["accounting_model"],
            "joint_dp_status": summary["privacy"]["joint_dp_status"],
            "coordinate_epsilon_mean": statistics.fmean(coordinate_epsilons),
            "coordinate_epsilon_max": max(coordinate_epsilons),
            "overlap_rdp_weighted_overlap": result["overlap_rdp"]["weighted_overlap"],
            "overlap_rdp_coefficient": result["overlap_rdp"]["rdp_coefficient"],
            "overlap_rdp_order": result["overlap_rdp"]["rdp_order"],
            "overlap_rdp_epsilon": result["overlap_rdp_epsilon"],
            "noisy_n_epsilon_target": summary["privacy"].get(
                "noisy_n_epsilon_target", ""
            ),
            "phase1_epsilon_target": summary["privacy"]["phase1_epsilon_target"],
            "phase2_epsilon_target": summary["privacy"]["phase2_epsilon_target"],
            "overlap_rdp_budget_slack": result["overlap_rdp_budget_slack"],
            "overlap_rdp_epsilon_ratio": result["overlap_rdp_epsilon"]
            / summary["privacy"]["phase2_epsilon_target"],
            "overlap_rdp_within_phase2_budget": float(
                result["overlap_rdp_within_phase2_budget"]
            ),
            "certified_total_epsilon": result["certified_total_epsilon"],
            "total_epsilon_budget_slack": result["total_epsilon_budget_slack"],
            "certified_total_epsilon_ratio": result["certified_total_epsilon"]
            / summary["privacy"]["protocol_epsilon_target"],
            "theoretical_total_within_budget": float(
                result["theoretical_total_within_budget_under_secret_prf"]
            ),
            "alpha_values": result["alpha_values"],
            "payload_bytes": result["estimated_alpha_payload_bytes"],
            "payload_mib": result["estimated_alpha_payload_bytes"] / 2**20,
            "epsilon_per_key_mean": statistics.fmean(per_key_epsilons),
            "epsilon_per_key_min": min(per_key_epsilons),
            "round1_uplink_mib": communication["round1_uplink_bytes"] / 2**20,
            "candidate_downlink_mib": communication["candidate_downlink_bytes"] / 2**20,
            "binning_downlink_mib": communication.get("binning_downlink_bytes", 0) / 2**20,
            "round2_uplink_mib": communication["round2_uplink_bytes"] / 2**20,
            "wire_total_mib": communication["total_bytes"] / 2**20,
            "client_seconds": result["client_report_seconds"],
            "server_seconds": result["server_estimate_seconds"],
            "mode_seconds": result["elapsed_seconds"],
            "total_seconds": summary["elapsed_seconds"],
        }
        row.update(result["metrics"])
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _t_critical_95(sample_count: int) -> float:
    # Two-sided 95% Student-t critical values indexed by sample count (df=n-1).
    table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
        12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
        17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
        22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
        27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
    }
    return table.get(sample_count, 1.96)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    numeric = METRIC_NAMES + (
        "report_keys",
        "total_public_overlap_bound",
        "max_public_overlap_bound",
        "coordinate_epsilon_mean",
        "coordinate_epsilon_max",
        "overlap_rdp_weighted_overlap",
        "overlap_rdp_coefficient",
        "overlap_rdp_order",
        "overlap_rdp_epsilon",
        "noisy_rows",
        "noisy_n_epsilon_target",
        "phase1_epsilon_target",
        "phase2_epsilon_target",
        "overlap_rdp_budget_slack",
        "overlap_rdp_epsilon_ratio",
        "overlap_rdp_within_phase2_budget",
        "certified_total_epsilon",
        "total_epsilon_budget_slack",
        "certified_total_epsilon_ratio",
        "theoretical_total_within_budget",
        "alpha_values",
        "payload_mib",
        "epsilon_per_key_mean",
        "epsilon_per_key_min",
        "round1_uplink_mib",
        "candidate_downlink_mib",
        "binning_downlink_mib",
        "round2_uplink_mib",
        "wire_total_mib",
        "client_seconds",
        "server_seconds",
        "mode_seconds",
        "total_seconds",
        "candidate_count",
        "candidate_count_target",
        "estimated_candidate_count",
        "estimable_candidate_ratio",
        "direct_candidate_count",
        "max_client_report_keys",
    )
    dimensions = (
        "axes",
        "dataset",
        "dataset_kind",
        "correlation",
        "layout",
        "domain_size",
        "zipf_exponent",
        "rows",
        "attributes",
        "domain_sizes",
        "max_rows",
        "sample_ratio",
        "feature_ratio",
        "domain_ratio",
        "k",
        "epsilon",
        "noisy_n_ratio",
        "num_clients",
        "attribute_ratio",
        "attribute_counts",
        "m",
        "candidate_multiplier",
        "first_stage_k",
        "first_stage_report_limit",
        "first_stage_multiplier",
        "second_stage_multiplier",
        "second_stage_upload_limit",
        "first_stage_mode",
        "candidate_pruning",
        "phase1_ratio",
        "mixed_joint_budget_weight",
        "estimator",
        "mode",
        "scheme",
        "accounting_model",
        "joint_dp_status",
    )
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        # 历史实验 CSV 没有后续加入的隐私会计维度；保留空标签即可合并，
        # 不能把缺失字段伪装成某个已经证明的会计模型。
        grouped.setdefault(
            tuple(row.get(name, "") for name in dimensions), []
        ).append(row)

    result = []
    for key, samples in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        aggregate = dict(zip(dimensions, key, strict=True))
        aggregate["trials"] = len(samples)
        aggregate["seeds"] = ":".join(str(row["seed"]) for row in samples)
        for name in numeric:
            values = []
            for row in samples:
                raw = row.get(name, "")
                if raw in {"", "None", None}:
                    continue
                value = float(raw)
                if math.isfinite(value):
                    values.append(value)
            if not values:
                aggregate[f"{name}_mean"] = math.nan
                aggregate[f"{name}_std"] = math.nan
                aggregate[f"{name}_ci95"] = math.nan
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            ci95 = (
                _t_critical_95(len(values)) * std / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            aggregate[f"{name}_mean"] = mean
            aggregate[f"{name}_std"] = std
            aggregate[f"{name}_ci95"] = ci95
        result.append(aggregate)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行可恢复、可复现的 PrivFim 论文实验矩阵"
    )
    parser.add_argument("--output-dir", default="results/paper")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--axes", nargs="+", choices=DEFAULT_AXES, default=DEFAULT_AXES
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(sorted(_dataset_specs())),
        help="仅保留指定数据集的配置，便于独立复核某个默认实验",
    )
    parser.add_argument(
        "--sweep-datasets",
        nargs="+",
        choices=tuple(sorted(_dataset_specs())),
        default=("Retail",),
        help="将参数变化和消融轴应用到这些数据集，默认仅 Retail",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        help="仅保留指定总 epsilon 的配置，便于预算敏感性复核",
    )
    parser.add_argument("--ks", nargs="+", type=int,
                        help="覆盖 k 轴取值；不改变其他实验的默认横坐标")
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=(MAP_ESTIMATOR, FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR),
        default=(MAP_ESTIMATOR,),
        help="要运行的第二轮协议；FM 协议只运行正向类别桶模式",
    )
    parser.add_argument(
        "--map-modes",
        nargs="+",
        choices=(
            DIRECT_UNION_COMPLEMENT,
            SINGLETON_ALPHA,
            LOCAL_ITEMSET_ALPHA,
            MIXED_ITEMSET_ALPHA,
            MIXED_BINNED_ITEMSET_ALPHA,
            MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
            LOCAL_TOP_SINGLETON_ALPHA,
            LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
        ),
        default=ProtocolConfig().modes,
        help="MAP 估计器要输出的方案；可省略 MAP-S-All 以加速 S/L/M 对比",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--map-m-bin-count",
        type=int,
        help="启用 MAP-M-Bin 时的第一轮私有连续桶数",
    )
    parser.add_argument(
        "--mixed-joint-weight",
        type=float,
        help="MAP-M 分组预算开关；0 为完全关闭并逐键均分",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="仅运行连续敏感性轴中尚未覆盖的稠密横坐标点",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-oracle-cache", action="store_true")
    parser.add_argument("--oracle-cache-entries", type=int, default=2,
                        help="共享哈希秩缓存条目上限，避免大数据多 seed 累积内存")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.oracle_cache_entries < 1:
        raise ValueError("oracle-cache-entries 必须为正整数")
    if (
        (
            MIXED_BINNED_ITEMSET_ALPHA in args.map_modes
            or MIXED_PROTECTED_BINNED_ITEMSET_ALPHA in args.map_modes
        )
        and args.map_m_bin_count is None
    ):
        raise ValueError("MAP-M-Bin/BinP 需要 --map-m-bin-count")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(
        args.seeds, set(args.axes), dense_only=args.dense_only,
        sweep_datasets=args.sweep_datasets, ks=args.ks,
    )
    if args.datasets is not None:
        selected_datasets = set(args.datasets)
        cases = [case for case in cases if case.dataset.name in selected_datasets]
    if args.epsilons is not None:
        selected_epsilons = set(args.epsilons)
        cases = [case for case in cases if case.epsilon in selected_epsilons]
    if args.limit is not None:
        cases = cases[: args.limit]
    planned_runs = [
        (case, estimator)
        for case in cases
        for estimator in args.estimators
    ]

    missing = sorted(
        {str(case.dataset.csv_path) for case in cases if not case.dataset.csv_path.exists()}
    )
    if missing:
        raise FileNotFoundError(
            "实验数据不存在，请先运行 experiments/prepare_data.py:\n"
            + "\n".join(missing)
        )

    plan = {
        "created_unix": time.time(),
        "implementation_version": PROTOCOL_IMPLEMENTATION_VERSION,
        "seeds": list(args.seeds),
        "axes": list(args.axes),
        "datasets": args.datasets,
        "sweep_datasets": list(args.sweep_datasets),
        "epsilons": args.epsilons,
        "ks": args.ks,
        "dense_only": args.dense_only,
        "case_count": len(planned_runs),
        "estimators": list(args.estimators),
        "map_modes": list(args.map_modes),
        "method_count": sum(
            len(args.map_modes) if estimator == MAP_ESTIMATOR else 1
            for estimator in args.estimators
        ),
        "default_parameters": {
            "k": 15,
            "epsilon": 1.0,
            "num_clients": 4,
            "attribute_ratio": "1:1:1:1",
            "m": 2048,
            "candidate_multiplier": 2,
            "first_stage_mode": "singleton",
            "candidate_pruning": "topk",
            "phase1_ratio": 0.5,
            "mixed_joint_budget_weight": (
                ProtocolConfig().mixed_joint_budget_weight
                if args.mixed_joint_weight is None
                else args.mixed_joint_weight
            ),
            "map_m_bin_count": args.map_m_bin_count,
        },
        "cases": [
            {
                "run_id": _run_id(
                    case,
                    estimator,
                    args.map_m_bin_count,
                    args.mixed_joint_weight,
                ),
                "estimator": estimator,
                "axes": list(case.axes),
                "dataset": case.dataset.name,
                "seed": case.seed,
                "k": case.k,
                "epsilon": case.epsilon,
                "num_clients": case.num_clients,
                "attribute_ratio": _ratio_text(
                    case.attribute_ratios, case.num_clients
                ),
                "m": case.m,
                "max_rows": case.max_rows,
                "sample_ratio": case.sample_ratio,
                "feature_ratio": case.feature_ratio,
                "domain_ratio": case.domain_ratio,
                "candidate_multiplier": case.candidate_multiplier,
                "first_stage_k": case.first_stage_k,
                "first_stage_report_limit": (
                    case.k
                    if case.first_stage_report_limit is None
                    else case.first_stage_report_limit
                ),
                "first_stage_multiplier": case.first_stage_multiplier,
                "second_stage_multiplier": case.second_stage_multiplier,
                "first_stage_mode": case.first_stage_mode,
                "candidate_pruning": case.candidate_pruning,
                "phase1_ratio": case.phase1_ratio,
            }
            for case, estimator in planned_runs
        ],
    }
    (output_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows: list[dict] = []
    failures: list[dict] = []
    oracle_cache: dict[tuple[int, int, float, int], RankOracle] | None = (
        None if args.no_oracle_cache else {}
    )
    for index, (case, estimator) in enumerate(planned_runs, start=1):
        run_id = _run_id(
            case,
            estimator,
            args.map_m_bin_count,
            args.mixed_joint_weight,
        )
        run_dir = raw_dir / run_id
        summary_path = run_dir / "summary.json"
        try:
            if summary_path.exists() and not args.force:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                logging.info(
                    "[%d/%d] 恢复已有结果 %s",
                    index,
                    len(planned_runs),
                    run_id,
                )
            else:
                logging.info(
                    "[%d/%d] 运行 %s axes=%s dataset=%s seed=%d",
                    index,
                    len(planned_runs),
                    run_id,
                    "+".join(case.axes),
                    case.dataset.name,
                    case.seed,
                )
                summary = run_experiment(
                    _config_for(
                        case,
                        run_dir,
                        estimator,
                        map_modes=tuple(args.map_modes),
                        mixed_joint_weight=args.mixed_joint_weight,
                        map_m_bin_count=args.map_m_bin_count,
                    ),
                    oracle_cache=oracle_cache,
                )
            rows.extend(_flatten(run_id, case, estimator, summary))
            if oracle_cache is not None:
                while len(oracle_cache) > args.oracle_cache_entries:
                    del oracle_cache[next(iter(oracle_cache))]
        except Exception as exc:  # 批量实验需要记录失败并支持断点续跑。
            failure = {
                "run_id": run_id,
                "dataset": case.dataset.name,
                "seed": case.seed,
                "estimator": estimator,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            logging.exception("实验失败: %s", run_id)
            if args.stop_on_error:
                raise

        _write_csv(output_dir / "runs.csv", rows)
        aggregated = aggregate_rows(rows)
        _write_csv(output_dir / "aggregated.csv", aggregated)
        status = {
            "planned_cases": len(planned_runs),
            "completed_cases": len({row["run_id"] for row in rows}),
            "failed_cases": len(failures),
            "failures": failures,
        }
        (output_dir / "status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(
        f"实验矩阵完成: {len({row['run_id'] for row in rows})}/{len(planned_runs)} 个配置，"
        f"失败 {len(failures)} 个；结果位于 {output_dir}"
    )


if __name__ == "__main__":
    main()
