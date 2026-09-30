from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.run_suite import (
    DEFAULT_SEEDS,
    DatasetSpec,
    ExperimentCase,
    _dataset_specs,
    _write_csv,
    aggregate_rows,
)
from privfim.candidate import construct_direct_candidates, construct_svsm_candidates, select_frequent_singletons
from privfim.client import VerticalClient
from privfim.data import load_vertical_csv
from privfim.dpfm import stable_seed
from privfim.fo import (
    oue_column_sums,
    oue_encode,
    oue_intersection_estimate,
    oue_parameters,
    oue_positive_membership,
)
from privfim.metrics import evaluate_estimates, exact_candidate_supports
from privfim.server import PrivFimServer
from privfim.types import (
    Candidate,
    CandidateEstimate,
    LOCAL_ITEMSET_ALPHA,
    MIXED_ITEMSET_ALPHA,
    SINGLETON_ALPHA,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Adult", "Bank", "CensusIncomeKDD", "LetterRecognition", "DefaultCredit", "Covertype", "PokerHand", "MiniBooNE")
IMPLEMENTATION_VERSION = "fo-m-v6-plan-budget"


def _clients(dataset) -> list[VerticalClient]:
    return [
        VerticalClient(
            client_id=f"client_{index}",
            attributes=partition,
            local_data=dataset.data[:, partition],
            domains=dataset.domains,
        )
        for index, partition in enumerate(dataset.partitions)
    ]


def _candidate_sets(
    noisy_counts: dict[tuple[int, int], float],
    n_rows: int,
    k: int,
    candidate_multiplier: float,
    first_stage_k: int | None = None,
) -> tuple[list[Candidate], list[Candidate]]:
    frequent_items = select_frequent_singletons(
        noisy_counts, limit=k if first_stage_k is None else first_stage_k
    )
    candidates = construct_svsm_candidates(
        frequent_items=frequent_items,
        n_rows=n_rows,
        candidate_count=math.ceil(candidate_multiplier * k),
        min_size=1,
        max_size=4,
    )
    direct = construct_direct_candidates(
        frequent_items=frequent_items,
        n_rows=n_rows,
        min_size=1,
        max_size=4,
    )
    if len(candidates) < k:
        raise RuntimeError(f"候选数不足: {len(candidates)}")
    return candidates, direct


def _cdp_round1(
    clients: list[VerticalClient], epsilon: float, n_epsilon: float, seed: int
) -> tuple[dict[tuple[int, int], float], int, int]:
    """可信中心在完整表上做 Laplace 单项计数；零通信是模型定义的一部分。"""
    per_client = epsilon / len(clients)
    noisy_counts: dict[tuple[int, int], float] = {}
    released_values = 0
    reports = []
    for client in clients:
        report = client.round1_report(
            per_client,
            seed,
            n_epsilon=n_epsilon / len(clients),
        )
        reports.append(report)
        noisy_counts.update(report.counts)
        released_values += len(report.counts)
    noisy_n = max(1.0, PrivFimServer.aggregate_noisy_n(reports))
    return noisy_counts, released_values, noisy_n


def _ldp_round1(
    clients: list[VerticalClient],
    epsilon: float,
    seed: int,
    n_rows: int,
    report_limit: int | None = None,
) -> tuple[dict[tuple[int, int], float], int]:
    """每列完整值域先做 OUE，再按私有化估计选择本地 Top-P 上报。

    键选择仅依赖 OUE 输出，因此是隐私机制的后处理。通信量按最终实际
    上报的 OUE 列计算，使 P_i 消融与 MAP/FM 的第一轮口径一致。
    """
    per_client = epsilon / len(clients)
    noisy_counts: dict[tuple[int, int], float] = {}
    uplink_bits = 0
    for client in clients:
        epsilon_attr = per_client / len(client.attributes)
        client_counts: dict[tuple[int, int], float] = {}
        for local_index, attr in enumerate(client.attributes):
            domain = client.domains[attr] if client.domains is not None else tuple(np.unique(client.local_data[:, local_index]))
            column_sums = oue_column_sums(
                client.local_data[:, local_index],
                domain,
                epsilon_attr,
                stable_seed(seed, "ldp-oue-round1", client.client_id, attr),
            )
            p, q = oue_parameters(epsilon_attr)
            estimates = (column_sums - client.n_rows * q) / (p - q)
            for value_index, value in enumerate(domain):
                client_counts[(attr, int(value))] = float(
                    np.clip(
                        estimates[value_index],
                        0.0,
                        n_rows,
                    )
                )
        if report_limit is not None:
            client_counts = dict(
                sorted(
                    client_counts.items(),
                    key=lambda pair: (-pair[1], pair[0]),
                )[:report_limit]
            )
        noisy_counts.update(client_counts)
        uplink_bits += client.n_rows * len(client_counts)
    return noisy_counts, uplink_bits


def _cdp_estimates(
    dataset,
    candidates: list[Candidate],
    epsilon: float,
    seed: int,
    n_rows: int,
) -> list[CandidateEstimate]:
    epsilon_query = epsilon / len(candidates)
    rng = np.random.default_rng(stable_seed(seed, "cdp-laplace-round2"))
    estimates = []
    for candidate in candidates:
        count = dataset.support(candidate.itemset)
        noisy = float(
            np.clip(count + rng.laplace(0.0, 1.0 / epsilon_query), 0.0, n_rows)
        )
        estimates.append(
            CandidateEstimate(
                itemset=candidate.itemset,
                estimated_count=noisy,
                guessed_count=candidate.guessed_count,
                local_blocks=(candidate.itemset,),
            )
        )
    return sorted(estimates, key=lambda estimate: (-estimate.estimated_count, estimate.itemset))


def _ldp_estimates(
    dataset,
    clients: list[VerticalClient],
    candidates: list[Candidate],
    epsilon: float,
    seed: int,
    mode: str,
    n_rows: int,
    noisy_singleton_counts: dict[tuple[int, int], float] | None = None,
    report_key_limit: int = 15,
) -> tuple[list[CandidateEstimate], int, int, list[float], dict[str, float]]:
    """对固定正向块逐键 OUE，并以对齐用户报告的去偏积估计跨方交集。"""
    local_policy = (
        "top_k"
        if mode == MIXED_ITEMSET_ALPHA
        else "all"
    )
    keys_by_client = {
        client.client_id: client.report_keys(
            candidates,
            mode,
            local_key_policy=local_policy,
            report_key_limit=report_key_limit if local_policy == "top_k" else None,
            noisy_singleton_counts=noisy_singleton_counts,
            normalization_n=n_rows,
        )
        for client in clients
    }
    client_started = time.perf_counter()
    reports: dict[tuple[str, tuple[tuple[int, int], ...]], np.ndarray] = {}
    epsilon_by_key: dict[tuple[str, tuple[tuple[int, int], ...]], float] = {}
    uplink_bits = 0
    for client in clients:
        keys = keys_by_client[client.client_id]
        if not keys:
            continue
        epsilon_key = epsilon / len(clients) / len(keys)
        for key in keys:
            report = oue_positive_membership(
                client._membership(key),
                epsilon_key,
                stable_seed(seed, "ldp-oue-round2", client.client_id, mode, key),
            )
            reports[(client.client_id, key)] = report
            epsilon_by_key[(client.client_id, key)] = epsilon_key
            uplink_bits += client.n_rows

    client_seconds = time.perf_counter() - client_started
    server_started = time.perf_counter()
    owner = dataset.owner_by_attribute
    server = PrivFimServer(n_rows, dataset.partitions)
    estimates = []
    for candidate in candidates:
        blocks = server._blocks_for_candidate(
            candidate.itemset,
            mode,
            available_report_keys=set(reports),
        )
        if blocks is None:
            continue
        block_reports = []
        block_epsilons = []
        for block in blocks:
            client_id = f"client_{owner[block[0][0]]}"
            block_reports.append(reports[(client_id, block)])
            block_epsilons.append(epsilon_by_key[(client_id, block)])
        estimated = oue_intersection_estimate(block_reports, block_epsilons, n_rows)
        estimates.append(
            CandidateEstimate(
                itemset=candidate.itemset,
                estimated_count=estimated,
                guessed_count=candidate.guessed_count,
                local_blocks=blocks,
            )
        )
    estimates.sort(key=lambda estimate: (-estimate.estimated_count, estimate.itemset))
    return (
        estimates,
        sum(len(keys) for keys in keys_by_client.values()),
        uplink_bits,
        list(epsilon_by_key.values()),
        {
            "client": client_seconds,
            "server": time.perf_counter() - server_started,
        },
    )


def _base_row(case: ExperimentCase, dataset, run_id: str, scheme: str, estimator: str, mode: str, candidate_count: int, direct_count: int, report_keys: int, epsilon_per_key: float, round1_mib: float, downlink_mib: float, round2_mib: float, timing: dict[str, float], metrics: dict[str, float], accounting_model: str, joint_status: str) -> dict:
    row = {
        "run_id": run_id,
        "implementation_version": IMPLEMENTATION_VERSION,
        "axes": "+".join(case.axes),
        "dataset": case.dataset.name,
        "dataset_kind": case.dataset.kind,
        "correlation": case.dataset.correlation,
        "layout": case.dataset.layout,
        "domain_size": case.dataset.domain_size,
        "zipf_exponent": case.dataset.zipf_exponent,
        "seed": case.seed,
        "rows": dataset.n_rows,
        "attributes": dataset.n_attributes,
        "domain_sizes": ":".join(str(len(domain)) for domain in dataset.domains),
        "max_rows": "",
        "sample_ratio": case.sample_ratio,
        "feature_ratio": case.feature_ratio,
        "domain_ratio": case.domain_ratio,
        "k": case.k,
        "epsilon": case.epsilon,
        "num_clients": len(dataset.partitions),
        "attribute_ratio": ":".join("1" for _ in dataset.partitions),
        "attribute_counts": ":".join(str(len(partition)) for partition in dataset.partitions),
        "m": 0,
        "candidate_multiplier": case.candidate_multiplier,
        "first_stage_report_limit": (
            case.k
            if case.first_stage_report_limit is None
            else case.first_stage_report_limit
        ),
        "first_stage_multiplier": case.first_stage_multiplier,
        "second_stage_multiplier": case.second_stage_multiplier,
        "second_stage_upload_limit": (
            "" if case.second_stage_multiplier is None
            else max(1, round(case.k * case.second_stage_multiplier))
        ),
        "phase1_ratio": case.phase1_ratio,
        "estimator": estimator,
        "mode": mode,
        "scheme": scheme,
        "candidate_count": candidate_count,
        "direct_candidate_count": direct_count,
        "report_keys": report_keys,
        "total_public_overlap_bound": 0,
        "max_public_overlap_bound": 0,
        "accounting_model": accounting_model,
        "joint_dp_status": joint_status,
        "coordinate_epsilon_mean": math.nan,
        "coordinate_epsilon_max": math.nan,
        "overlap_rdp_weighted_overlap": math.nan,
        "overlap_rdp_coefficient": math.nan,
        "overlap_rdp_order": math.nan,
        "overlap_rdp_epsilon": math.nan,
        "noisy_n_epsilon_target": case.epsilon * case.phase1_ratio * 0.1,
        "phase1_epsilon_target": case.epsilon * case.phase1_ratio * 0.9,
        "phase2_epsilon_target": case.epsilon * (1.0 - case.phase1_ratio),
        "overlap_rdp_budget_slack": math.nan,
        "overlap_rdp_epsilon_ratio": math.nan,
        "overlap_rdp_within_phase2_budget": 1.0,
        "certified_total_epsilon": case.epsilon,
        "total_epsilon_budget_slack": 0.0,
        "certified_total_epsilon_ratio": 1.0,
        "theoretical_total_within_budget": 1.0,
        "alpha_values": 0,
        "payload_bytes": round2_mib * 2**20,
        "payload_mib": round2_mib,
        "epsilon_per_key_mean": epsilon_per_key,
        "epsilon_per_key_min": epsilon_per_key,
        "round1_uplink_mib": round1_mib,
        "candidate_downlink_mib": downlink_mib,
        "round2_uplink_mib": round2_mib,
        "wire_total_mib": round1_mib + downlink_mib + round2_mib,
        "client_seconds": timing["client"],
        "server_seconds": timing["server"],
        "mode_seconds": timing["total"],
        "total_seconds": timing["total"],
    }
    row.update(metrics)
    return row


def _run_case(case: ExperimentCase) -> dict:
    started = time.perf_counter()
    dataset = load_vertical_csv(
        case.dataset.csv_path,
        num_clients=case.num_clients,
        sample_ratio=case.sample_ratio,
        feature_ratio=case.feature_ratio,
        domain_ratio=case.domain_ratio,
        transform_seed=case.seed,
        row_sampling_seed=case.seed,
    )
    clients = _clients(dataset)
    phase1_total_epsilon = case.epsilon * case.phase1_ratio
    noisy_n_epsilon = phase1_total_epsilon * 0.1
    phase1_epsilon = phase1_total_epsilon - noisy_n_epsilon
    phase2_epsilon = case.epsilon * (1.0 - case.phase1_ratio)
    n_rng = np.random.default_rng(stable_seed(case.seed, "fo-noisy-n"))
    noisy_n_reports = dataset.n_rows + n_rng.laplace(
        0.0,
        len(clients) / noisy_n_epsilon,
        size=len(clients),
    )
    noisy_n = max(1.0, float(np.mean(noisy_n_reports)))
    ldp_round1_started = time.perf_counter()
    ldp_counts, ldp_round1_bits = _ldp_round1(
        clients,
        epsilon=phase1_epsilon,
        seed=case.seed,
        n_rows=noisy_n,
        report_limit=(
            case.k
            if case.first_stage_report_limit is None
            else case.first_stage_report_limit
        ),
    )
    ldp_round1_bits += len(clients) * 64
    ldp_candidates, ldp_direct = _candidate_sets(
        ldp_counts,
        noisy_n,
        case.k,
        case.candidate_multiplier,
        case.first_stage_k,
    )
    ldp_round1_seconds = time.perf_counter() - ldp_round1_started
    ldp_supports = exact_candidate_supports(dataset, ldp_direct)
    downlink_mib = len(ldp_candidates) * len(clients) * 4 * 8 / 2**20

    estimates, key_count, round2_bits, epsilons, round2_timing = _ldp_estimates(
        dataset,
        clients,
        ldp_candidates,
        epsilon=phase2_epsilon,
        seed=case.seed,
        mode=MIXED_ITEMSET_ALPHA,
        n_rows=noisy_n,
        noisy_singleton_counts=ldp_counts,
        report_key_limit=(
            case.k if case.second_stage_multiplier is None
            else max(1, round(case.k * case.second_stage_multiplier))
        ),
    )
    metrics = evaluate_estimates(
        estimates,
        ldp_supports,
        case.k,
        dataset.n_rows,
        {candidate.itemset for candidate in ldp_candidates},
    ).to_dict()
    candidate_frequency_mib = len(ldp_candidates) * len(clients) * 8 / 2**20
    timing = {
        "client": ldp_round1_seconds + round2_timing["client"],
        "server": round2_timing["server"],
        "total": ldp_round1_seconds
        + round2_timing["client"]
        + round2_timing["server"],
    }
    rows = [
        _base_row(
            case,
            dataset,
            "",
            "FO-M",
            "ldp_oue",
            MIXED_ITEMSET_ALPHA,
            len(ldp_candidates),
            len(ldp_direct),
            key_count,
            float(np.mean(epsilons)) if epsilons else 0.0,
            ldp_round1_bits / 8 / 2**20,
            downlink_mib + candidate_frequency_mib,
            round2_bits / 8 / 2**20,
            timing,
            metrics,
            "local_dp_oue_basic_composition",
            "formal_pure_ldp",
        )
    ]
    return {
        "dataset": {
            "rows": dataset.n_rows,
            "noisy_rows": noisy_n,
            "attributes": dataset.n_attributes,
        },
        "created_unix": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def _build_sweep_cases(
    specs: dict[str, DatasetSpec],
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    axes: tuple[str, ...],
    ks: tuple[int, ...],
    epsilons: tuple[float, ...],
    client_counts: tuple[int, ...],
    sample_ratios: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    feature_ratios: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    domain_ratios: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    candidate_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 5.0),
    first_stage_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
    second_stage_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
    phase1_ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> list[ExperimentCase]:
    cases: dict[tuple, ExperimentCase] = {}

    def add(axis: str, **changes: object) -> None:
        for dataset_name in datasets:
            for seed in seeds:
                case = ExperimentCase(
                    dataset=specs[dataset_name], seed=seed, axes=(axis,), **changes
                )
                key = case.identity()
                if key in cases:
                    cases[key] = replace(
                        cases[key], axes=tuple(sorted(set(cases[key].axes) | {axis}))
                    )
                else:
                    cases[key] = case

    if "epsilon" in axes:
        for epsilon in epsilons:
            add("epsilon", epsilon=epsilon)
    if "k" in axes:
        for k in ks:
            add("k", k=k)
    if "clients" in axes:
        for client_count in client_counts:
            add("clients", num_clients=client_count)
    if "sample_ratio" in axes:
        for ratio in sample_ratios:
            add("sample_ratio", sample_ratio=ratio)
    if "feature_ratio" in axes:
        for ratio in feature_ratios:
            add("feature_ratio", feature_ratio=ratio)
    if "domain_ratio" in axes:
        for ratio in domain_ratios:
            add("domain_ratio", domain_ratio=ratio)
    if "candidate_pool" in axes:
        for multiplier in candidate_multipliers:
            add("candidate_pool", candidate_multiplier=multiplier)
    if "first_stage_pool" in axes:
        for multiplier in first_stage_multipliers:
            add(
                "first_stage_pool",
                first_stage_multiplier=multiplier,
                first_stage_report_limit=max(1, round(15 * multiplier)),
            )
    if "second_stage_pool" in axes:
        for multiplier in second_stage_multipliers:
            add("second_stage_pool", second_stage_multiplier=multiplier)
    if "budget_split" in axes:
        for ratio in phase1_ratios:
            add("budget_split", phase1_ratio=ratio)
    return sorted(cases.values(), key=lambda case: str(case.identity()))


def main() -> None:
    parser = argparse.ArgumentParser(description="运行参数化 FO-M（经典 OUE）基线")
    parser.add_argument("--output-dir", default="results/privacy_model_baselines")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    seed_group.add_argument("--seed-count", type=int, help="从 2026 起的连续种子数；例如 30")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument(
        "--axes", nargs="+",
        choices=("epsilon", "k", "clients", "sample_ratio", "feature_ratio",
                 "domain_ratio", "candidate_pool", "first_stage_pool",
                 "second_stage_pool", "budget_split"),
        default=("epsilon", "k", "clients"),
    )
    parser.add_argument("--epsilons", nargs="+", type=float, default=(0.25, 0.5, 1.0, 2.0, 4.0))
    parser.add_argument("--ks", nargs="+", type=int, default=(5, 10, 15, 20, 25))
    parser.add_argument("--clients", nargs="+", type=int, default=(2, 4, 8))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = tuple(range(2026, 2026 + args.seed_count)) if args.seed_count else tuple(args.seeds)
    if not seeds:
        raise ValueError("至少需要一个种子")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    specs = _dataset_specs()
    cases = _build_sweep_cases(
        specs,
        tuple(args.datasets),
        seeds,
        tuple(args.axes),
        tuple(args.ks),
        tuple(args.epsilons),
        tuple(args.clients),
    )
    rows: list[dict] = []
    failures = []
    for index, case in enumerate(cases, start=1):
        identity = json.dumps(case.identity(), default=str, separators=(",", ":"))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        run_id = f"fo_m_{case.dataset.name}_s{case.seed}_{digest}"
        summary_path = raw_dir / run_id / "summary.json"
        try:
            if summary_path.exists() and not args.force:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                logging.info("[%d/%d] reuse %s", index, len(cases), run_id)
            else:
                logging.info("[%d/%d] run %s", index, len(cases), run_id)
                summary = _run_case(case)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            for row in summary["rows"]:
                rows.append({**row, "run_id": run_id})
        except Exception as exc:
            logging.exception("failed: %s", run_id)
            failures.append({"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"})
        _write_csv(output_dir / "runs.csv", rows)
        _write_csv(output_dir / "aggregated.csv", aggregate_rows(rows))
        (output_dir / "status.json").write_text(json.dumps({"planned_cases": len(cases), "completed_cases": len({row["run_id"] for row in rows}), "failed_cases": len(failures), "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"隐私模型基线完成: {len({row['run_id'] for row in rows})}/{len(cases)}")


if __name__ == "__main__":
    main()
