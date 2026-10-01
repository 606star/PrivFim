from __future__ import annotations

import csv
import json
import math
import logging
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .binning import (
    BinningPlan,
    private_quantile_bins,
    protected_private_quantile_bins,
)
from .client import VerticalClient
from .communication import communication_summary
from .config import ExperimentConfig
from .data import VerticalDataset, load_vertical_csv
from .dpfm import RankOracle
from .metrics import (
    errors_by_support_band,
    evaluate_estimates,
    exact_candidate_supports,
    metrics_by_itemset_size,
)
from .privacy import (
    JOINT_DP_STATUS,
    NOMINAL_ACCOUNTING_MODEL,
    measurement_groups,
    overlap_rdp_account,
    public_measurement_group_overlap_bound,
    public_overlap_bound,
)
from .server import PrivFimServer
from .types import (
    ESTIMATOR_LABELS,
    FM_FULL_ESTIMATOR,
    FM_INVERSE_ESTIMATOR,
    FM_OTHER_ESTIMATOR,
    MAP_BOUNDS_ONLY_ESTIMATOR,
    MAP_NO_FRECHET_ESTIMATOR,
    PROTOCOL_IMPLEMENTATION_VERSION,
    REPORT_MODE_LABELS,
    CandidateEstimate,
    itemset_text,
)
from .types import (
    DIRECT_UNION_COMPLEMENT,
    LOCAL_ITEMSET_ALPHA,
    LOCAL_TOP_ITEMSET_ALPHA,
    MIXED_BINNED_ITEMSET_ALPHA,
    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
    MIXED_COVER_ITEMSET_ALPHA,
    MIXED_DUMMY_ITEMSET_ALPHA,
    MIXED_ITEMSET_ALPHA,
    SINGLETON_ALPHA,
    LOCAL_TOP_SINGLETON_ALPHA,
    LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
)


LOGGER = logging.getLogger("privfim")

BINNED_MIXED_MODES = {
    MIXED_BINNED_ITEMSET_ALPHA,
    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
}


def _build_clients(dataset: VerticalDataset) -> list[VerticalClient]:
    return [
        VerticalClient(
            client_id=f"client_{owner}",
            attributes=attributes,
            local_data=dataset.data[:, attributes],
            domains=dataset.domains,
        )
        for owner, attributes in enumerate(dataset.partitions)
    ]


def _write_mode_details(
    path: Path,
    estimates: list[CandidateEstimate],
    exact_supports: dict,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "itemset",
                "estimated_count",
                "true_count",
                "absolute_error",
                "guessed_count",
                "local_blocks",
            ],
        )
        writer.writeheader()
        for rank, estimate in enumerate(estimates, start=1):
            true_count = exact_supports[estimate.itemset]
            writer.writerow(
                {
                    "rank": rank,
                    "itemset": itemset_text(estimate.itemset),
                    "estimated_count": f"{estimate.estimated_count:.6f}",
                    "true_count": true_count,
                    "absolute_error": f"{abs(estimate.estimated_count - true_count):.6f}",
                    "guessed_count": f"{estimate.guessed_count:.6f}",
                    "local_blocks": " | ".join(itemset_text(block) for block in estimate.local_blocks),
                }
            )


def _average_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _candidate_diagnostics(candidates, direct_candidates, exact_supports, k: int) -> dict:
    guessed = [candidate.guessed_count for candidate in direct_candidates]
    truth = [exact_supports[candidate.itemset] for candidate in direct_candidates]
    if len(guessed) > 1:
        guessed_ranks = _average_ranks(guessed)
        truth_ranks = _average_ranks(truth)
        if np.std(guessed_ranks) == 0 or np.std(truth_ranks) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(guessed_ranks, truth_ranks)[0, 1])
    else:
        correlation = 0.0
    true_order = sorted(exact_supports, key=lambda key: (-exact_supports[key], key))
    true_top = true_order[:k]
    candidate_items = {candidate.itemset for candidate in candidates}
    by_size = {}
    for size in sorted({len(itemset) for itemset in true_top}):
        selected = [itemset for itemset in true_top if len(itemset) == size]
        by_size[str(size)] = {
            "true_top_count": len(selected),
            "recalled": len(set(selected) & candidate_items),
            "recall": len(set(selected) & candidate_items) / len(selected),
        }
    return {
        "guess_true_spearman": correlation,
        "true_top_itemset_size_counts": {
            str(size): sum(len(itemset) == size for itemset in true_top)
            for size in sorted({len(itemset) for itemset in true_top})
        },
        "candidate_recall_by_true_top_size": by_size,
    }


def run_experiment(
    config: ExperimentConfig,
    oracle_cache: dict[tuple[int, int, float, int], RankOracle] | None = None,
) -> dict:
    config.validate()
    started = time.perf_counter()
    data_started = time.perf_counter()
    dataset = load_vertical_csv(
        csv_path=config.data.csv_path,
        num_clients=config.data.num_clients,
        max_rows=config.data.max_rows,
        attribute_ratios=config.data.attribute_ratios,
        partition_seed=config.data.partition_seed,
        partition_strategy=config.data.partition_strategy,
        row_sampling_seed=config.data.row_sampling_seed,
        sample_ratio=config.data.sample_ratio,
        feature_ratio=config.data.feature_ratio,
        domain_ratio=config.data.domain_ratio,
        transform_seed=config.data.transform_seed,
    )
    clients = _build_clients(dataset)
    p = config.protocol
    data_seconds = time.perf_counter() - data_started

    LOGGER.info(
        "加载数据完成: rows=%d, attrs=%d, clients=%d",
        dataset.n_rows,
        dataset.n_attributes,
        len(clients),
    )

    # The experiment plan defines the noisy N budget as a fraction of round 1,
    # rather than a fraction of the total budget.  This gives
    #   eps_N = r_N * r_1 * eps, eps_1 = (1-r_N) * r_1 * eps,
    #   eps_2 = (1-r_1) * eps,
    # and therefore composes exactly to eps.
    phase1_total_epsilon = p.epsilon * p.phase1_ratio
    noisy_n_epsilon = phase1_total_epsilon * p.noisy_n_ratio
    # Vertical releases compose across clients, so split each public budget across parties.
    phase1_epsilon = phase1_total_epsilon - noisy_n_epsilon
    phase1_client_epsilon = phase1_epsilon / len(clients)
    phase2_epsilon = p.epsilon * (1.0 - p.phase1_ratio)
    phase2_client_epsilon = phase2_epsilon / len(clients)
    noisy_n_client_epsilon = noisy_n_epsilon / len(clients)
    phase2_client_delta = p.delta / len(clients)

    round1_started = time.perf_counter()
    round1_reports = [
        client.round1_report(
            phase1_client_epsilon,
            p.seed,
            n_epsilon=noisy_n_client_epsilon,
            report_limit=p.first_stage_report_limit,
        )
        for client in clients
    ]
    raw_noisy_n_mean = PrivFimServer.aggregate_noisy_n(round1_reports)
    # Positive clipping is post-processing of the released mean; no second noisy
    # count is introduced, so the server's normalization is the mean itself.
    noisy_n = max(1.0, float(raw_noisy_n_mean))
    server = PrivFimServer(
        noisy_n, dataset.partitions, domains=dataset.domains
    )
    LOGGER.info(
        "noisy N 聚合完成: reports=%d, mean=%.3f, server_n=%.3f, epsilon=%.6f",
        len(round1_reports),
        raw_noisy_n_mean,
        noisy_n,
        noisy_n_epsilon,
    )
    noisy_counts = server.aggregate_round1(round1_reports)
    first_stage_k = p.k if p.first_stage_k is None else p.first_stage_k
    candidate_count_target = math.ceil(p.candidate_multiplier * p.k)
    candidate_count_limit = (
        candidate_count_target if p.candidate_pruning == "topk" else None
    )
    frequent_items, candidates, direct_candidates = server.build_candidates(
        noisy_counts=noisy_counts,
        frequent_singleton_count=first_stage_k,
        candidate_count=candidate_count_limit,
        first_stage_mode=p.first_stage_mode,
        min_itemset_size=p.min_itemset_size,
        max_itemset_size=p.max_itemset_size,
    )
    if p.candidate_pruning == "topk" and len(candidates) < candidate_count_target:
        raise RuntimeError(
            "候选项集数量不足，无法构造完整候选集合: "
            f"candidates={len(candidates)}, expected={candidate_count_target}"
        )
    # MAP-M-Bin/BinP boundaries use only DP first-round histograms, public domains,
    # and candidate priors. Fix them for round two and send them to relevant owners.
    item_scores = {item: max(float(count), 0.0) for item, count in frequent_items}
    map_m_binnings: dict[str, BinningPlan] = {}
    if p.map_m_bin_count is not None:
        if MIXED_BINNED_ITEMSET_ALPHA in p.modes:
            map_m_binnings[MIXED_BINNED_ITEMSET_ALPHA] = private_quantile_bins(
                dataset.domains, noisy_counts, p.map_m_bin_count
            )
        if MIXED_PROTECTED_BINNED_ITEMSET_ALPHA in p.modes:
            map_m_binnings[MIXED_PROTECTED_BINNED_ITEMSET_ALPHA] = (
                protected_private_quantile_bins(
                    dataset.domains,
                    noisy_counts,
                    p.map_m_bin_count,
                    item_scores,
                )
            )
    round1_seconds = time.perf_counter() - round1_started
    LOGGER.info(
        "第一轮完成: frequent_items=%d, candidate_count=%d, all_candidates=%d",
        len(frequent_items),
        len(candidates),
        len(direct_candidates),
    )

    # Evaluate all methods on MAP-S-All's shared candidate universe, counting screening misses.
    truth_started = time.perf_counter()
    exact_supports = exact_candidate_supports(dataset, direct_candidates)
    truth_seconds = time.perf_counter() - truth_started
    common_error_itemsets = {candidate.itemset for candidate in candidates}

    oracle_started = time.perf_counter()
    oracle_key = (dataset.n_rows, p.m, p.gamma, p.seed)
    oracle_cache_hit = oracle_cache is not None and oracle_key in oracle_cache
    if oracle_cache_hit:
        oracle = oracle_cache[oracle_key]
    else:
        oracle = RankOracle.build(
            n_rows=dataset.n_rows,
            m=p.m,
            gamma=p.gamma,
            seed=p.seed,
            block_size=p.hash_block_size,
        )
        if oracle_cache is not None:
            oracle_cache[oracle_key] = oracle
    oracle_seconds = time.perf_counter() - oracle_started

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "implementation_version": PROTOCOL_IMPLEMENTATION_VERSION,
        "config": asdict(config),
        "dataset": {
            "rows": dataset.n_rows,
            "noisy_rows": noisy_n,
            "attributes": dataset.n_attributes,
            "partitions": [list(partition) for partition in dataset.partitions],
            "transform": dataset.transform,
            "domain_sizes": [len(domain) for domain in dataset.domains],
        },
        "privacy": {
            "protocol_epsilon_target": p.epsilon,
            "protocol_delta_target": p.delta,
            "noisy_n_epsilon_target": noisy_n_epsilon,
            "noisy_n_client_epsilon": noisy_n_client_epsilon,
            "noisy_n_mean_raw": raw_noisy_n_mean,
            "noisy_n_mean": noisy_n,
            "noisy_n_reports": [
                {
                    "client_id": report.client_id,
                    "noisy_n": report.noisy_n,
                    "epsilon": report.n_epsilon,
                }
                for report in round1_reports
            ],
            "statistics_epsilon_target": phase1_epsilon + phase2_epsilon,
            "phase1_total_epsilon_target": phase1_total_epsilon,
            "phase1_epsilon_target": phase1_epsilon,
            "phase1_client_epsilon": phase1_client_epsilon,
            "phase2_epsilon_target": phase2_epsilon,
            "phase2_delta_target": p.delta,
            "phase2_client_epsilon": phase2_client_epsilon,
            "phase2_client_delta": phase2_client_delta,
            "accounting_model": NOMINAL_ACCOUNTING_MODEL,
            "joint_dp_status": JOINT_DP_STATUS,
            "formal_end_to_end_dp": False,
            "theoretical_phase2_dp_under_secret_prf": False,
            "all_modes_individually_within_total_budget_under_secret_prf": False,
            "simultaneous_mode_release_accounted": False,
            "round1_uses_public_complete_domain": all(
                client.domains is not None for client in clients
            ),
            "randomness_mode": "reproducible_experiment",
            "shared_hash_key_hidden_from_server": False,
            "evaluation_truth_is_protocol_output": False,
        },
        "frequent_items": [
            {"item": list(item), "noisy_count": count}
            for item, count in frequent_items
        ],
        "candidate_count": len(candidates),
        "candidate_count_target": candidate_count_target,
        "candidate_pruning": p.candidate_pruning,
        "first_stage_k": first_stage_k,
        "first_stage_report_limit": p.first_stage_report_limit,
        "first_stage_mode": p.first_stage_mode,
        "direct_candidate_count": len(direct_candidates),
        "candidate_diagnostics": _candidate_diagnostics(
            candidates, direct_candidates, exact_supports, p.k
        ),
        "timing": {
            "data_seconds": data_seconds,
            "round1_and_candidates_seconds": round1_seconds,
            "exact_support_seconds": truth_seconds,
            "oracle_seconds": oracle_seconds,
            "oracle_cache_hit": oracle_cache_hit,
        },
        "modes": {},
    }

    for mode in p.modes:
        mode_started = time.perf_counter()
        mixed_mode = mode in {
            MIXED_ITEMSET_ALPHA,
            MIXED_BINNED_ITEMSET_ALPHA,
            MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
            MIXED_COVER_ITEMSET_ALPHA,
            MIXED_DUMMY_ITEMSET_ALPHA,
        }
        upload_limit = (
            p.second_stage_upload_limit
            if mixed_mode and p.second_stage_upload_limit is not None
            else p.k
        )
        mode_binning = map_m_binnings.get(mode)
        binned_map_m = (
            mode in BINNED_MIXED_MODES
            and p.estimator == "map"
            and p.map_m_bin_count is not None
        )
        if binned_map_m and mode_binning is None:
            raise RuntimeError(f"{mode} 缺少第二轮分箱计划")
        # Without product guessing, MAP considers all legal itemsets induced by the
        # first-round item pool, not top-ck. Local top-k keys determine which are estimable.
        mode_candidates = (
            direct_candidates
            if mode in {
                DIRECT_UNION_COMPLEMENT,
                LOCAL_TOP_SINGLETON_ALPHA,
                LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
            }
            else candidates
        )
        client_started = time.perf_counter()
        reports = [
            report
            for client in clients
            for report in client.round2_reports(
                candidates=mode_candidates,
                mode=mode,
                epsilon=phase2_client_epsilon,
                delta=phase2_client_delta,
                oracle=oracle,
                seed=p.seed,
                local_key_policy=(
                    "top_k"
                    if mode in {
                        MIXED_ITEMSET_ALPHA,
                        MIXED_BINNED_ITEMSET_ALPHA,
                        MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                        MIXED_COVER_ITEMSET_ALPHA,
                        MIXED_DUMMY_ITEMSET_ALPHA,
                    }
                    else "capped"
                    if mode == LOCAL_TOP_ITEMSET_ALPHA
                    else "local_top"
                    if mode in {
                        LOCAL_TOP_SINGLETON_ALPHA,
                        LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
                    }
                    else "required"
                ),
                local_projection_limit=p.local_projection_limit,
                report_key_limit=upload_limit,
                estimator=p.estimator,
                # MAP-L-Top and MAP-M use their respective joint-key budget settings;
                # MAP-S and full MAP-L retain equal allocation for internal comparisons.
                local_joint_budget_weight=(
                    p.local_joint_budget_weight
                    if mode == LOCAL_TOP_ITEMSET_ALPHA
                    else p.mixed_joint_budget_weight
                    if mode in {
                        MIXED_ITEMSET_ALPHA,
                        MIXED_BINNED_ITEMSET_ALPHA,
                        MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                        MIXED_COVER_ITEMSET_ALPHA,
                        MIXED_DUMMY_ITEMSET_ALPHA,
                    }
                    else 1.0
                ),
                map_m_bin_count=p.map_m_bin_count,
                map_m_binning=mode_binning,
                # Rank all keys using server-aggregated first-round DP item counts.
                noisy_singleton_counts=noisy_counts,
                normalization_n=float(noisy_n),
            )
        ]
        client_seconds = time.perf_counter() - client_started
        server_started = time.perf_counter()
        estimates = server.estimate_candidates(
            candidates=mode_candidates,
            reports=reports,
            mode=mode,
            gamma=p.gamma,
            map_step=p.map_step,
            max_map_points=p.max_map_points,
            estimator=p.estimator,
            map_m_bin_count=(
                p.map_m_bin_count
                if binned_map_m
                else None
            ),
            map_m_binning=(
                mode_binning if binned_map_m else None
            ),
            round1_noisy_counts=noisy_counts,
        )
        server_seconds = time.perf_counter() - server_started
        metrics = evaluate_estimates(
            estimates,
            exact_supports,
            p.k,
            n_rows=dataset.n_rows,
            error_itemsets=common_error_itemsets,
        )
        report_keys_by_client = {
            client.client_id: tuple(
                report.key
                for report in reports
                if report.client_id == client.client_id
            )
            for client in clients
        }
        report_counts = {
            client_id: len(keys)
            for client_id, keys in report_keys_by_client.items()
        }
        grouped_map_m = (
            mode
            in {
                MIXED_ITEMSET_ALPHA,
                MIXED_BINNED_ITEMSET_ALPHA,
                MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                MIXED_COVER_ITEMSET_ALPHA,
                MIXED_DUMMY_ITEMSET_ALPHA,
            }
            and p.estimator not in {FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR}
            and p.mixed_joint_budget_weight > 0
        )
        measurement_groups_by_client = {
            client_id: measurement_groups(keys)
            for client_id, keys in report_keys_by_client.items()
        }
        report_lookup = {
            (report.client_id, report.key): report
            for report in reports
        }
        measurement_group_budgets = (
            {
                client_id: [
                    {
                        "attributes": list(group),
                        "epsilon": report_lookup[(client_id, group_keys[0])].epsilon,
                        "delta": report_lookup[(client_id, group_keys[0])].delta,
                        "key_count": len(group_keys),
                    }
                    for group, group_keys in groups.items()
                ]
                for client_id, groups in measurement_groups_by_client.items()
            }
            if grouped_map_m
            else None
        )
        overlap_bounds = {
            client_id: (
                public_measurement_group_overlap_bound(keys)
                if grouped_map_m
                else public_overlap_bound(keys)
            )
            for client_id, keys in report_keys_by_client.items()
        }
        epsilon_per_key = {
            client.client_id: (
                max(
                    (
                        report.epsilon
                        for report in reports
                        if report.client_id == client.client_id
                    ),
                    default=0.0,
                )
            )
            for client, count in zip(clients, report_counts.values(), strict=True)
        }
        delta_per_key = {
            client.client_id: (
                max(
                    (
                        report.delta
                        for report in reports
                        if report.client_id == client.client_id
                    ),
                    default=0.0,
                )
            )
            for client, count in zip(clients, report_counts.values(), strict=True)
        }
        coordinate_epsilons = {}
        for client in clients:
            client_reports = [
                report for report in reports if report.client_id == client.client_id
            ]
            coordinate_epsilons[client.client_id] = (
                max(
                    (
                        float(report.coordinate_epsilon)
                        for report in client_reports
                        if report.coordinate_epsilon is not None
                    ),
                    default=0.0,
                )
            )
        rdp_account = overlap_rdp_account(
            repetitions=p.m,
            target_delta=p.delta,
            overlap_bounds=overlap_bounds,
            coordinate_epsilons=coordinate_epsilons,
        )
        rdp_within_budget = rdp_account.rdp_epsilon <= phase2_epsilon + 1e-12
        certified_total_epsilon = (
            noisy_n_epsilon + phase1_epsilon + rdp_account.rdp_epsilon
        )
        total_within_budget = certified_total_epsilon <= p.epsilon + 1e-12
        alpha_values = sum(report_counts.values()) * p.m
        communication = communication_summary(
            round1_reports=round1_reports,
            candidates=mode_candidates,
            round2_reports=reports,
            client_count=len(clients),
            include_candidate_guessed_count=(
                mode in {
                    MIXED_ITEMSET_ALPHA,
                    MIXED_BINNED_ITEMSET_ALPHA,
                    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                    MIXED_COVER_ITEMSET_ALPHA,
                    MIXED_DUMMY_ITEMSET_ALPHA,
                }
            ),
            noisy_singleton_counts=(
                noisy_counts
                if mode in {
                    MIXED_ITEMSET_ALPHA,
                    MIXED_BINNED_ITEMSET_ALPHA,
                    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                    MIXED_COVER_ITEMSET_ALPHA,
                    MIXED_DUMMY_ITEMSET_ALPHA,
                }
                else None
            ),
            binning=mode_binning if binned_map_m else None,
            binning_attributes_by_client=(
                {
                    client_id: tuple(
                        sorted(
                            {
                                attr
                                for key in keys
                                for attr, _ in key
                            }
                        )
                    )
                    for client_id, keys in report_keys_by_client.items()
                }
                if binned_map_m
                else None
            ),
        )
        elapsed = time.perf_counter() - mode_started
        scheme_label = REPORT_MODE_LABELS[mode]
        if p.estimator in {FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR}:
            scheme_label = ESTIMATOR_LABELS[p.estimator]
        elif p.estimator in {"fm", FM_INVERSE_ESTIMATOR}:
            scheme_label = f"{scheme_label}-FM-Inverse"
        elif p.estimator == MAP_NO_FRECHET_ESTIMATOR:
            scheme_label = f"{scheme_label}-NoLowerBound"
        elif p.estimator == MAP_BOUNDS_ONLY_ESTIMATOR:
            scheme_label = f"{scheme_label}-BoundsOnly"
        elif mode == LOCAL_TOP_ITEMSET_ALPHA:
            scheme_label = "MAP-L-Top"
        elif mode == LOCAL_TOP_SINGLETON_ALPHA:
            scheme_label = "MAP-NoGuess-Items"
        elif mode == LOCAL_TOP_ITEMSET_COMPONENT_ALPHA:
            scheme_label = "MAP-NoGuess-ItemsetComponents"
        elif mode == MIXED_BINNED_ITEMSET_ALPHA and binned_map_m:
            scheme_label = f"MAP-M-Bin{p.map_m_bin_count}"
        elif mode == MIXED_PROTECTED_BINNED_ITEMSET_ALPHA and binned_map_m:
            scheme_label = f"MAP-M-BinP{p.map_m_bin_count}"
        elif mode == MIXED_ITEMSET_ALPHA and p.mixed_joint_budget_weight == 0:
            scheme_label = "MAP-M-Unweighted"
        elif mode not in (
            DIRECT_UNION_COMPLEMENT,
            SINGLETON_ALPHA,
            LOCAL_ITEMSET_ALPHA,
            MIXED_ITEMSET_ALPHA,
            MIXED_BINNED_ITEMSET_ALPHA,
            MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
            MIXED_COVER_ITEMSET_ALPHA,
            MIXED_DUMMY_ITEMSET_ALPHA,
        ):
            if p.local_key_policy == "required":
                scheme_label = "MAP-L-Required"
            elif p.local_key_policy == "capped":
                scheme_label = f"MAP-L-B{p.local_projection_limit}"
        summary["modes"][mode] = {
            "scheme_label": scheme_label,
            "estimator": p.estimator,
            "local_key_policy": (
                "top_k"
                    if mode in {
                        MIXED_ITEMSET_ALPHA,
                        MIXED_BINNED_ITEMSET_ALPHA,
                        MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                        MIXED_COVER_ITEMSET_ALPHA,
                        MIXED_DUMMY_ITEMSET_ALPHA,
                    }
                else
                "capped" if mode == LOCAL_TOP_ITEMSET_ALPHA else
                "local_top" if mode in {
                    LOCAL_TOP_SINGLETON_ALPHA,
                    LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
                } else
                "required" if mode == LOCAL_ITEMSET_ALPHA else None
            ),
            "local_projection_limit": p.local_projection_limit,
            "local_joint_budget_weight": p.local_joint_budget_weight,
            # Positive values enable measurement-group budgets; zero uses equal per-key budgets.
            "mixed_joint_budget_weight": p.mixed_joint_budget_weight,
            "frequency_guessing": mode not in {
                LOCAL_TOP_SINGLETON_ALPHA,
                LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
            },
            "selection_privacy_status": (
                "local_key_identity_selection_not_dp"
                if mode in {
                    LOCAL_TOP_SINGLETON_ALPHA,
                    LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
                }
                else "postprocessing_of_round1_dp_transcript"
            ),
            "budget_allocation": (
                "binned_measurement_group_parallel"
                if binned_map_m and grouped_map_m
                else "measurement_group_parallel"
                if grouped_map_m
                else "per_key_equal"
                if mode
                in {
                    MIXED_ITEMSET_ALPHA,
                    MIXED_BINNED_ITEMSET_ALPHA,
                    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                    MIXED_COVER_ITEMSET_ALPHA,
                    MIXED_DUMMY_ITEMSET_ALPHA,
                }
                and p.mixed_joint_budget_weight == 0
                else "per_key_weighted"
            ),
            "report_binning": {
                "enabled": binned_map_m,
                "bin_count": p.map_m_bin_count if binned_map_m else None,
                "strategy": (
                    (
                        "first_round_dp_protected_value_quantile_tail"
                        if mode == MIXED_PROTECTED_BINNED_ITEMSET_ALPHA
                        else "first_round_dp_equal_mass_contiguous"
                    )
                    if binned_map_m
                    else None
                ),
                "recovery": (
                    (
                        "svsm_candidate_prior_within_bucket_normalization"
                        if mode == MIXED_PROTECTED_BINNED_ITEMSET_ALPHA
                        else "first_round_noisy_within_bin_product"
                    )
                    if binned_map_m
                    else None
                ),
                "dp_bins": (
                    {
                        str(attr): [list(bucket) for bucket in mode_binning[attr]]
                        for attr in range(len(dataset.domains))
                    }
                    if binned_map_m
                    else None
                ),
            },
            "measurement_groups": {
                client_id: [
                    {
                        "attributes": list(group),
                        "keys": [
                            [list(item) for item in key]
                            for key in group_keys
                        ],
                    }
                    for group, group_keys in groups.items()
                ]
                for client_id, groups in measurement_groups_by_client.items()
            },
            "measurement_group_count": {
                client_id: len(groups)
                for client_id, groups in measurement_groups_by_client.items()
            },
            "measurement_group_budgets": measurement_group_budgets,
            "report_key_limit": (
                upload_limit if mixed_mode and upload_limit != 0 else None
            ),
            "candidate_count": len(mode_candidates),
            "estimated_candidate_count": len(estimates),
            "skipped_candidate_count": len(mode_candidates) - len(estimates),
            "estimable_candidate_ratio": (
                len(estimates) / len(mode_candidates) if mode_candidates else 0.0
            ),
            "metrics": metrics.to_dict(),
            "report_key_count": report_counts,
            # Diagnose the upload-cap ablation using only public candidates and DP item counts.
            "available_report_key_count": (
                {
                    client.client_id: len(client.report_keys(
                        mode_candidates,
                        mode,
                        report_key_limit=0,
                        noisy_singleton_counts=noisy_counts,
                        normalization_n=noisy_n,
                    ))
                    for client in clients
                }
                if mode == MIXED_ITEMSET_ALPHA and p.estimator == "map"
                else None
            ),
            "report_keys": {
                client_id: [
                    [list(item) for item in key]
                    for key in keys
                ]
                for client_id, keys in report_keys_by_client.items()
            },
            "max_report_key_count": max(report_counts.values(), default=0),
            "report_key_limit_respected": (
                (upload_limit == 0 or max(report_counts.values(), default=0) <= upload_limit)
                if mixed_mode
                else None
            ),
            "public_overlap_bound": overlap_bounds,
            "total_public_overlap_bound": sum(overlap_bounds.values()),
            "max_public_overlap_bound": max(overlap_bounds.values(), default=0),
            "epsilon_per_key": epsilon_per_key,
            "delta_per_key": delta_per_key,
            "coordinate_epsilon": coordinate_epsilons,
            "overlap_rdp": rdp_account.to_dict(),
            "overlap_rdp_epsilon": rdp_account.rdp_epsilon,
            "overlap_rdp_budget_slack": phase2_epsilon - rdp_account.rdp_epsilon,
            "overlap_rdp_within_phase2_budget": rdp_within_budget,
            "certified_total_epsilon": certified_total_epsilon,
            "total_epsilon_budget_slack": p.epsilon - certified_total_epsilon,
            "theoretical_total_within_budget_under_secret_prf": total_within_budget,
            "per_key_budget_role": (
                "shared_measurement_group_budget_audited_by_overlap_rdp"
                if grouped_map_m
                else "allocation_audited_by_overlap_rdp"
            ),
            "total_report_keys": sum(report_counts.values()),
            "alpha_values": alpha_values,
            "estimated_alpha_payload_bytes": alpha_values * 8,
            "communication": communication,
            "metrics_by_itemset_size": metrics_by_itemset_size(
                estimates, exact_supports, p.k, dataset.n_rows
            ),
            "errors_by_support_band": errors_by_support_band(
                estimates, exact_supports, dataset.n_rows
            ),
            "elapsed_seconds": elapsed,
            "client_report_seconds": client_seconds,
            "server_estimate_seconds": server_seconds,
            "top_k": [
                {
                    "itemset": [list(item) for item in estimate.itemset],
                    "estimated_count": estimate.estimated_count,
                    "true_count": exact_supports[estimate.itemset],
                }
                for estimate in estimates[: p.k]
            ],
            # Export all estimates for independent global auditing, without feeding truth to the protocol.
            "estimates": [
                {"itemset": [list(item) for item in estimate.itemset],
                 "estimated_count": estimate.estimated_count,
                 "guessed_count": estimate.guessed_count,
                 "local_blocks": [[list(item) for item in block] for block in estimate.local_blocks]}
                for estimate in estimates
            ],
        }
        _write_mode_details(
            output_dir / f"candidate_details_{mode}.csv",
            estimates,
            exact_supports,
        )
        LOGGER.info(
            "%s (%s) 完成: keys=%d, estimable=%d/%d, F1=%.4f, NCR=%.4f, RMSE=%.2f, "
            "RDP-eps=%.4f, time=%.2fs",
            scheme_label,
            mode,
            sum(report_counts.values()),
            len(estimates),
            len(mode_candidates),
            metrics.f1,
            metrics.ncr,
            metrics.rmse,
            rdp_account.rdp_epsilon,
            elapsed,
        )

    summary["privacy"]["theoretical_phase2_dp_under_secret_prf"] = all(
        result["overlap_rdp_within_phase2_budget"]
        for result in summary["modes"].values()
    )
    summary["privacy"][
        "all_modes_individually_within_total_budget_under_secret_prf"
    ] = all(
        result["theoretical_total_within_budget_under_secret_prf"]
        for result in summary["modes"].values()
    )
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["resources"] = {
        "dataset_bytes": int(dataset.data.nbytes),
        "rank_oracle_bytes": int(oracle.ranks.nbytes),
        "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
