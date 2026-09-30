"""主方案、并补、FO 与单轮消融使用同一数据和输出接口。"""
from __future__ import annotations

import math
import time
from itertools import combinations, product

import numpy as np

from experiments.run_privacy_model_baselines import _clients, _ldp_round1, _ldp_estimates
from privfim.communication import round1_uplink_bytes, round2_uplink_bytes, candidate_message_bytes
from privfim.dpfm import RankOracle, private_cardinality_map_estimate, stable_seed
from privfim.privacy import overlap_rdp_account, public_measurement_group_overlap_bound
from privfim.server import PrivFimServer
from privfim.types import Candidate, CandidateEstimate, MIXED_ITEMSET_ALPHA, NoisyCountReport


CORE_METHODS = {
    "MAP-M": ("map", "mixed_itemset_alpha"),
    "MAP-S": ("map", "singleton_alpha"),
    "MAP-L": ("map", "local_itemset_alpha"),
    "IE-Full": ("fm_full", "singleton_alpha"),
    "IE-Other": ("fm_other", "singleton_alpha"),
}
METHODS = (*CORE_METHODS, "FO", "First-round", "Second-round", "Items-only")


def fo(dataset, p):
    clients = _clients(dataset)
    started = time.perf_counter()
    n_eps = p.epsilon * p.phase1_ratio * p.noisy_n_ratio
    rng = np.random.default_rng(stable_seed(p.seed, "fo-noisy-n"))
    noisy_n = max(1., float(np.mean(dataset.n_rows + rng.laplace(0., len(clients) / n_eps, len(clients)))))
    counts, bits1 = _ldp_round1(clients, p.epsilon * p.phase1_ratio - n_eps,
                               p.seed, noisy_n, p.first_stage_report_limit)
    server = PrivFimServer(noisy_n, dataset.partitions, dataset.domains)
    _, candidates, _ = server.build_candidates(counts, p.first_stage_k or p.k,
                                               math.ceil(p.candidate_multiplier * p.k),
                                               p.min_itemset_size, p.max_itemset_size)
    phase1_seconds = time.perf_counter() - started
    estimates, keys, bits2, epsilons, times = _ldp_estimates(
        dataset, clients, candidates, p.epsilon * (1 - p.phase1_ratio), p.seed,
        MIXED_ITEMSET_ALPHA, noisy_n, counts,
        p.k if p.second_stage_upload_limit is None else p.second_stage_upload_limit,
    )
    # OUE 位串 + noisy N + 广播的键、猜测频数与目标单项频数。
    download = len(clients) * (candidate_message_bytes(candidates, True) + 4 + 20 * len(counts))
    return estimates, {
        "round1_seconds": phase1_seconds, "rank_seconds": 0.,
        "client_seconds": times["client"], "server_seconds": times["server"],
        "total_seconds": phase1_seconds + times["client"] + times["server"],
        "communication_bytes": math.ceil(bits1 / 8) + 8 * len(clients) + math.ceil(bits2 / 8) + download,
        "communication_scope": "compact_payload_model", "report_keys": keys,
        "candidate_count": len(candidates), "noisy_n": noisy_n,
        "privacy_model": "OUE reports with composed local budgets; aligned users assumed",
    }


def one_round(dataset, p, method):
    clients = _clients(dataset)
    count = len(clients)
    reports = []
    start = time.perf_counter()
    oracle_seconds = 0.
    if method == "First-round":
        # 只有一轮，全部预算在本轮消耗，其中 10% 用于 noisy N。
        r1 = [c.round1_report(p.epsilon * .9 / count, p.seed,
                              n_epsilon=p.epsilon * .1 / count,
                              report_limit=p.first_stage_report_limit) for c in clients]
        server = PrivFimServer(max(1., PrivFimServer.aggregate_noisy_n(r1)), dataset.partitions, dataset.domains)
        counts = server.aggregate_round1(r1)
        _, candidates, _ = server.build_candidates(counts, p.first_stage_k or p.k,
            math.ceil(p.k * p.candidate_multiplier), p.min_itemset_size, p.max_itemset_size)
        estimates = sorted([CandidateEstimate(c.itemset, c.guessed_count, c.guessed_count, ()) for c in candidates],
                           key=lambda e: (-e.estimated_count, e.itemset))
        client_seconds = time.perf_counter() - start
        server_seconds = 0.
        privacy = {"model": "Laplace full-domain histogram, noisy top-P postprocessing"}
    else:
        # 本地键在接收任何服务器反馈前由公开值域确定。
        n_eps = p.epsilon * .05
        alpha_eps = p.epsilon - n_eps
        r1 = []
        for c in clients:
            rng = np.random.default_rng(stable_seed(p.seed, "one-shot-noisy-n", c.client_id))
            r1.append(NoisyCountReport(c.client_id, {}, 0., c.n_rows + rng.laplace(0., count / n_eps), n_eps / count))
        t = time.perf_counter()
        oracle = RankOracle.build(dataset.n_rows, p.m, p.gamma, p.seed, p.hash_block_size)
        oracle_seconds = time.perf_counter() - t
        for c in clients:
            keys = []
            for length in range(1, min(len(c.attributes), 1 if method == "Items-only" else 3) + 1):
                for attrs in combinations(c.attributes, length):
                    keys.extend(tuple(zip(attrs, values)) for values in product(*(dataset.domains[a] for a in attrs)))
            reports.extend(c.round2_reports([Candidate(key, 0., 0.) for key in sorted(keys)], MIXED_ITEMSET_ALPHA,
                alpha_eps / count, p.delta / count, oracle, p.seed, report_key_limit=0,
                local_joint_budget_weight=4., normalization_n=1.))
        client_seconds = time.perf_counter() - start - oracle_seconds
        t = time.perf_counter()
        server = PrivFimServer(max(1., PrivFimServer.aggregate_noisy_n(r1)), dataset.partitions, dataset.domains)
        counts = {r.key[0]: private_cardinality_map_estimate(r, server.n_rows, p.gamma, p.map_step, p.max_map_points)
                  for r in reports if len(r.key) == 1}
        _, candidates, _ = server.build_candidates(counts, p.first_stage_k or p.k,
            math.ceil(p.k * p.candidate_multiplier), p.min_itemset_size, p.max_itemset_size)
        needed = {(f"client_{i}", tuple(item for item in c.itemset if item[0] in attrs))
                  for c in candidates for i, attrs in enumerate(dataset.partitions)}
        needed |= {(f"client_{server.owner_by_attribute[item[0]]}", (item,)) for c in candidates for item in c.itemset}
        selected = [r for r in reports if (r.client_id, r.key) in needed]
        estimates = server.estimate_candidates(candidates, selected, MIXED_ITEMSET_ALPHA,
                                                p.gamma, p.map_step, p.max_map_points)
        server_seconds = time.perf_counter() - t
        bounds, coordinate_eps = {}, {}
        for c in clients:
            local = [r for r in reports if r.client_id == c.client_id]
            bounds[c.client_id] = public_measurement_group_overlap_bound(tuple(r.key for r in local))
            coordinate_eps[c.client_id] = max((r.coordinate_epsilon or 0. for r in local), default=0.)
        account = overlap_rdp_account(p.m, p.delta, bounds, coordinate_eps)
        privacy = {"alpha_rdp_epsilon": account.rdp_epsilon, "target_alpha_epsilon": alpha_eps,
                   "theoretical_budget_pass_under_secret_prf": account.rdp_epsilon <= alpha_eps + 1e-12,
                   "formal_end_to_end_dp": False}
    return estimates, {
        "round1_seconds": 0., "rank_seconds": oracle_seconds,
        "client_seconds": client_seconds, "server_seconds": server_seconds,
        "total_seconds": oracle_seconds + client_seconds + server_seconds,
        "communication_bytes": round1_uplink_bytes(r1)[0] + round2_uplink_bytes(reports)[0],
        "communication_scope": "compact_payload_model", "report_keys": len(reports),
        "candidate_count": len(candidates), "noisy_n": server.n_rows, "privacy": privacy,
    }
