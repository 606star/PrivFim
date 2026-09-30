"""协议完成后，用完整合法项集空间独立复核，不限于候选 S。"""
from __future__ import annotations

import time
from dataclasses import asdict

from experiments.global_topk_search import exact_global_topk
from privfim.metrics import evaluate_estimates
from privfim.types import CandidateEstimate
from workflows.common import write_csv, write_json


def restore_estimates(records):
    return [CandidateEstimate(tuple(tuple(item) for item in r["itemset"]),
                              r["estimated_count"], r.get("guessed_count", 0.),
                              tuple(tuple(tuple(item) for item in block) for block in r.get("local_blocks", [])))
            for r in records]


def audit(dataset, estimates, k, min_size, max_size, output, truth=None):
    started = time.perf_counter()
    top, details = truth or exact_global_topk(dataset, k, max_size, min_size=min_size)
    supports = dict(top)
    for estimate in estimates:
        supports[estimate.itemset] = dataset.support(estimate.itemset)
    metric = evaluate_estimates(estimates, supports, k, dataset.n_rows,
                               error_itemsets={e.itemset for e in estimates}).to_dict()
    metric["mse"] = metric["rmse"] ** 2
    metric["frequency_mse"] = metric["mse"] / dataset.n_rows ** 2
    metric["evaluation_scope"] = "global_exact"
    metric["evaluation_seconds"] = time.perf_counter() - started
    write_json(output / "global_truth.json", {
        "search": details, "top_k": [{"itemset": key, "count": count} for key, count in top.items()],
        "metrics": metric,
    })
    write_json(output / "estimates.json", [asdict(e) for e in estimates])
    write_csv(output / "estimates.csv", [
        {"rank": rank, "itemset": repr(e.itemset), "estimated_count": e.estimated_count,
         "true_count": supports[e.itemset], "guessed_count": e.guessed_count}
        for rank, e in enumerate(estimates, 1)
    ] or [{"rank": "", "itemset": "", "estimated_count": "", "true_count": "", "guessed_count": ""}])
    return metric
