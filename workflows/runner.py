"""可恢复的实验闭环：参数→协议→全局真值→指标→汇总。"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, replace
from pathlib import Path

from privfim.config import DataConfig, ExperimentConfig, ProtocolConfig
from privfim.data import load_vertical_csv
from privfim.pipeline import run_experiment
from workflows.common import ROOT, digest, implementation_digest, write_csv, write_json
from workflows.evaluate import audit, restore_estimates
from workflows.methods import CORE_METHODS, fo, one_round


EXPERIMENTS = {
    1: ("epsilon", (.25, .5, 1, 2, 4)),
    2: ("k", (5, 10, 15, 20, 25)),
    3: ("clients", (2, 4, 8)),
    4: ("sample_ratio", (.2, .4, .6, .8, 1)),
    5: ("feature_ratio", (.2, .4, .6, .8, 1)),
    6: ("domain_ratio", (.2, .4, .6, .8, 1)),
    7: ("first_stage_pool", (.5, .75, 1, 1.5, 2)),
    8: ("second_stage_pool", (.5, .75, 1, 1.5, 2)),
    9: ("candidate_pool", (1, 1.5, 2, 2.5, 5)),
    10: ("budget_split", (.1, .2, .3, .4, .5, .6, .7, .8, .9)),
    11: ("mechanism", (1,)),
}
DEFAULT_DATASETS = {
    1: ("CensusIncomeKDD", "MiniBooNE"), 2: ("CensusIncomeKDD", "MiniBooNE"),
    3: ("CensusIncomeKDD", "MiniBooNE"), 4: ("Bank", "DefaultCredit"),
    5: ("Bank", "DefaultCredit"), 6: ("Bank", "DefaultCredit"),
    7: ("DefaultCredit", "MiniBooNE"), 8: ("Bank", "MiniBooNE"),
    9: ("CensusIncomeKDD", "PokerHand"), 10: ("PokerHand", "MiniBooNE"),
    11: ("Bank", "DefaultCredit"),
}
DEFAULT_METHODS = ("MAP-M", "IE-Full", "IE-Other", "FO")


def case_config(csv_path, seed=2026, k=15, epsilon=1., clients=4, m=2048,
                axis="default", value=1., max_rows=None, min_size=1, max_size=4):
    data = DataConfig(str(Path(csv_path).resolve()), num_clients=clients, max_rows=max_rows,
                      partition_strategy="order", row_sampling_seed=seed, transform_seed=seed)
    p = ProtocolConfig(k=k, epsilon=epsilon, m=m, seed=seed, min_itemset_size=min_size,
                       max_itemset_size=max_size, modes=("mixed_itemset_alpha",))
    if axis == "epsilon":
        p = replace(p, epsilon=float(value))
    elif axis == "k":
        p = replace(p, k=int(value))
    elif axis == "clients":
        data = replace(data, num_clients=int(value))
    elif axis in {"sample_ratio", "feature_ratio", "domain_ratio"}:
        if axis == "sample_ratio" and max_rows is not None:
            raise ValueError("Experiment 4 uses sample_ratio; remove --max-rows")
        data = replace(data, **{axis: float(value)})
    elif axis == "candidate_pool":
        p = replace(p, candidate_multiplier=float(value))
    elif axis == "budget_split":
        p = replace(p, phase1_ratio=float(value))
    p = replace(p, first_stage_report_limit=max(1, round(p.k * value)) if axis == "first_stage_pool" else p.k,
                second_stage_upload_limit=max(1, round(p.k * value)) if axis == "second_stage_pool" else p.k)
    return ExperimentConfig(data, p)


def run_case(config, method, output, *, dataset_name="custom", axis="default", value=1.):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config": asdict(config), "method": method, "dataset": dataset_name,
                "axis": axis, "value": value, "data_sha256": digest(config.data.csv_path),
                "implementation_sha256": implementation_digest()}
    identity["config"].pop("output_dir")
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    existing = output / "result.json"
    if existing.exists():
        prior = json.loads(existing.read_text())
        if prior["signature"] != signature:
            raise ValueError(f"Code/data/config changed: choose a new output directory: {output}")
        return prior["row"]
    write_json(output / "config.json", identity)
    write_json(output / "status.json", {"status": "running", "method": method})
    try:
        config.validate()
        p = config.protocol
        # 相同数据配置也用于离线真值，包括抽样、属性合并和值域缩减。
        dataset = load_vertical_csv(**asdict(config.data))
        if method in CORE_METHODS:
            estimator, mode = CORE_METHODS[method]
            actual = replace(config, protocol=replace(p, estimator=estimator, modes=(mode,)),
                             output_dir=str(output / "protocol"))
            summary = run_experiment(actual)
            result = summary["modes"][mode]
            estimates = restore_estimates(result["estimates"])
            timing = summary["timing"]
            info = {"round1_seconds": timing["round1_and_candidates_seconds"],
                    "rank_seconds": timing["oracle_seconds"],
                    "client_seconds": result["client_report_seconds"],
                    "server_seconds": result["server_estimate_seconds"],
                    "total_seconds": timing["round1_and_candidates_seconds"] + timing["oracle_seconds"]
                                     + result["client_report_seconds"] + result["server_estimate_seconds"],
                    "communication_bytes": result["communication"]["total_bytes"],
                    "communication_scope": "compact_payload_model", "report_keys": result["total_report_keys"],
                    "candidate_count": summary["candidate_count"], "noisy_n": summary["dataset"]["noisy_rows"],
                    "privacy": summary["privacy"],
                    "certified_total_epsilon": result["certified_total_epsilon"]}
        elif method == "FO":
            estimates, info = fo(dataset, p)
        else:
            estimates, info = one_round(dataset, p, method)
        metrics = audit(dataset, estimates, p.k, p.min_itemset_size, p.max_itemset_size, output)
        row = {"dataset": dataset_name, "axis": axis, "value": value, "seed": p.seed,
               "method": method, "k": p.k, "epsilon": p.epsilon, "clients": len(dataset.partitions),
               "m": p.m, "delta": p.delta, "rows": dataset.n_rows, "attributes": dataset.n_attributes,
               "items": sum(map(len, dataset.domains)), "min_size": p.min_itemset_size,
               "max_size": p.max_itemset_size, "result_kind": "measured",
               **metrics, **{k: v for k, v in info.items() if not isinstance(v, dict)}}
        write_json(existing, {"signature": signature, "row": row, "details": info})
        write_json(output / "status.json", {"status": "complete", "evaluation_scope": "global_exact"})
        print(f"{dataset_name} {axis}={value} seed={p.seed} {method}: "
              f"F1={row['f1']:.4f} NCR={row['ncr']:.4f} time={row['total_seconds']:.3f}s", flush=True)
        return row
    except BaseException as exc:
        write_json(output / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                            "error": repr(exc)})
        raise


def summarize(rows, output):
    write_csv(output / "runs.csv", rows)
    groups = {}
    for row in rows:
        key = (row["dataset"], row["axis"], row["value"], row["method"])
        groups.setdefault(key, []).append(row)
    aggregates = []
    for (dataset, axis, value, method), group in groups.items():
        seeds = sorted(r["seed"] for r in group)
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate seed in the same method/configuration")
        row = dict(dataset=dataset, axis=axis, value=value, method=method, seeds=len(group),
                   seed_list=":".join(map(str, seeds)),
                   evaluation_scope="global_exact", result_kind="measured")
        for metric in ("f1", "ncr", "total_seconds", "communication_bytes", "mse", "frequency_mse"):
            values = [r[metric] for r in group]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.
        aggregates.append(row)
    write_csv(output / "aggregated.csv", aggregates)
    return aggregates


def suite(args):
    from workflows.plot import plot
    root = Path(args.output).resolve()
    for number in args.experiments:
        axis, values = EXPERIMENTS[number]
        datasets = args.datasets or DEFAULT_DATASETS[number]
        methods = args.methods or (("First-round", "Second-round", "MAP-M") if number == 11 else DEFAULT_METHODS)
        output = root / f"experiment{number:02d}_{axis}"
        rows = []
        for name in datasets:
            for value in (args.values or values):
                for seed in args.seeds:
                    config = case_config(Path(args.data_dir) / "real" / f"{name}.csv", seed, args.k,
                                         args.epsilon, args.clients, args.m, axis, value, args.max_rows,
                                         args.min_size, args.max_size)
                    for method in methods:
                        row = run_case(config, method, output / name / f"value_{value:g}" / f"seed_{seed}" / method,
                                       dataset_name=name, axis=axis, value=value)
                        rows.append(row)
                        summarize(rows, output)
        plot(output / "aggregated.csv", output / "figures")
        write_json(output / "status.json", {"status": "complete", "runs": len(rows), "datasets": datasets})
