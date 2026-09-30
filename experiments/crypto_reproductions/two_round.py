"""PriVFim first round followed by encrypted support queries on its candidate set S.

This is a hybrid baseline. It does not reproduce either cryptographic paper's
independent itemset-discovery phase, and it is not eligible for original-paper
performance claims.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.crypto_reproductions.run import ROOT, digest, evaluate, run_nipp, write_json
from experiments.crypto_reproductions.vpp import run_vpp
from privfim.client import VerticalClient
from privfim.data import VerticalDataset, load_vertical_csv
from privfim.server import PrivFimServer


def first_round(ds: VerticalDataset, *, epsilon: float, k: int, seed: int,
                phase1_ratio: float = 0.5, noisy_n_ratio: float = 0.1,
                report_limit: int | None = None):
    if epsilon <= 0 or k < 1 or not 0 < phase1_ratio < 1 or not 0 < noisy_n_ratio < 1:
        raise ValueError("invalid first-round parameters")
    started = time.perf_counter()
    parties = len(ds.partitions)
    clients = [VerticalClient(f"client_{owner}", attrs, ds.data[:, attrs], ds.domains)
               for owner, attrs in enumerate(ds.partitions)]
    first_epsilon = epsilon * phase1_ratio
    item_epsilon = first_epsilon * (1 - noisy_n_ratio) / parties
    n_epsilon = first_epsilon * noisy_n_ratio / parties
    reports = [client.round1_report(item_epsilon, seed, n_epsilon=n_epsilon,
                                    report_limit=k if report_limit is None else report_limit)
               for client in clients]
    noisy_n = max(1., PrivFimServer.aggregate_noisy_n(reports))
    server = PrivFimServer(noisy_n, ds.partitions, domains=ds.domains)
    noisy_counts = server.aggregate_round1(reports)
    frequent, candidates, _ = server.build_candidates(
        noisy_counts, frequent_singleton_count=k, candidate_count=2 * k,
        min_itemset_size=1, max_itemset_size=4)
    if len(candidates) < min(2 * k, len(frequent)):
        raise RuntimeError("not enough first-round candidates")
    return [candidate.itemset for candidate in candidates], {
        "seconds": time.perf_counter() - started, "epsilon": epsilon,
        "phase1_ratio": phase1_ratio, "noisy_n_ratio": noisy_n_ratio,
        "noisy_n": noisy_n, "item_epsilon_per_client": item_epsilon,
        "n_epsilon_per_client": n_epsilon, "report_limit_per_client": k if report_limit is None else report_limit,
        "frequent_singletons": len(frequent), "candidate_count": len(candidates),
        "candidate_source": "private_round1_noisy_items_svsm_top_2k",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="toy")
    parser.add_argument("--methods", nargs="+", choices=("NIPP-FIM", "VPP-OFIM"),
                        default=("NIPP-FIM", "VPP-OFIM"))
    parser.add_argument("--max-rows", type=int, default=8,
                        help="0 means every row; positive values are explicit samples")
    parser.add_argument("--attributes", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--phase1-ratio", type=float, default=.5)
    parser.add_argument("--sample-ratio", type=float)
    parser.add_argument("--feature-ratio", type=float)
    parser.add_argument("--domain-ratio", type=float)
    parser.add_argument("--first-round-only", action="store_true")
    parser.add_argument("--first-round-input", type=Path,
                        help="reuse a verified first-round release for both crypto backends")
    parser.add_argument("--paillier-bits", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=0, help="0 means no time limit")
    parser.add_argument("--max-ciphertext-gib", type=float, default=4.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_rows < 0 or (args.max_rows and args.sample_ratio is not None):
        parser.error("max-rows must be nonnegative and cannot be combined with sample-ratio")
    if args.dataset == "toy":
        ds = VerticalDataset(data=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 1]]),
                             attributes=(0, 1, 2), partitions=((0,), (1, 2)),
                             domains=((0, 1), (0, 1), (0, 1)))
        data_hash = "toy"
    else:
        source = ROOT / "data/real" / f"{args.dataset}.csv"
        full = load_vertical_csv(source, num_clients=args.num_clients,
                                 max_rows=args.max_rows or None,
                                 sample_ratio=args.sample_ratio,
                                 feature_ratio=args.feature_ratio,
                                 domain_ratio=args.domain_ratio,
                                 transform_seed=args.seed,
                                 row_sampling_seed=args.seed, partition_strategy="order")
        count = full.n_attributes if args.attributes == 0 else args.attributes
        if not args.num_clients <= count <= full.n_attributes:
            parser.error("attributes must lie between num-clients and full width")
        partitions = tuple(tuple(int(attr) for attr in block)
                           for block in np.array_split(np.arange(count), args.num_clients))
        ds = VerticalDataset(full.data[:, :count].copy(), tuple(range(count)),
                             partitions, full.domains[:count])
        data_hash = digest(source)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    config = {**vars(args), "output": str(out),
              "first_round_input": None if args.first_round_input is None else str(args.first_round_input),
              "actual_rows": ds.n_rows,
              "actual_attributes": ds.n_attributes, "data_sha256": data_hash,
              "transformed_data_sha256": hashlib.sha256(ds.data.tobytes()).hexdigest(),
              "domain_sizes": [len(domain) for domain in ds.domains],
              "hybrid_protocol": True, "publication_eligible": False}
    write_json(out / "config.json", config)
    status = {"status": "running_first_round", "completed": []}
    write_json(out / "status.json", status)
    try:
        if args.first_round_input is None:
            candidates, stage1 = first_round(ds, epsilon=args.epsilon, k=args.k,
                                             seed=args.seed, phase1_ratio=args.phase1_ratio)
        else:
            released = args.first_round_input
            prior = json.loads((released / "config.json").read_text(encoding="utf-8"))
            prior_status = json.loads((released / "status.json").read_text(encoding="utf-8"))
            parameters = ("dataset", "seed", "epsilon", "k", "num_clients", "sample_ratio",
                          "feature_ratio", "domain_ratio", "actual_rows", "actual_attributes",
                          "transformed_data_sha256", "phase1_ratio")
            if prior_status.get("status") != "first_round_complete" or any(
                prior.get(key) != config.get(key) for key in parameters
            ):
                raise ValueError("first-round release does not match this dataset and configuration")
            candidates = [tuple((int(attribute), int(value)) for attribute, value in itemset)
                          for itemset in json.loads((released / "candidates.json").read_text(encoding="utf-8"))]
            stage1 = json.loads((released / "first_round.json").read_text(encoding="utf-8"))
            if len(candidates) != stage1["candidate_count"] or len(candidates) != len(set(candidates)):
                raise ValueError("first-round candidate release is incomplete")
        write_json(out / "first_round.json", stage1)
        write_json(out / "candidates.json", candidates)
        if args.first_round_only:
            status["status"] = "first_round_complete"
            write_json(out / "status.json", status)
            print(json.dumps({"candidate_count": len(candidates),
                              "first_round_seconds": stage1["seconds"],
                              "actual_rows": ds.n_rows, "actual_attributes": ds.n_attributes}), flush=True)
            return
        summary = []
        for method in args.methods:
            status["status"] = f"running_{method}"
            write_json(out / "status.json", status)
            if method == "NIPP-FIM":
                supports, protocol = run_nipp(ds, candidates, 0, out, timeout=args.timeout,
                                              max_ciphertext_gib=args.max_ciphertext_gib,
                                              selected_items_only=True, public_query=True)
            else:
                def report_progress(phase, completed, total):
                    write_json(out / "vpp_progress.json",
                               {"phase": phase, "completed": completed, "total": total})

                supports, protocol = run_vpp(ds, threshold=0, candidates=candidates,
                                             key_bits=args.paillier_bits,
                                             candidate_limit=max(100, 10 * len(candidates)),
                                             progress=report_progress)
                supports = {key: supports.get(key, 0) for key in candidates}
            metrics, truth = evaluate(ds, supports, candidates, args.k, 4)
            total = stage1["seconds"] + protocol["protocol_seconds"]
            transcript = {"first_round": stage1, "second_round": protocol,
                          "hybrid_label": "PriVFim+NIPP-TFHE" if method == "NIPP-FIM" else "PriVFim+VPP-Paillier",
                          "first_round_seconds": stage1["seconds"],
                          "second_round_seconds": protocol["protocol_seconds"],
                          "total_seconds": total, "metrics": metrics,
                          "candidate_source": stage1["candidate_source"],
                          "hybrid_protocol": True, "publication_eligible": False}
            write_json(out / f"{method}.json", transcript)
            with (out / f"{method}_supports.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("itemset", "encrypted_support", "offline_truth"))
                writer.writerows((json.dumps(key), supports[key], truth[key]) for key in candidates)
            summary.append({"method": method, "f1": metrics["f1"], "ncr": metrics["ncr"],
                            "hybrid_label": transcript["hybrid_label"],
                            "first_round_seconds": stage1["seconds"],
                            "second_round_seconds": protocol["protocol_seconds"],
                            "total_seconds": total, "candidate_count": len(candidates),
                            "publication_eligible": False})
            status["completed"].append(method)
            write_json(out / "summary.json", summary)
        status["status"] = "completed_and_verified"
        write_json(out / "status.json", status)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        write_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
