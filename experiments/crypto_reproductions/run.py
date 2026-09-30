"""Run the upstream NIPP code and the VPP reproduction with independent truth.

Example: python -m experiments.crypto_reproductions.run --dataset toy --check-upstream
No legacy plaintext backend is reachable from this runner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from experiments.crypto_reproductions.vpp import run_vpp
from experiments.global_topk_search import exact_global_topk
from privfim.data import VerticalDataset, load_vertical_csv

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "build/crypto_reproductions/nipp_upstream"


def public_candidates(ds, max_size, limit):
    count = sum(sum(math_product(len(ds.domains[a]) for a in attrs)
                    for attrs in itertools.combinations(ds.attributes, length))
                for length in range(1, min(max_size, ds.n_attributes) + 1))
    if count > limit:
        raise ValueError(f"{count} public candidates exceed limit {limit}; no oracle truncation is allowed")
    keys = []
    for length in range(1, min(max_size, ds.n_attributes) + 1):
        for attrs in itertools.combinations(ds.attributes, length):
            for values in itertools.product(*(ds.domains[a] for a in attrs)):
                keys.append(tuple((int(a), int(v)) for a, v in zip(attrs, values)))
    return sorted(keys)


def math_product(values):
    result = 1
    for value in values:
        result *= value
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_nipp(ds, candidates, threshold, output, check=False, timeout=None,
             max_ciphertext_gib=4., selected_items_only=False, public_query=False):
    started = time.perf_counter()
    timeout = None if timeout is None or timeout <= 0 else timeout
    items = (sorted({item for key in candidates for item in key}) if selected_items_only else
             [(int(a), int(v)) for a in ds.attributes for v in ds.domains[a]])
    if not items:
        raise ValueError("at least one encrypted item is required")
    # 当前 TFHE 参数档每个 LWE 样本的实测序列化大小为 2536 bytes。
    # 这里只做保守的输入规模检查，实际进程还需要 bootstrapping key 和临时内存。
    width = ds.n_rows.bit_length()
    encrypted_queries = len(candidates) if not public_query or check else 0
    ciphertext_count = (ds.n_rows + encrypted_queries) * len(items) + len(candidates) * (width + 2) + width
    estimated_payload = ciphertext_count * 2536
    if estimated_payload > max_ciphertext_gib * 1024 ** 3:
        raise ValueError(f"TFHE ciphertext payload alone is approximately {estimated_payload / 1024**3:.2f} GiB, "
                         f"above the {max_ciphertext_gib:g} GiB input limit; use a smaller explicit run")
    item_owners = [ds.owner_by_attribute[a] for a, _ in items]
    matrix = np.column_stack([ds.data[:, a] == v for a, v in items]).astype(np.int8)
    queries = np.array([[int(item in key) for item in items] for key in candidates], dtype=np.int8)
    preparation = time.perf_counter() - started
    input_path = output / "nipp_input.txt"
    np.savetxt(input_path, np.concatenate((np.array(item_owners), matrix.ravel(), queries.ravel())),
               fmt="%d", comments="",
               header=f"{ds.n_rows} {len(items)} {len(ds.partitions)} {len(candidates)} {threshold}")
    command = [str(BINARY), str(input_path)] + (["--check-upstream"] if check else []) + (
        ["--public-query"] if public_query else [])
    with (output / "nipp_progress.log").open("w") as progress:
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=progress,
                                   text=True, timeout=timeout)
    result = json.loads(completed.stdout)
    write_json(output / "nipp_native.json", result)
    result.update(owner_encoding_seconds=preparation,
                  protocol_seconds=result["protocol_seconds"] + preparation,
                  candidate_source="public_full_universe", paper_alignment_verified=False,
                  workflow_implemented=True, timing_scope="protocol_end_to_end",
                  upstream_commit="d862f5c0d9dd4fa8cd155abdd8bbaa3cc6e63a81",
                  binary_sha256=digest(BINARY), input_sha256=digest(input_path),
                  estimated_ciphertext_payload_bytes=estimated_payload,
                  selected_items_only=selected_items_only,
                  public_query=public_query,
                  limits=["full paper unavailable, alignment checked against repository source only",
                          "support output and collaborative top-k ranking extend the original threshold API",
                          "owners share a TFHE key, not a threshold or multi-key FHE construction",
                          "aligned users assumed; PSI and network latency excluded"])
    supports = dict(zip(candidates, result["supports"], strict=True))
    if result["below_threshold"] != [int(supports[key] < threshold) for key in candidates]:
        raise AssertionError("native encrypted less-than mismatch")
    if result["above_threshold"] != [int(supports[key] > threshold) for key in candidates]:
        raise AssertionError("native encrypted greater-than mismatch")
    return supports, result


def evaluate(ds, supports, candidates, k, max_size):
    # 仅在协议返回以后接触原始数据计算真值，真值不进入上述任何云端接口。
    expected = {key: ds.support(key) for key in candidates}
    mismatches = [key for key in candidates if supports.get(key, 0) != expected[key]]
    if mismatches:
        raise AssertionError(f"encrypted support mismatch for {mismatches[:5]}")
    global_truth, metadata = exact_global_topk(ds, k, max_size=max_size)
    ranked = sorted(candidates, key=lambda key: (-supports.get(key, 0), key))[:k]
    truth = sorted(global_truth, key=lambda key: (-global_truth[key], key))[:k]
    overlap = len(set(truth) & set(ranked))
    precision = overlap / len(ranked) if ranked else 0.
    recall = overlap / len(truth) if truth else 0.
    weights = {key: len(truth) - i for i, key in enumerate(truth)}
    return {"f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "ncr": sum(weights.get(key, 0) for key in ranked) / sum(weights.values()),
            "support_mae": 0., "all_public_supports_verified": len(expected),
            "global_truth_verified": True, "global_truth_metadata": metadata,
            "topk": [{"itemset": key, "support": supports.get(key, 0)} for key in ranked]}, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="toy")
    parser.add_argument("--methods", nargs="+", choices=["NIPP-FIM", "VPP-OFIM"],
                        default=["NIPP-FIM", "VPP-OFIM"])
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--attributes", type=int, default=2)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=4)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--threshold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--paillier-bits", type=int, default=2048)
    parser.add_argument("--frequency-anonymity", type=int, default=2)
    parser.add_argument("--sample-ratio", type=float)
    parser.add_argument("--feature-ratio", type=float)
    parser.add_argument("--domain-ratio", type=float)
    parser.add_argument("--max-ciphertext-gib", type=float, default=4.)
    parser.add_argument("--candidate-limit", type=int, default=100_000)
    parser.add_argument("--timeout", type=int, default=0, help="0 means no time limit")
    parser.add_argument("--check-upstream", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.k < 1 or args.max_size < 1 or args.max_rows < 1:
        parser.error("k, max-size, max-rows must be positive")
    if args.dataset == "toy":
        ds = VerticalDataset(data=np.array([[0,0,0], [0,0,1], [0,1,1], [0,1,1]]),
                             attributes=(0,1,2), partitions=((0,), (1,2)),
                             domains=((0,1), (0,1), (0,1)))
        source_hash = hashlib.sha256(ds.data.tobytes()).hexdigest()
    else:
        source = ROOT / "data/real" / f"{args.dataset}.csv"
        full = load_vertical_csv(source, num_clients=args.num_clients,
                                 max_rows=args.max_rows, row_sampling_seed=args.seed,
                                 sample_ratio=args.sample_ratio, feature_ratio=args.feature_ratio,
                                 domain_ratio=args.domain_ratio, transform_seed=args.seed,
                                 partition_strategy="order")
        m = full.n_attributes if args.attributes == 0 else args.attributes
        if not args.num_clients <= m <= full.n_attributes:
            parser.error("selected attributes must be >= clients and <= original attributes")
        attrs = tuple(range(m))
        partitions = tuple(tuple(map(int, a)) for a in np.array_split(np.arange(m), args.num_clients))
        ds = VerticalDataset(data=full.data[:, :m].copy(), attributes=attrs,
                             partitions=partitions, domains=full.domains[:m])
        source_hash = digest(source)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    config = {**vars(args), "output": str(out), "actual_rows": ds.n_rows,
              "actual_attributes": ds.n_attributes, "partitions": ds.partitions,
              "domain_sizes": [len(d) for d in ds.domains], "data_sha256": source_hash,
              "evaluation_scope": "complete public itemset universe of the selected projection",
              "publication_eligible": False}
    write_json(out / "config.json", config)
    status = {"status": "preparing", "completed": [], "publication_eligible": False}

    def checkpoint(phase):
        status["status"] = phase
        write_json(out / "status.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    summaries = []
    try:
        checkpoint("preparing")
        phase = time.perf_counter()
        candidates = public_candidates(ds, args.max_size, args.candidate_limit)
        public_preparation = time.perf_counter() - phase
        for method in args.methods:
            checkpoint(f"running_{method}")
            if method == "NIPP-FIM":
                supports, transcript = run_nipp(ds, candidates, args.threshold, out,
                                                args.check_upstream, args.timeout, args.max_ciphertext_gib)
                transcript["protocol_seconds"] += public_preparation
            else:
                supports, transcript = run_vpp(ds, threshold=args.threshold, max_size=args.max_size,
                                                anonymity=args.frequency_anonymity,
                                                key_bits=args.paillier_bits,
                                                candidate_limit=args.candidate_limit)
                # Eclat 不出现的公开候选没有匹配 TID，其支持数必为 0。
                supports = {key: supports.get(key, 0) for key in candidates}
            phase = time.perf_counter()
            sorted(supports, key=lambda key: (-supports[key], key))[:args.k]
            transcript["protocol_seconds"] += time.perf_counter() - phase
            checkpoint(f"verifying_{method}")
            metrics, truth = evaluate(ds, supports, candidates, args.k, args.max_size)
            transcript.update(metrics=metrics, dataset=args.dataset, data_sha256=source_hash,
                              scope=config["evaluation_scope"])
            write_json(out / f"{method}.json", transcript)
            with (out / f"{method}_supports.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["itemset", "encrypted_result", "offline_truth"])
                writer.writerows((json.dumps(key), supports[key], truth[key]) for key in candidates)
            summaries.append({"method": method, "f1": metrics["f1"], "ncr": metrics["ncr"],
                              "seconds": transcript["protocol_seconds"],
                              "supports_verified": len(candidates), "publication_eligible": False})
            status["completed"].append(method)
            write_json(out / "summary.json", summaries)
        implementations = sorted(Path(__file__).parent.glob("*.py")) + sorted(Path(__file__).parent.glob("*.cpp"))
        write_json(out / "implementation_hashes.json", {str(p.relative_to(ROOT)): digest(p) for p in implementations})
        checkpoint("completed_and_verified")
        print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        status["error"] = repr(exc)
        checkpoint("failed")
        raise


if __name__ == "__main__":
    main()
