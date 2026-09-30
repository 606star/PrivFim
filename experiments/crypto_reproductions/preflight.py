"""Measure the size of the public-universe crypto workload before execution."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from experiments.crypto_reproductions.run import ROOT, math_product
from privfim.data import load_vertical_csv


def candidate_count(domain_sizes: list[int], max_size: int) -> int:
    return sum(math_product(domain_sizes[attr] for attr in attributes)
               for length in range(1, min(max_size, len(domain_sizes)) + 1)
               for attributes in itertools.combinations(range(len(domain_sizes)), length))


def inspect(dataset: str, max_size: int = 4) -> dict:
    source = ROOT / "data/real" / f"{dataset}.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = sum(1 for _ in csv.reader(handle)) - 1
    # The loader scans the complete CSV to obtain public domains even with one sampled row.
    ds = load_vertical_csv(source, num_clients=4, max_rows=1, partition_strategy="order")
    sizes = [len(domain) for domain in ds.domains]
    candidates = candidate_count(sizes, max_size)
    item_bits = rows * sum(sizes)
    return {"dataset": dataset, "rows": rows, "attributes": len(sizes),
            "public_items": sum(sizes), "domain_sizes": sizes,
            "public_candidates_1_to_max_size": candidates,
            "max_size": max_size,
            "tfhe_input_ciphertext_bytes_lower_bound": item_bits * 2536,
            "row_candidate_tests": rows * candidates,
            "direct_full_universe_feasible": candidates <= 100_000 and item_bits * 2536 <= 4 * 1024 ** 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["CensusIncomeKDD", "MiniBooNE", "DefaultCredit"])
    parser.add_argument("--max-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = {"candidate_boundary": "complete public itemsets of sizes 1 through max_size",
              "ciphertext_bytes_per_bit_measured": 2536,
              "note": "Input-only lower bound; excludes query, cloud-key, temporary ciphertexts, and network.",
              "datasets": [inspect(dataset, args.max_size) for dataset in args.datasets]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
