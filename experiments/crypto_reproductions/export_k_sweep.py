"""Export a bounded k sweep from one complete encrypted-support run.

The cryptographic query universe is independent of k. Runtime is one measured
run reused across ranks, not a new measurement at every plotted k.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


METHODS = ("NIPP-FIM", "VPP-OFIM")
FIELDS = ("dataset", "method", "seed", "k", "f1", "ncr", "total_seconds",
          "actual_rows", "actual_attributes", "num_clients", "candidate_count",
          "source_run", "source_sha256", "time_reused_from_run",
          "evaluation_scope", "publication_eligible")


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_supports(path: Path) -> tuple[dict[tuple, int], dict[tuple, int]]:
    measured, truth = {}, {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = tuple(tuple(pair) for pair in json.loads(row["itemset"]))
            if key in measured:
                raise ValueError(f"duplicate itemset in {path}: {key}")
            measured[key] = int(row["encrypted_result"])
            truth[key] = int(row["offline_truth"])
    if not measured or measured != truth:
        raise ValueError(f"incomplete or mismatched encrypted supports: {path}")
    return measured, truth


def rank_metrics(supports: dict[tuple, int], truth: dict[tuple, int], k: int) -> tuple[float, float]:
    if k < 1 or k > len(truth):
        raise ValueError(f"k={k} outside complete candidate universe of size {len(truth)}")
    ranked = sorted(supports, key=lambda key: (-supports[key], key))[:k]
    exact = sorted(truth, key=lambda key: (-truth[key], key))[:k]
    overlap = len(set(ranked) & set(exact))
    precision = overlap / len(ranked)
    recall = overlap / len(exact)
    weights = {key: k - i for i, key in enumerate(exact)}
    return 2 * precision * recall / (precision + recall), sum(weights.get(key, 0) for key in ranked) / sum(weights.values())


def export(source: Path, ks: tuple[int, ...]) -> list[dict]:
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed_and_verified":
        raise ValueError(f"source run is not complete: {source}")
    summary = {row["method"]: row for row in json.loads((source / "summary.json").read_text(encoding="utf-8"))}
    rows = []
    for method in METHODS:
        if method not in summary:
            continue
        supports_path = source / f"{method}_supports.csv"
        measured, truth = _read_supports(supports_path)
        if len(measured) != summary[method]["supports_verified"]:
            raise ValueError(f"support count disagrees with summary for {method}")
        transcript = json.loads((source / f"{method}.json").read_text(encoding="utf-8"))
        if not transcript.get("native_crypto_measured") or transcript.get("publication_eligible"):
            raise ValueError(f"unexpected crypto provenance for {method}")
        if abs(transcript["protocol_seconds"] - summary[method]["seconds"]) > 1e-6:
            raise ValueError(f"runtime disagrees with summary for {method}")
        for k in ks:
            f1, ncr = rank_metrics(measured, truth, k)
            rows.append(dict(dataset=config["dataset"], method=method, seed=config["seed"], k=k,
                             f1=f1, ncr=ncr, total_seconds=transcript["protocol_seconds"],
                             actual_rows=config["actual_rows"], actual_attributes=config["actual_attributes"],
                             num_clients=config["num_clients"], candidate_count=len(measured),
                             source_run=str(source), source_sha256=_digest(supports_path),
                             time_reused_from_run="true", evaluation_scope=config["evaluation_scope"],
                             publication_eligible="false"))
    if not rows:
        raise ValueError(f"no complete methods in {source}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ks", type=int, nargs="+", default=(5, 10, 15, 20, 25))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    rows = export(args.source, tuple(args.ks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} provisional rows to {args.output}")


if __name__ == "__main__":
    main()
