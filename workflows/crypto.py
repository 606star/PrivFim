"""按数据集保存真实密码学评估，不设置默认时间窗口或自动外推。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.crypto_reproductions.run import BINARY, public_candidates, run_nipp
from experiments.crypto_reproductions.two_round import first_round
from experiments.crypto_reproductions.vpp import run_vpp
from privfim.data import load_vertical_csv
from privfim.types import CandidateEstimate
from workflows.common import digest, implementation_digest, write_csv, write_json
from workflows.evaluate import audit


def run(args):
    root = Path(args.output).resolve()
    for name in args.datasets:
        source = Path(args.data_dir) / "real" / f"{name}.csv"
        directory = root / name / f"seed{args.seed}_k{args.k}_K{args.clients}_{args.candidate_source}"
        directory.mkdir(parents=True, exist_ok=True)
        ds = load_vertical_csv(source, args.clients, max_rows=args.max_rows,
                               row_sampling_seed=args.seed, partition_strategy="order")
        identity = {"dataset": name, "data_sha256": digest(source), "rows": ds.n_rows,
                    "attributes": ds.n_attributes, "k": args.k, "clients": args.clients,
                    "seed": args.seed, "epsilon": args.epsilon, "max_rows": args.max_rows,
                    "candidate_source": args.candidate_source, "paillier_bits": args.paillier_bits,
                    "implementation_sha256": implementation_digest(), "time_limit_seconds": None,
                    "result_kind": "measured", "max_size": 4,
                    "native_binary_sha256": digest(BINARY) if BINARY.exists() else None}
        config_file = directory / "config.json"
        if config_file.exists() and json.loads(config_file.read_text()) != identity:
            raise ValueError(f"Configuration changed, choose another output directory: {directory}")
        write_json(config_file, identity)
        stage_file = directory / "candidates.json"
        if stage_file.exists():
            first = json.loads(stage_file.read_text())
            candidates = [tuple(tuple(item) for item in key) for key in first["candidates"]]
            preparation = first["seconds"]
        else:
            if args.candidate_source == "two-round":
                candidates, info = first_round(ds, epsilon=args.epsilon, k=args.k, seed=args.seed)
                preparation = info["seconds"]
            else:
                started = time.perf_counter()
                candidates = public_candidates(ds, 4, args.candidate_limit)
                preparation = time.perf_counter() - started
                info = {"candidate_source": "public_full_universe"}
            write_json(stage_file, {"candidates": candidates, "seconds": preparation, "details": info})
        rows = []
        for method in args.methods:
            output = directory / method
            output.mkdir(exist_ok=True)
            result_path = output / "result.json"
            if result_path.exists():
                rows.append(json.loads(result_path.read_text())["row"])
                continue
            write_json(output / "status.json", {"status": "running", "time_limit_seconds": None})
            try:
                if method == "NIPP-FIM":
                    if not BINARY.exists():
                        raise FileNotFoundError("Run python scripts/setup_crypto.py before NIPP-FIM")
                    supports, transcript = run_nipp(ds, candidates, 0, output, timeout=None,
                        max_ciphertext_gib=float("inf"), selected_items_only=args.candidate_source == "two-round",
                        public_query=args.candidate_source == "two-round")
                else:
                    supports, transcript = run_vpp(ds, threshold=0, max_size=4, key_bits=args.paillier_bits,
                        candidates=candidates if args.candidate_source == "two-round" else None,
                        candidate_limit=args.candidate_limit)
                    supports = {key: supports.get(key, 0) for key in candidates}
                # 支持数必须逐项通过明文离线核验，不能预设 F1/NCR 为 1。
                if any(supports[key] != ds.support(key) for key in candidates):
                    raise AssertionError("Encrypted support disagrees with independent plaintext verification")
                ordered = sorted(supports, key=lambda key: (-supports[key], key))
                estimates = [CandidateEstimate(key, float(supports[key]), 0., ()) for key in ordered]
                metrics = audit(ds, estimates, args.k, 1, 4, output)
                row = {"dataset": name, "method": method, "k": args.k, "clients": args.clients,
                       "seed": args.seed, "rows": ds.n_rows, "attributes": ds.n_attributes,
                       "first_round_seconds": preparation, "second_round_seconds": transcript["protocol_seconds"],
                       "total_seconds": preparation + transcript["protocol_seconds"],
                       "candidate_count": len(candidates), "candidate_source": args.candidate_source,
                       "result_kind": "measured", **metrics}
                write_json(result_path, {"row": row, "transcript": transcript})
                write_json(output / "status.json", {"status": "complete_and_supports_verified"})
                rows.append(row)
                write_csv(directory / "comparison.csv", rows)
                print(f"{name} {method}: F1={row['f1']:.4f} NCR={row['ncr']:.4f} time={row['total_seconds']:.3f}s", flush=True)
            except BaseException as exc:
                write_json(output / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                                    "error": repr(exc)})
                raise
        write_csv(directory / "comparison.csv", rows)
