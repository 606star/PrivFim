"""Command-line entry point for the portable release."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from workflows.common import ROOT
from workflows.methods import METHODS
from workflows.prepare import DATASETS, prepare
from workflows.runner import EXPERIMENTS, DEFAULT_METHODS, case_config, run_case, suite, summarize


def experiment_options(parser):
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2031)))
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument("--max-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="PrivFim: prepare → mine → global audit → CSV/PDF")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Download/encode datasets, or create the offline Toy dataset")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=["Toy"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p = sub.add_parser("run", help="Run methods at one configuration and audit against global truth")
    experiment_options(p)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=DATASETS)
    source.add_argument("--csv", type=Path, help="Integer CSV, header 0,1,...,M-1, aligned rows")
    p = sub.add_parser("configured", help="Run a complete JSON configuration, paths relative to its file")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    p = sub.add_parser("suite", help="Run any of the 11 experiment families")
    experiment_options(p)
    p.add_argument("--experiments", nargs="+", type=int, choices=EXPERIMENTS, default=list(EXPERIMENTS))
    p.add_argument("--datasets", nargs="+", choices=DATASETS)
    p.add_argument("--values", nargs="+", type=float, help="Override one experiment's x values")
    p = sub.add_parser("plot", help="Plot measured aggregated.csv")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("crypto", help="Evaluate real cryptographic backends separately for each dataset, without a timeout")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--methods", nargs="+", choices=("NIPP-FIM", "VPP-OFIM"), default=["NIPP-FIM", "VPP-OFIM"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--epsilon", type=float, default=1.)
    p.add_argument("--paillier-bits", type=int, default=2048)
    p.add_argument("--candidate-source", choices=("two-round", "public"), default="two-round")
    p.add_argument("--candidate-limit", type=int, default=100000, help="Public-enumeration complexity guard, not a timing window")
    p.add_argument("--max-rows", type=int, help="Explicit test sample only; omitted means all records")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ("seeds", "methods", "datasets", "experiments"):
        values = getattr(args, key, None)
        if values is not None and len(values) != len(set(values)):
            parser.error(f"--{key} must not contain duplicate values")
    if getattr(args, "max_rows", None) is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive; omit it for the full dataset")
    logging.basicConfig(level=logging.WARNING)
    if args.command == "prepare":
        for name in args.datasets:
            print(prepare(name, args.data_dir))
    elif args.command == "run":
        from workflows.plot import plot
        name = args.dataset or args.csv.stem
        source = args.csv or args.data_dir / "real" / f"{name}.csv"
        rows = []
        for seed in args.seeds:
            config = case_config(source, seed, args.k, args.epsilon, args.clients, args.m,
                                 max_rows=args.max_rows, min_size=args.min_size, max_size=args.max_size)
            for method in (args.methods or DEFAULT_METHODS):
                rows.append(run_case(config, method, args.output / name / f"seed_{seed}" / method, dataset_name=name))
                summarize(rows, args.output)
        plot(args.output / "aggregated.csv", args.output / "figures")
    elif args.command == "configured":
        from privfim.config import load_config
        from workflows.plot import plot
        config = load_config(args.config)
        output = Path(config.output_dir)
        name = Path(config.data.csv_path).stem
        rows = [run_case(config, method, output / name / method, dataset_name=name)
                for method in args.methods]
        summarize(rows, output)
        plot(output / "aggregated.csv", output / "figures")
    elif args.command == "suite":
        if args.values and len(args.experiments) != 1:
            parser.error("--values applies to exactly one experiment")
        suite(args)
    elif args.command == "plot":
        from workflows.plot import plot
        plot(args.input, args.output)
    elif args.command == "crypto":
        from workflows.crypto import run
        run(args)


if __name__ == "__main__":
    main()
