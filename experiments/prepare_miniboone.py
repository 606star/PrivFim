"""Prepare the official UCI MiniBooNE particle-identification data.

The UCI file contains two counts on its first line (signal and background),
followed by 50 real-valued measurements.  PrivFim operates on categorical
items, so each measurement is mapped to deterministic decile bins.  The
classification label is written separately and is not used as an item.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/miniboone/MiniBooNE_PID.txt"
OUT = ROOT / "data/real/MiniBooNE.csv"
TARGET = ROOT / "data/real/MiniBooNE_target.csv"
META = ROOT / "data/real/MiniBooNE.json"
N_FEATURES = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw() -> tuple[np.ndarray, np.ndarray, int, int]:
    if not RAW.is_file():
        raise FileNotFoundError(RAW)
    with RAW.open("r", encoding="ascii") as handle:
        first = handle.readline().split()
        if len(first) != 2:
            raise ValueError(f"首行应为 signal/background 两个计数，实际为 {first!r}")
        n_signal, n_background = (int(value) for value in first)
        n_rows = n_signal + n_background
        values = np.empty((n_rows, N_FEATURES), dtype=np.float64)
        for row_index in range(n_rows):
            line = handle.readline()
            if not line:
                raise ValueError(f"数据提前结束：期望 {n_rows} 行，已读 {row_index}")
            row = np.fromstring(line, sep=" ", dtype=np.float64)
            if row.size != N_FEATURES:
                raise ValueError(
                    f"第 {row_index + 2} 行应有 {N_FEATURES} 个特征，实际为 {row.size}"
                )
            values[row_index] = row
        trailing = [line for line in handle if line.strip()]
        if trailing:
            raise ValueError(f"文件包含计数之外的额外非空行：{len(trailing)}")
    labels = np.zeros(n_rows, dtype=np.int8)
    labels[:n_signal] = 1
    return values, labels, n_signal, n_background


def encode(values: np.ndarray) -> tuple[np.ndarray, list[int], list[list[str]]]:
    encoded = np.zeros(values.shape, dtype=np.int32)
    domain_sizes: list[int] = []
    maps: list[list[str]] = []
    # Quantile edges are fitted once on the complete released table.  Repeated
    # quantiles are removed so constant/low-cardinality columns remain valid.
    quantiles = np.linspace(0.0, 1.0, 11)
    for column in range(values.shape[1]):
        edges = np.unique(np.quantile(values[:, column], quantiles))
        if edges.size <= 1:
            column_values = np.zeros(values.shape[0], dtype=np.int32)
        else:
            column_values = np.searchsorted(
                edges[1:-1], values[:, column], side="right"
            ).astype(np.int32)
        encoded[:, column] = column_values
        size = int(column_values.max()) + 1 if column_values.size else 0
        domain_sizes.append(size)
        maps.append([f"bin_{index}" for index in range(size)])
    return encoded, domain_sizes, maps


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    values, labels, n_signal, n_background = read_raw()
    data, domain_sizes, maps = encode(values)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(range(N_FEATURES))
        writer.writerows(data.tolist())
    with TARGET.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"])
        writer.writerows([[int(label)] for label in labels])
    metadata = {
        "name": "MiniBooNE",
        "source": {
            "official_archive": "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt",
            "official_package": "https://archive.ics.uci.edu/static/public/199/miniboone+particle+identification.zip",
        },
        "rows": int(data.shape[0]),
        "attributes": N_FEATURES,
        "feature_names": [f"feature_{index}" for index in range(N_FEATURES)],
        "label": "particle_type",
        "label_values": {"background": 0, "signal": 1},
        "label_counts": {"background": int(n_background), "signal": int(n_signal)},
        "numeric_encoding": "10 quantile bins fitted on all rows",
        "domain_sizes": domain_sizes,
        "maps": maps,
        "raw_sha256": sha256(RAW),
        "csv_sha256": sha256(OUT),
        "target_sha256": sha256(TARGET),
    }
    META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("rows", "attributes", "domain_sizes", "label_counts", "csv_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
