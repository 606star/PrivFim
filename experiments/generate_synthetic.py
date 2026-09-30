from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SyntheticSpec:
    name: str
    n_rows: int
    n_attributes: int
    domain_size: int
    group_size: int
    correlation: float
    zipf_exponent: float
    seed: int
    layout: str = "grouped"


DEFAULT_SPECS = (
    SyntheticSpec("independent_uniform", 50_000, 24, 8, 4, 0.0, 0.0, 3101),
    SyntheticSpec("weakcorr_uniform", 50_000, 24, 8, 4, 0.25, 0.0, 3101),
    SyntheticSpec("mediumcorr_uniform", 50_000, 24, 8, 4, 0.5, 0.0, 3101),
    SyntheticSpec("correlated_uniform", 50_000, 24, 8, 4, 0.75, 0.0, 3101),
    SyntheticSpec("highcorr_uniform", 50_000, 24, 8, 4, 0.9, 0.0, 3101),
    SyntheticSpec(
        "correlated_zipf_grouped", 50_000, 24, 8, 4, 0.75, 1.2, 3201, "grouped"
    ),
    SyntheticSpec(
        "correlated_zipf_interleaved",
        50_000,
        24,
        8,
        4,
        0.75,
        1.2,
        3201,
        "interleaved",
    ),
    SyntheticSpec(
        "highcorr_zipf_grouped", 50_000, 24, 8, 4, 0.9, 1.5, 3202, "grouped"
    ),
    # 固定相关性和值域，只改变边缘长尾强度，避免把长尾与相关性混为一谈。
    SyntheticSpec("tail_zipf_0", 50_000, 24, 8, 4, 0.75, 0.0, 3301),
    SyntheticSpec("tail_zipf_0_6", 50_000, 24, 8, 4, 0.75, 0.6, 3301),
    SyntheticSpec("tail_zipf_1_2", 50_000, 24, 8, 4, 0.75, 1.2, 3301),
    SyntheticSpec("tail_zipf_1_8", 50_000, 24, 8, 4, 0.75, 1.8, 3301),
    # 固定相关性和 Zipf 指数，只改变每个属性的分类值域大小。
    SyntheticSpec("domain_4", 50_000, 24, 4, 4, 0.75, 1.2, 3401),
    SyntheticSpec("domain_8", 50_000, 24, 8, 4, 0.75, 1.2, 3401),
    SyntheticSpec("domain_16", 50_000, 24, 16, 4, 0.75, 1.2, 3401),
    SyntheticSpec("domain_32", 50_000, 24, 32, 4, 0.75, 1.2, 3401),
)


def _categorical_probabilities(domain_size: int, exponent: float) -> np.ndarray:
    if exponent <= 0:
        return np.full(domain_size, 1.0 / domain_size)
    ranks = np.arange(1, domain_size + 1, dtype=np.float64)
    probabilities = np.power(ranks, -exponent)
    return probabilities / probabilities.sum()


def generate(spec: SyntheticSpec) -> np.ndarray:
    """生成保持目标边缘分布、具有可控组内相关性的分类表格。"""
    if spec.n_attributes % spec.group_size != 0:
        raise ValueError("n_attributes 必须能被 group_size 整除")
    if not 0 <= spec.correlation <= 1:
        raise ValueError("correlation 必须在 [0, 1] 内")
    if spec.layout not in {"grouped", "interleaved"}:
        raise ValueError("layout 必须是 grouped 或 interleaved")

    rng = np.random.default_rng(spec.seed)
    probabilities = _categorical_probabilities(
        spec.domain_size, spec.zipf_exponent
    )
    group_count = spec.n_attributes // spec.group_size
    latent = rng.choice(
        spec.domain_size,
        size=(spec.n_rows, group_count),
        p=probabilities,
    )

    grouped_data = np.empty((spec.n_rows, spec.n_attributes), dtype=np.int64)
    for attr in range(spec.n_attributes):
        group = attr // spec.group_size
        independent_values = rng.choice(
            spec.domain_size, size=spec.n_rows, p=probabilities
        )
        use_latent = rng.random(spec.n_rows) < spec.correlation
        grouped_data[:, attr] = np.where(
            use_latent, latent[:, group], independent_values
        )

    if spec.layout == "grouped":
        return grouped_data
    order = [
        group * spec.group_size + offset
        for offset in range(spec.group_size)
        for group in range(group_count)
    ]
    return grouped_data[:, order]


def group_labels(spec: SyntheticSpec) -> np.ndarray:
    """返回输出列对应的潜变量组，用于验证布局而不读取私有真值。"""
    group_count = spec.n_attributes // spec.group_size
    if spec.layout == "grouped":
        return np.repeat(np.arange(group_count), spec.group_size)
    return np.tile(np.arange(group_count), spec.group_size)


def _entropy(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _normalized_mutual_information(first: np.ndarray, second: np.ndarray) -> float:
    first_values, first_inverse = np.unique(first, return_inverse=True)
    second_values, second_inverse = np.unique(second, return_inverse=True)
    table = np.zeros((len(first_values), len(second_values)), dtype=np.float64)
    np.add.at(table, (first_inverse, second_inverse), 1.0)
    table /= table.sum()
    p_first = table.sum(axis=1, keepdims=True)
    p_second = table.sum(axis=0, keepdims=True)
    independent = p_first @ p_second
    valid = table > 0
    mutual_information = float(
        np.sum(table[valid] * np.log2(table[valid] / independent[valid]))
    )
    denominator = math.sqrt(max(_entropy(first) * _entropy(second), 1e-15))
    return mutual_information / denominator


def validate(data: np.ndarray, spec: SyntheticSpec) -> dict:
    labels = group_labels(spec)
    entropies = [_entropy(data[:, attr]) for attr in range(spec.n_attributes)]
    within_group = []
    across_group = []
    for first in range(spec.n_attributes):
        for second in range(first + 1, spec.n_attributes):
            value = _normalized_mutual_information(data[:, first], data[:, second])
            if labels[first] == labels[second]:
                within_group.append(value)
            else:
                across_group.append(value)
    top_supports = []
    for attr in range(spec.n_attributes):
        _, counts = np.unique(data[:, attr], return_counts=True)
        top_supports.append(float(counts.max() / len(data)))
    return {
        "shape": list(data.shape),
        "value_min": int(data.min()),
        "value_max": int(data.max()),
        "mean_attribute_entropy_bits": float(np.mean(entropies)),
        "mean_within_group_nmi": float(np.mean(within_group)),
        "mean_across_group_nmi": float(np.mean(across_group)),
        "mean_top_singleton_support": float(np.mean(top_supports)),
        "attribute_group_labels": labels.tolist(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_suite(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"generator": "latent_group_categorical_v2", "datasets": []}
    for spec in DEFAULT_SPECS:
        data = generate(spec)
        csv_path = output_dir / f"{spec.name}.csv"
        np.savetxt(
            csv_path,
            data,
            fmt="%d",
            delimiter=",",
            header=",".join(str(index) for index in range(spec.n_attributes)),
            comments="",
        )
        metadata = {
            "spec": asdict(spec),
            "validation": validate(data, spec),
            "generation_model": (
                "Attributes are divided into fixed-size groups. Each group has a "
                "row-level categorical latent value Z drawn from the target "
                "marginal p. Each attribute equals Z with probability correlation "
                "and is independently redrawn from p otherwise. This mixture "
                "preserves p exactly in expectation. Marginals use p(v) "
                "proportional to (v+1)^(-zipf_exponent). Grouped and interleaved "
                "layouts change only column placement, not generated records."
            ),
            "sha256": _sha256(csv_path),
        }
        metadata_path = output_dir / f"{spec.name}.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest["datasets"].append(
            {
                "name": spec.name,
                "csv": str(csv_path),
                "metadata": str(metadata_path),
                **metadata,
            }
        )
        print(
            f"{spec.name}: shape={data.shape}, "
            f"within_nmi={metadata['validation']['mean_within_group_nmi']:.4f}, "
            f"across_nmi={metadata['validation']['mean_across_group_nmi']:.4f}"
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成可控的分类频繁项集合成数据")
    parser.add_argument("--output-dir", default="data/synthetic")
    args = parser.parse_args()
    generate_suite(Path(args.output_dir))


if __name__ == "__main__":
    main()
