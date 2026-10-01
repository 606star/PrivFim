from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import Itemset


def _normalized_mutual_information(data: np.ndarray) -> np.ndarray:
    """Compute a public categorical-attribute affinity matrix for colocation experiments."""
    n_attributes = data.shape[1]
    scores = np.eye(n_attributes, dtype=np.float64)
    n_rows = max(len(data), 1)
    for left in range(n_attributes):
        _, left_inverse = np.unique(data[:, left], return_inverse=True)
        left_prob = np.bincount(left_inverse).astype(np.float64) / n_rows
        left_entropy = -float(
            np.sum(left_prob[left_prob > 0] * np.log(left_prob[left_prob > 0]))
        )
        for right in range(left + 1, n_attributes):
            _, right_inverse = np.unique(data[:, right], return_inverse=True)
            right_prob = np.bincount(right_inverse).astype(np.float64) / n_rows
            right_entropy = -float(
                np.sum(right_prob[right_prob > 0] * np.log(right_prob[right_prob > 0]))
            )
            if left_entropy == 0.0 or right_entropy == 0.0:
                continue
            joint = np.zeros(
                (len(left_prob), len(right_prob)), dtype=np.float64
            )
            np.add.at(joint, (left_inverse, right_inverse), 1.0)
            joint /= n_rows
            positive = joint > 0
            denominator = left_prob[:, None] * right_prob[None, :]
            mutual_information = float(
                np.sum(
                    joint[positive]
                    * np.log(joint[positive] / denominator[positive])
                )
            )
            score = np.clip(
                mutual_information / np.sqrt(left_entropy * right_entropy),
                0.0,
                1.0,
            )
            scores[left, right] = score
            scores[right, left] = score
    return scores


def _affinity_partitions(
    data: np.ndarray,
    counts: list[int],
    seed: int | None,
) -> tuple[tuple[int, ...], ...]:
    """Colocate related attributes while respecting each client's capacity."""
    affinity = _normalized_mutual_information(data)
    remaining = set(range(data.shape[1]))
    groups: list[list[int]] = [[] for _ in counts]

    # Separate clusters with farthest-point seeds, then add the most similar attributes.
    for owner, capacity in enumerate(counts):
        if capacity <= 0:
            continue
        if owner == 0:
            ranked = sorted(
                remaining,
                key=lambda attr: (-float(np.sum(affinity[attr])), attr),
            )
        else:
            previous = [attr for group in groups for attr in group]
            ranked = sorted(
                remaining,
                key=lambda attr: (
                    min(
                        (float(affinity[attr, other]) for other in previous),
                        default=0.0,
                    ),
                    -float(np.sum(affinity[attr])),
                    attr,
                ),
            )
        chosen = ranked[0]
        groups[owner].append(chosen)
        remaining.remove(chosen)

    while remaining:
        best_score: tuple[float, int, int] | None = None
        best_attr = best_owner = -1
        for attr in sorted(remaining):
            for owner, group in enumerate(groups):
                if len(group) >= counts[owner]:
                    continue
                score = (
                    float(np.mean([affinity[attr, other] for other in group]))
                    if group
                    else 0.0
                )
                # Break ties in favor of smaller clients, preserving the public ratios.
                candidate = (score, -len(group), -owner)
                if best_score is None or candidate > best_score:
                    best_score = candidate
                    best_attr = attr
                    best_owner = owner
        if best_score is None:
            raise ValueError("无法按属性容量构造 affinity 分组")
        groups[best_owner].append(best_attr)
        remaining.remove(best_attr)

    return tuple(tuple(group) for group in groups)


def _reduce_domains(
    data: np.ndarray,
    domains: list[set[int]],
    ratio: float,
    seed: int,
) -> tuple[np.ndarray, list[set[int]], dict[str, object]]:
    """Randomly merge values into a smaller public domain.

    This is a benchmark transformation, not an additional privacy mechanism.  The
    mapping is deterministic for a fixed seed and is applied before vertical
    partitioning so every client sees the same transformed rows.
    """
    if ratio >= 1.0:
        return data, domains, {"ratio": 1.0, "changed": False}
    rng = np.random.default_rng(seed)
    transformed = data.copy()
    mappings: list[dict[int, int]] = []
    target_sizes: list[int] = []
    for attr, values_set in enumerate(domains):
        values = sorted(int(value) for value in values_set)
        target = max(1, int(round(len(values) * ratio)))
        target = min(target, len(values))
        target_sizes.append(target)
        if target == len(values):
            mapping = {value: index for index, value in enumerate(values)}
        else:
            permuted = list(values)
            rng.shuffle(permuted)
            mapping = {
                value: int(index * target / len(permuted))
                for index, value in enumerate(permuted)
            }
        mappings.append(mapping)
        transformed[:, attr] = np.asarray(
            [mapping[int(value)] for value in transformed[:, attr]],
            dtype=np.int64,
        )
    reduced_domains = [set(range(size)) for size in target_sizes]
    return transformed, reduced_domains, {
        "ratio": ratio,
        "changed": True,
        "target_domain_sizes": target_sizes,
        "seed": seed,
    }


def _merge_features(
    data: np.ndarray,
    ratio: float,
    num_clients: int,
    seed: int,
) -> tuple[np.ndarray, list[set[int]], dict[str, object]]:
    """Merge original columns into reversible tuple-valued attributes."""
    n_attributes = data.shape[1]
    target = max(num_clients, int(round(n_attributes * ratio)))
    target = min(target, n_attributes)
    if target == n_attributes:
        return data, [set(int(v) for v in np.unique(data[:, i])) for i in range(n_attributes)], {
            "ratio": 1.0,
            "changed": False,
            "groups": [[i] for i in range(n_attributes)],
        }
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(n_attributes))
    groups: list[list[int]] = [[] for _ in range(target)]
    for index, attribute in enumerate(order):
        groups[index % target].append(int(attribute))
    # Stable tuple-to-integer encoding keeps the core protocol integer-only.
    merged = np.empty((data.shape[0], target), dtype=np.int64)
    domains: list[set[int]] = []
    for new_attr, group in enumerate(groups):
        values, inverse = np.unique(data[:, group], axis=0, return_inverse=True)
        merged[:, new_attr] = inverse.astype(np.int64)
        domains.append(set(range(len(values))))
    return merged, domains, {
        "ratio": ratio,
        "changed": True,
        "groups": groups,
        "target_attributes": target,
        "effective_ratio": target / n_attributes,
    }


@dataclass(frozen=True)
class VerticalDataset:
    data: np.ndarray
    attributes: tuple[int, ...]
    partitions: tuple[tuple[int, ...], ...]
    # Public domains from the complete file, including values absent from sampled rows.
    domains: tuple[tuple[int, ...], ...]
    transform: dict[str, object] | None = None

    @property
    def n_rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_attributes(self) -> int:
        return int(self.data.shape[1])

    @property
    def owner_by_attribute(self) -> dict[int, int]:
        return {
            attr: owner
            for owner, attrs in enumerate(self.partitions)
            for attr in attrs
        }

    def membership(self, itemset: Itemset) -> np.ndarray:
        mask = np.ones(self.n_rows, dtype=bool)
        for attr, value in itemset:
            mask &= self.data[:, attr] == value
        return mask

    def support(self, itemset: Itemset) -> int:
        return int(np.count_nonzero(self.membership(itemset)))


def load_vertical_csv(
    csv_path: str | Path,
    num_clients: int,
    max_rows: int | None = None,
    attribute_ratios: tuple[float, ...] | None = None,
    partition_seed: int | None = None,
    partition_strategy: str = "auto",
    row_sampling_seed: int | None = 2026,
    sample_ratio: float | None = None,
    feature_ratio: float | None = None,
    domain_ratio: float | None = None,
    transform_seed: int | None = None,
) -> VerticalDataset:
    if sample_ratio is not None and not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio 必须在 (0, 1] 内")
    if feature_ratio is not None and not 0 < feature_ratio <= 1:
        raise ValueError("feature_ratio 必须在 (0, 1] 内")
    if domain_ratio is not None and not 0 < domain_ratio <= 1:
        raise ValueError("domain_ratio 必须在 (0, 1] 内")
    if max_rows is not None and sample_ratio is not None:
        raise ValueError("max_rows 与 sample_ratio 不能同时设置")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows 必须为正数")
    path = Path(csv_path)
    if sample_ratio is not None:
        with path.open("r", encoding="utf-8", newline="") as handle:
            total_rows = max(sum(1 for _ in handle) - 1, 1)
        max_rows = max(1, int(round(total_rows * sample_ratio)))
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        headings = tuple(int(value) for value in next(reader))
        rows: list[list[int]] = []
        domain_values: list[set[int]] = [set() for _ in headings]
        rng = (
            np.random.default_rng(row_sampling_seed)
            if max_rows is not None and row_sampling_seed is not None
            else None
        )
        for index, row in enumerate(reader):
            values = [int(value) for value in row]
            if len(values) != len(headings):
                raise ValueError(f"第 {index + 2} 行列数不一致")
            for attribute, value in enumerate(values):
                domain_values[attribute].add(value)
            if max_rows is None:
                rows.append(values)
            elif len(rows) < max_rows:
                rows.append(values)
            elif rng is not None:
                # Reservoir sampling avoids the ordering bias of taking a file prefix.
                replacement = int(rng.integers(0, index + 1))
                if replacement < max_rows:
                    rows[replacement] = values

    data = np.asarray(rows, dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != len(headings):
        raise ValueError(f"CSV 数据形状不合法: {data.shape}, headings={len(headings)}")
    if headings != tuple(range(len(headings))):
        raise ValueError("当前实现要求属性编号为从 0 开始的连续整数")
    transform_seed = row_sampling_seed if transform_seed is None else transform_seed
    transform: dict[str, object] = {
        "sample_ratio": sample_ratio,
        "feature_ratio": feature_ratio,
        "domain_ratio": domain_ratio,
        "seed": transform_seed,
    }
    if domain_ratio is not None:
        data, domain_values, domain_transform = _reduce_domains(
            data, domain_values, domain_ratio, int(transform_seed or 0)
        )
        transform["domain"] = domain_transform
    if feature_ratio is not None:
        data, domain_values, feature_transform = _merge_features(
            data, feature_ratio, num_clients, int(transform_seed or 0) + 7919
        )
        transform["features"] = feature_transform
    headings = tuple(range(data.shape[1]))
    if num_clients > len(headings):
        raise ValueError("客户端数量不能超过属性数量")

    if partition_strategy not in {"auto", "order", "random", "affinity"}:
        raise ValueError("partition_strategy 必须是 auto、order、random 或 affinity")
    if partition_strategy == "auto":
        partition_strategy = "random" if partition_seed is not None else "order"
    partition_headings = np.asarray(headings, dtype=np.int64)
    if partition_strategy == "random" and partition_seed is not None:
        partition_headings = np.random.default_rng(partition_seed).permutation(
            partition_headings
        )

    if attribute_ratios is None:
        attribute_counts = [
            len(part) for part in np.array_split(partition_headings, num_clients)
        ]
    else:
        if len(attribute_ratios) != num_clients or any(
            ratio <= 0 for ratio in attribute_ratios
        ):
            raise ValueError("属性比例必须包含 num_clients 个正数")
        weights = np.asarray(attribute_ratios, dtype=np.float64)
        quotas = len(headings) * weights / weights.sum()
        counts = np.floor(quotas).astype(int)
        remainder = len(headings) - int(counts.sum())
        order = sorted(
            range(num_clients),
            key=lambda index: (-(quotas[index] - counts[index]), index),
        )
        for index in order[:remainder]:
            counts[index] += 1

        # Ensure each owner receives an attribute, even under extreme ratios.
        for empty in np.flatnonzero(counts == 0):
            donors = np.flatnonzero(counts > 1)
            if len(donors) == 0:
                raise ValueError("属性数量不足，无法按比例为每方至少分配一个属性")
            donor = int(donors[np.argmax(counts[donors] - quotas[donors])])
            counts[donor] -= 1
            counts[empty] += 1
        attribute_counts = counts.tolist()

    if partition_strategy == "affinity":
        partitions = _affinity_partitions(data, attribute_counts, partition_seed)
        domains = tuple(tuple(sorted(values)) for values in domain_values)
        return VerticalDataset(
            data=data,
            attributes=headings,
            partitions=partitions,
            domains=domains,
            transform=transform,
        )

    partitions = []
    start = 0
    for count in attribute_counts:
        stop = start + count
        partitions.append(
            tuple(int(attr) for attr in partition_headings[start:stop])
        )
        start = stop
    partitions = tuple(partitions)
    domains = tuple(tuple(sorted(values)) for values in domain_values)
    return VerticalDataset(
        data=data,
        attributes=headings,
        partitions=partitions,
        domains=domains,
        transform=transform,
    )
