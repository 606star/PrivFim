from __future__ import annotations

from collections import defaultdict

import numpy as np

from .types import AlphaReport, Candidate, NoisyCountReport


# Compact wire format: uint32 lengths/attribute IDs, uint16 item counts,
# int64 attribute values, and float64 guessed counts, statistics, and Alpha.
U16_BYTES = 2
U32_BYTES = 4
I64_BYTES = 8
F64_BYTES = 8
BOOL_BYTES = 1


def _text_bytes(value: str) -> int:
    return U16_BYTES + len(value.encode("utf-8"))


def _itemset_bytes(itemset: tuple[tuple[int, int], ...]) -> int:
    return U16_BYTES + len(itemset) * (U32_BYTES + I64_BYTES)


def round1_report_bytes(report: NoisyCountReport) -> int:
    header = _text_bytes(report.client_id) + 3 * F64_BYTES + U32_BYTES
    entry = U32_BYTES + I64_BYTES + F64_BYTES
    return header + len(report.counts) * entry


def round1_uplink_bytes(reports: list[NoisyCountReport]) -> tuple[int, dict[str, int]]:
    per_client = {report.client_id: round1_report_bytes(report) for report in reports}
    return sum(per_client.values()), per_client


def candidate_message_bytes(
    candidates: list[Candidate], include_guessed_count: bool = False
) -> int:
    frequency_bytes = F64_BYTES if include_guessed_count else 0
    return U32_BYTES + sum(
        _itemset_bytes(candidate.itemset) + frequency_bytes
        for candidate in candidates
    )


def candidate_downlink_bytes(
    candidates: list[Candidate],
    client_count: int,
    include_guessed_count: bool = False,
    noisy_singleton_counts: dict[tuple[int, int], float] | None = None,
) -> tuple[int, int]:
    one_message = candidate_message_bytes(candidates, include_guessed_count)
    if noisy_singleton_counts is not None:
        # Entry format: attribute:uint32, value:int64, noisy_count:float64.
        one_message += U32_BYTES + len(noisy_singleton_counts) * (
            U32_BYTES + I64_BYTES + F64_BYTES
        )
    return one_message * client_count, one_message


def binning_message_bytes(
    binning: tuple[tuple[tuple[int, ...], ...], ...] | None,
    attributes: tuple[int, ...] | None = None,
) -> int:
    """Encode the public binning plan required by the second round.

    Standard Bin uses contiguous ranges, requiring only end cut points. BinP may
    protect candidate values in noncontiguous bins, which use explicit value lists.
    Both encodings depend only on the public plan.
    """
    if binning is None:
        return 0
    selected_attributes = (
        tuple(range(len(binning))) if attributes is None else tuple(attributes)
    )
    if any(attr < 0 or attr >= len(binning) for attr in selected_attributes):
        raise ValueError("分箱通信中包含不存在的属性")
    total = U32_BYTES
    for attr in selected_attributes:
        buckets = tuple(tuple(int(value) for value in bucket) for bucket in binning[attr])
        contiguous = all(
            buckets[index][-1] + 1 == buckets[index + 1][0]
            for index in range(len(buckets) - 1)
        )
        if contiguous:
            total += U32_BYTES + U16_BYTES + max(len(buckets) - 1, 0) * I64_BYTES
        else:
            total += U32_BYTES + U16_BYTES
            total += sum(U16_BYTES + len(bucket) * I64_BYTES for bucket in buckets)
    return total


def binning_downlink_bytes(
    binning: tuple[tuple[tuple[int, ...], ...], ...] | None,
    attributes_by_client: dict[str, tuple[int, ...]] | None,
) -> tuple[int, dict[str, int]]:
    """Count MAP-M-Bin plan bytes sent selectively to the relevant attribute owners."""
    if binning is None:
        return 0, {}
    if attributes_by_client is None:
        raise ValueError("MAP-M-Bin 通信核算需要每个客户端的属性归属")
    per_client = {
        client_id: binning_message_bytes(binning, attributes)
        for client_id, attributes in attributes_by_client.items()
    }
    return sum(per_client.values()), per_client


def round2_uplink_bytes(reports: list[AlphaReport]) -> tuple[int, dict[str, int]]:
    by_client: dict[str, list[AlphaReport]] = defaultdict(list)
    for report in reports:
        by_client[report.client_id].append(report)

    per_client = {}
    for client_id, client_reports in by_client.items():
        size = _text_bytes(client_id) + U32_BYTES
        for report in client_reports:
            alpha = np.asarray(report.alpha, dtype="<f8")
            size += (
                _itemset_bytes(report.key)
                + 3 * F64_BYTES
                + BOOL_BYTES
                + U32_BYTES
                + alpha.nbytes
            )
        per_client[client_id] = size
    return sum(per_client.values()), per_client


def communication_summary(
    round1_reports: list[NoisyCountReport],
    candidates: list[Candidate],
    round2_reports: list[AlphaReport],
    client_count: int,
    include_candidate_guessed_count: bool = False,
    noisy_singleton_counts: dict[tuple[int, int], float] | None = None,
    binning: tuple[tuple[tuple[int, ...], ...], ...] | None = None,
    binning_attributes_by_client: dict[str, tuple[int, ...]] | None = None,
) -> dict:
    round1_total, round1_clients = round1_uplink_bytes(round1_reports)
    downlink_total, downlink_per_client = candidate_downlink_bytes(
        candidates,
        client_count,
        include_candidate_guessed_count,
        noisy_singleton_counts,
    )
    binning_total, binning_clients = binning_downlink_bytes(
        binning, binning_attributes_by_client
    )
    round2_total, round2_clients = round2_uplink_bytes(round2_reports)
    client_ids = set(round1_clients) | set(round2_clients) | set(binning_clients)
    per_client_total = {
        client_id: round1_clients.get(client_id, 0)
        + downlink_per_client
        + binning_clients.get(client_id, 0)
        + round2_clients.get(client_id, 0)
        for client_id in client_ids
    }
    return {
        "wire_format": "compact_binary_v1",
        "candidate_includes_guessed_count": include_candidate_guessed_count,
        "candidate_includes_noisy_singleton_counts": (
            noisy_singleton_counts is not None
        ),
        "round1_uplink_bytes": round1_total,
        "candidate_downlink_bytes": downlink_total,
        "candidate_downlink_per_client_bytes": downlink_per_client,
        "binning_downlink_bytes": binning_total,
        "binning_downlink_by_client_bytes": binning_clients,
        "max_binning_downlink_per_client_bytes": max(
            binning_clients.values(), default=0
        ),
        "round2_uplink_bytes": round2_total,
        "total_bytes": round1_total + downlink_total + binning_total + round2_total,
        "per_client_total_bytes": per_client_total,
        "max_client_total_bytes": max(per_client_total.values(), default=0),
    }
