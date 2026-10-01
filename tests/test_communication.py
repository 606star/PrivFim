import numpy as np

from privfim.communication import (
    binning_downlink_bytes,
    binning_message_bytes,
    candidate_message_bytes,
    communication_summary,
)
from privfim.types import AlphaReport, Candidate, NoisyCountReport


def test_compact_communication_counts_all_three_protocol_transfers():
    round1 = [
        NoisyCountReport(
            client_id="client_0",
            counts={(0, 1): 7.0, (0, 2): 3.0},
            epsilon=0.5,
        )
    ]
    candidates = [Candidate(itemset=((0, 1), (1, 2)), score=1.0, guessed_count=4.0)]
    round2 = [
        AlphaReport(
            client_id="client_0",
            key=((0, 1),),
            alpha=np.ones(16),
            epsilon=0.5,
            delta=1e-5,
            phantom_count=10.0,
            is_complement=True,
        )
    ]

    result = communication_summary(round1, candidates, round2, client_count=2)

    assert result["round1_uplink_bytes"] > 0
    assert result["candidate_downlink_bytes"] > 0
    assert result["binning_downlink_bytes"] == 0
    assert result["round2_uplink_bytes"] > 16 * 8
    assert result["total_bytes"] == (
        result["round1_uplink_bytes"]
        + result["candidate_downlink_bytes"]
        + result["binning_downlink_bytes"]
        + result["round2_uplink_bytes"]
    )
    assert candidate_message_bytes(candidates, include_guessed_count=True) == (
        candidate_message_bytes(candidates) + 8
    )


def test_private_binning_plan_is_counted_in_second_round_downlink():
    binning = (((0, 1, 2), (3,)), ((0,), (1, 2)))
    result = communication_summary(
        round1_reports=[],
        candidates=[],
        round2_reports=[],
        client_count=3,
        binning=binning,
        binning_attributes_by_client={
            "client_0": (0,),
            "client_1": (1,),
            "client_2": (0, 1),
        },
    )

    assert binning_message_bytes(binning) > 0
    assert result["binning_downlink_by_client_bytes"] == {
        "client_0": binning_message_bytes(binning, (0,)),
        "client_1": binning_message_bytes(binning, (1,)),
        "client_2": binning_message_bytes(binning, (0, 1)),
    }
    assert result["binning_downlink_bytes"] == sum(
        result["binning_downlink_by_client_bytes"].values()
    )
    assert binning_downlink_bytes(
        binning, {"client_0": (0,), "client_1": (1,), "client_2": (0, 1)}
    )[0] == result["binning_downlink_bytes"]
    assert result["total_bytes"] == (
        result["candidate_downlink_bytes"] + result["binning_downlink_bytes"]
    )


def test_map_m_downlink_counts_noisy_singleton_frequency_table():
    candidates = [Candidate(itemset=((0, 1), (1, 2)), score=1.0, guessed_count=4.0)]
    without_counts = communication_summary([], candidates, [], client_count=2)
    with_counts = communication_summary(
        [],
        candidates,
        [],
        client_count=2,
        noisy_singleton_counts={(0, 1): 7.0, (1, 2): 6.0},
    )

    # Each client additionally receives a uint32 length and two (uint32,int64,float64) entries.
    assert with_counts["candidate_downlink_bytes"] - without_counts[
        "candidate_downlink_bytes"
    ] == 2 * (4 + 2 * (4 + 8 + 8))
    assert with_counts["candidate_includes_noisy_singleton_counts"] is True


def test_noncontiguous_protected_bins_use_explicit_value_encoding():
    contiguous = (((0, 1), (2, 3)),)
    protected = (((0,), (1, 3), (2,)),)
    assert binning_message_bytes(protected) > binning_message_bytes(contiguous)
