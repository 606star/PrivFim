from __future__ import annotations

import numpy as np
import pytest

from experiments.crypto_protocols import (
    NippTFHEReference,
    ProtocolTranscript,
    _generate_ta,
    _build_vpp_owner_records,
    _one_hot_transactions,
    _rank_candidate_metrics,
    PaillierEvaluator,
    generate_paillier_keypair,
    run_nipp_fim,
    run_nipp_fim_packed,
    run_vpp_ofim,
    run_vpp_ofim_packed,
)
from privfim.data import VerticalDataset


def _tiny_dataset() -> VerticalDataset:
    data = np.asarray(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.int64,
    )
    return VerticalDataset(
        data=data,
        attributes=(0, 1, 2),
        partitions=((0,), (1,), (2,)),
        domains=((0, 1), (0, 1), (0, 1)),
    )


def test_paillier_round_trip_and_addition():
    public, private = generate_paillier_keypair(256)
    encrypted = public.encrypt(7)
    assert private.decrypt(encrypted) == 7
    assert private.decrypt(public.add(public.encrypt(4), public.encrypt(9))) == 13


def test_nipp_reference_boolean_semantics_only():
    transcript = ProtocolTranscript("NIPP-FIM", "test", "test")
    backend = NippTFHEReference(transcript)
    encrypted_true = backend.encrypt_bit(1)
    encrypted_false = backend.encrypt_bit(0)
    assert backend.decrypt_bit(backend.and_(encrypted_true, encrypted_false)) == 0
    assert backend.decrypt_bit(
        backend.compare_less(backend.encrypt_integer(2, 3), backend.encrypt_integer(3, 3))
    ) == 1
    # A private-looking field is still plaintext, not a secrecy guarantee.
    assert encrypted_true._value == 1
    assert transcript.operations["HomAND"] >= 1


def test_both_protocols_return_exact_support_on_tiny_dataset():
    dataset = _tiny_dataset()
    candidates = (((0, 0),), ((0, 0), (1, 0)), ((1, 1), (2, 0)))
    supports = {itemset: dataset.support(itemset) for itemset in candidates}
    nipp_metrics, nipp_transcript = run_nipp_fim(
        dataset,
        candidates,
        top_k=2,
        evaluation_supports=supports,
    )
    vpp_metrics, vpp_transcript = run_vpp_ofim(
        dataset,
        candidates,
        top_k=2,
        paillier_bits=256,
        artificial_fraction=0.25,
        seed=17,
        evaluation_supports=supports,
    )
    assert nipp_metrics["f1"] == 1.0
    assert vpp_metrics["f1"] == 1.0
    assert nipp_transcript.execution_status == "protocol_executed_reference_backend"
    assert vpp_transcript.execution_status == "protocol_executed_reference_adaptation"
    assert nipp_transcript.metadata["vertical"] is True
    assert nipp_transcript.metadata["row_alignment"] == "shared_tID_position"
    assert set(nipp_transcript.timings) >= {
        "owner_encrypt_upload",
        "csp_encrypted_support_queries",
        "ranking_and_metrics",
    }
    assert vpp_transcript.metadata["vertical"] is True
    assert vpp_transcript.metadata["row_alignment"] == "H(TID)"
    assert vpp_transcript.operations["hash_tid"] > 0
    assert vpp_transcript.operations["secure_compare_support_indicator"] >= len(candidates)
    assert vpp_transcript.operations["secure_compare_support_threshold"] == len(candidates)
    assert set(vpp_transcript.timings) >= {
        "owner_local_artificial_data",
        "paillier_keygen",
        "owner_encrypt_and_upload",
        "csp_evaluator_support_queries",
        "ranking_and_metrics",
    }
    assert vpp_transcript.artificial_rows >= 1
    assert vpp_transcript.metadata["artificial_mode"] == "paper"
    assert vpp_transcript.metadata["afi_count"] > 0
    assert nipp_transcript.to_dict()["runtime_verified"] is False
    assert vpp_transcript.to_dict()["runtime_verified"] is False


def test_ta_contains_only_group_unions_and_preserves_feasible_negative_border():
    afi = {("a",), ("b",)}
    rows = _generate_ta(afi, set(), [], supp_min=2, min_rows=3)
    assert len(rows) >= 3
    assert all(set(row) == {"a", "b"} for row in rows)


def test_packed_nipp_reference_matches_gate_reference():
    dataset = _tiny_dataset()
    candidates = (((0, 0),), ((0, 0), (1, 0)), ((1, 1), (2, 0)))
    supports = {itemset: dataset.support(itemset) for itemset in candidates}
    gate_metrics, _ = run_nipp_fim(
        dataset, candidates, top_k=2, evaluation_supports=supports
    )
    packed_metrics, packed_transcript = run_nipp_fim_packed(
        dataset, candidates, top_k=2, evaluation_supports=supports
    )
    assert packed_metrics["f1"] == gate_metrics["f1"]
    assert packed_metrics["ncr"] == gate_metrics["ncr"]
    assert packed_transcript.execution_status == "protocol_executed_reference_packed"
    assert packed_transcript.metadata["cryptographic_wall_clock"] is False


def test_packed_vpp_reference_preserves_candidate_supports():
    dataset = _tiny_dataset()
    candidates = (((0, 0),), ((0, 0), (1, 0)), ((1, 1), (2, 0)))
    supports = {itemset: dataset.support(itemset) for itemset in candidates}
    metrics, transcript = run_vpp_ofim_packed(
        dataset,
        candidates,
        top_k=2,
        paillier_bits=256,
        artificial_fraction=0.25,
        seed=17,
        evaluation_supports=supports,
    )
    assert metrics["f1"] == 1.0
    assert transcript.execution_status == "protocol_executed_reference_packed"
    assert transcript.metadata["cryptographic_wall_clock"] is False
    assert transcript.operations["secure_compare_support_threshold"] == len(candidates)


@pytest.mark.parametrize("status", ["protocol_executed_reference_packed", "protocol_executed_reference_backend",
                                    "protocol_executed_native_tfhe", "protocol_executed_reference_adaptation"])
def test_execution_status_does_not_certify_a_cryptographic_benchmark(status):
    transcript = ProtocolTranscript("test", "test", "public_candidate_oracle", execution_status=status)
    transcript.metadata["cryptographic_wall_clock"] = True
    assert transcript.to_dict()["runtime_verified"] is False
    assert transcript.to_dict()["publication_eligible"] is False


def test_vpp_frequency_flag_is_strict_greater_not_equality():
    public, private = generate_paillier_keypair(256)
    evaluator = PaillierEvaluator(public, private)
    for support in (0, 1, 2, 3, 5):
        flag = evaluator.compare_greater_reference(public.encrypt(support), 2)
        assert private.decrypt(flag) == int(support > 2)


def test_artificial_verification_rows_keep_realness_one():
    records, _ = _build_vpp_owner_records(_tiny_dataset(), .25, 17, "paper")
    seen = 0
    for record in records:
        for tid, flag in zip(record["tids"], record["flags"]):
            if ":TA:" in tid:
                assert flag == 1
                seen += 1
            elif ":F:" in tid:
                assert flag == 0
        assert record["afi"]
    assert seen > 0


def test_native_encoding_follows_each_items_attribute():
    ds = _tiny_dataset()
    items = [(2, 1), (0, 0), (1, 1)]
    encoded = _one_hot_transactions(ds, items)
    np.testing.assert_array_equal(encoded, np.column_stack([ds.data[:, a] == v for a, v in items]))


def test_vpp_public_unobserved_values_return_zero_support():
    from dataclasses import replace
    original = _tiny_dataset()
    ds = replace(original, domains=((0, 1, 2), *original.domains[1:]))
    candidates = (((0, 2),), ((0, 2), (1, 0)), ((0, 0),))
    supports = {key: ds.support(key) for key in candidates}
    _, transcript = run_vpp_ofim(ds, candidates, top_k=2, paillier_bits=256,
                                 candidate_source="public_full_universe", evaluation_supports=supports,
                                 seed=17, support_threshold=1)
    assert transcript.metadata["candidate_discovery"] == "public_full_universe"
    checks = transcript.metadata["threshold_checks"]
    assert [entry["support"] for entry in checks] == [0, 0, 2]
    assert [entry["frequent"] for entry in checks] == [0, 0, 1]
    assert transcript.to_dict()["publication_eligible"] is False


def test_offline_evaluation_does_not_fallback_to_candidate_truth():
    ds = _tiny_dataset()
    candidate = ((1, 1), (2, 0))
    scores = _rank_candidate_metrics(ds, [candidate], 1, {candidate: 1.0})
    assert scores["f1"] == 0


def test_large_domain_evaluation_never_switches_to_candidate_restricted_truth():
    data = np.zeros((4, 70), dtype=np.int64)
    data[0, 0] = 1
    ds = VerticalDataset(data=data, attributes=tuple(range(70)),
                         partitions=(tuple(range(35)), tuple(range(35,70))),
                         domains=tuple((0,1,2) for _ in range(70)))
    candidate = ((0, 1),)
    # 210 单项触发过旧实现的百万组合回退；该候选不在全局 Top-1。
    scores = _rank_candidate_metrics(ds, [candidate], 1, {candidate: 1.0})
    assert scores["f1"] == 0


def test_release_plotter_rejects_unmeasured_rows(tmp_path):
    from workflows.common import write_csv
    from workflows.plot import plot
    source = tmp_path / "aggregate.csv"
    write_csv(source, [{"result_kind": "estimated", "method": "NIPP-FIM"}])
    with pytest.raises(ValueError, match="measured"):
        plot(source, tmp_path / "figures")
