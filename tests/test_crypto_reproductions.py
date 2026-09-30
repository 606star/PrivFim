"""Correctness and separation tests, not proofs of cryptographic security."""
import itertools
from collections import Counter

import numpy as np
import pytest

from experiments.crypto_protocols import generate_paillier_keypair
from experiments.crypto_reproductions.export_k_sweep import rank_metrics
from experiments.crypto_reproductions.preflight import candidate_count
from experiments.crypto_reproductions.two_round import first_round
from experiments.crypto_reproductions.run import BINARY, public_candidates, run_nipp
from experiments.crypto_reproductions.vpp import (
    BlindedComparison, BlindedEvaluator, eclat, frequency_padding,
    run_vpp, subsets, verification_pattern, verify_border,
)
from privfim.data import VerticalDataset


def fixture():
    return VerticalDataset(data=np.array([[0,0,0], [0,0,1], [0,1,1], [0,1,1]]),
                           attributes=(0,1,2), partitions=((0,), (1,2)),
                           domains=((0,1), (0,1), (0,1)))


def test_export_rank_metrics_uses_complete_supports():
    truth = {((0, 0),): 3, ((0, 1),): 2, ((1, 0),): 1}
    assert rank_metrics(truth, truth, 2) == (1., 1.)
    with pytest.raises(ValueError, match="outside complete candidate"):
        rank_metrics(truth, truth, 4)


def test_preflight_candidate_count_matches_enumeration():
    ds = fixture()
    assert candidate_count([len(domain) for domain in ds.domains], 4) == len(public_candidates(ds, 4, 100))


def test_hybrid_round1_candidates_are_bounded_and_valid():
    candidates, details = first_round(fixture(), epsilon=1., k=3, seed=2026)
    assert 1 <= len(candidates) <= 6
    assert len(candidates) == len(set(candidates))
    assert all(len(key) <= 4 for key in candidates)
    assert details["candidate_source"] == "private_round1_noisy_items_svsm_top_2k"


def test_vpp_queries_only_selected_itemsets_and_verification_patterns():
    selected = [((0, 0),), ((0, 0), (1, 0))]
    supports, transcript = run_vpp(fixture(), threshold=0, key_bits=256,
                                   candidates=selected, candidate_limit=100)
    assert set(supports) == set(selected)
    assert all(supports[key] == fixture().support(key) for key in selected)
    assert transcript["candidate_source"] == "shared_round1_candidates"
    assert all(check["passed"] for check in transcript["border_checks"])


def test_vpp_selected_public_value_absent_from_rows_has_zero_support():
    selected = [((0, 1),), ((0, 1), (1, 0))]
    supports, _ = run_vpp(fixture(), threshold=0, key_bits=256,
                          candidates=selected, candidate_limit=100)
    assert supports == {key: fixture().support(key) for key in selected}


@pytest.mark.skipif(not BINARY.exists(), reason="native TFHE adapter has not been built")
def test_nipp_selected_query_may_omit_vertical_owners(tmp_path):
    data = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    ds = VerticalDataset(data=data, attributes=(0, 1, 2, 3),
                         partitions=((0,), (1,), (2,), (3,)),
                         domains=((0, 1),) * 4)
    key = ((0, 0),)
    supports, _ = run_nipp(ds, [key], 0, tmp_path, selected_items_only=True,
                           public_query=True)
    assert supports[key] == 3


def test_blinded_comparison_equality_both_directions_and_extremes():
    pk, sk = generate_paillier_keypair(256)  # small keys exclusively for tests
    evaluator = BlindedEvaluator(sk)
    compare = BlindedComparison(pk, evaluator, 100, 64)
    assert not hasattr(compare, "private")
    for x, y in itertools.product((0, 1, 2, 100), repeat=2):
        for _ in range(3):
            assert sk.decrypt(compare.greater(pk.encrypt(x), pk.encrypt(y))) == int(x > y)
    assert evaluator.calls == 48
    with pytest.raises(ValueError, match="overflow"):
        BlindedComparison(pk, evaluator, 1 << 220, 64)


@pytest.mark.parametrize("threshold", [0, 1, 3])
@pytest.mark.parametrize("width", [2, 3, 4])
def test_verification_pattern_and_tampering(threshold, width):
    afi, aii, rows = verification_pattern(0, threshold, width)
    frequent = set().union(*(subsets(key) for key in afi))
    assert all(sum(key <= row for row in rows) > threshold for key in afi)
    assert all(sum(key <= row for row in rows) <= threshold for key in aii)
    assert verify_border(afi, aii, frequent)["passed"]
    assert not verify_border(afi, aii, frequent - {next(iter(afi))})["passed"]
    assert not verify_border(afi, aii, frequent | aii)["passed"]


@pytest.mark.parametrize("anonymity", [2, 3, 4, 5])
def test_frequency_padding_reaches_required_class_sizes(anonymity):
    rows = [{"a", "b"}, {"a"}, {"a", "c"}, {"a", "d"}]
    universe = set("abcdefg")
    padding, _ = frequency_padding(rows, universe, anonymity)
    counts = Counter(x for row in rows + padding for x in row)
    classes = Counter(counts[item] for item in universe)
    assert min(classes.values()) >= anonymity


def test_eclat_discovers_every_positive_apparent_itemset():
    index = {"a": {0,1,2}, "b": {1,2}, "c": {2,3}}
    found = dict(eclat(index, 3, 100))
    expected = {key: set.intersection(*(index[x] for x in key)) for length in range(1,4)
                for key in itertools.combinations(sorted(index), length)}
    assert found == {key: tids for key, tids in expected.items() if tids}
    with pytest.raises(RuntimeError, match="limit"):
        list(eclat(index, 3, 1))


def test_public_candidates_do_not_depend_on_support():
    ds = fixture()
    candidates = public_candidates(ds, 4, 100)
    assert len(candidates) == 26
    assert ((0,1), (1,0), (2,0)) in candidates  # absent even in original records
    other = VerticalDataset(data=np.ones_like(ds.data), attributes=ds.attributes,
                            partitions=ds.partitions, domains=ds.domains)
    assert candidates == public_candidates(other, 4, 100)
    with pytest.raises(ValueError, match="no oracle"):
        public_candidates(ds, 4, 10)


@pytest.mark.parametrize("threshold", [0, 2, 4])
def test_vpp_cross_owner_support_and_border_checks(threshold):
    ds = fixture()
    supports, transcript = run_vpp(ds, threshold=threshold, key_bits=256,
                                    max_size=3, candidate_limit=1000)
    for key in public_candidates(ds, 3, 100):
        assert supports.get(key, 0) == ds.support(key)
    assert all(owner["frequency_anonymity_verified"] for owner in transcript["owners"])
    assert all(check["passed"] for check in transcript["border_checks"])
    assert transcript["comparison_calls"] > 0
    assert not transcript["publication_eligible"]
