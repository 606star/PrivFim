import itertools

import numpy as np
import pytest

from experiments.global_truth import exact_topk, support_tie_recall
from privfim.data import VerticalDataset
from privfim.metrics import evaluate_estimates
from privfim.types import CandidateEstimate


def dataset(data):
    data = np.asarray(data, dtype=np.int64)
    attrs = tuple(range(data.shape[1]))
    return VerticalDataset(data, attrs, (attrs,), tuple(tuple(map(int, np.unique(data[:, i]))) for i in attrs))


def brute_force(ds, k, max_size, allowed=None):
    items = [(attr, value) for attr in ds.attributes for value in ds.domains[attr]
             if allowed is None or (attr, value) in allowed]
    found = {}
    for size in range(1, max_size + 1):
        for key in itertools.combinations(items, size):
            if len({attr for attr, _ in key}) == size:
                found[key] = ds.support(key)
    return dict(sorted(found.items(), key=lambda pair: (-pair[1], pair[0]))[:k])


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("k", [1, 5, 15, 200])
def test_global_truth_matches_exhaustive_enumeration(seed, k):
    ds = dataset(np.random.default_rng(seed).integers(0, 3, size=(23, 4)))
    for max_size in [1, 2, 4]:
        actual, _ = exact_topk(ds, k, max_size)
        assert list(actual.items()) == list(brute_force(ds, k, max_size).items())


def test_constant_attributes_and_support_ties_are_not_pruned():
    ds = dataset(np.zeros((8, 5), dtype=int))
    actual, diagnostics = exact_topk(ds, 15)
    assert list(actual.items()) == list(brute_force(ds, 15, 4).items())
    assert diagnostics["itemsets_tied_at_cutoff"] == 30
    other_keys = list(brute_force(ds, 30, 4))[15:]
    assert support_tie_recall(other_keys, actual, {key: 8 for key in other_keys}) == 1


def test_globally_missed_item_does_not_disappear_from_truth():
    ds = dataset([[0, 0], [0, 0], [0, 1], [0, 1], [0, 1]])
    key = ((1, 1),)
    estimate = [CandidateEstimate(key, 3, 3, (key,))]
    global_truth, _ = exact_topk(ds, 1)
    restricted, _ = exact_topk(ds, 1, allowed_items={(1, 1)})
    assert restricted == brute_force(ds, 1, 4, {(1, 1)})
    assert evaluate_estimates(estimate, restricted, 1).f1 == 1
    global_truth[key] = 3
    assert evaluate_estimates(estimate, global_truth, 1).f1 == 0


def test_zero_support_public_domain_values_remain_eligible_when_needed():
    ds = VerticalDataset(np.array([[0, 0]]), (0, 1), ((0, 1),), ((0, 1), (0, 1)))
    actual, _ = exact_topk(ds, 20)
    assert list(actual.items()) == list(brute_force(ds, 20, 4).items())


def test_restricted_audit_truth_matches_full_restricted_universe():
    ds = dataset(np.random.default_rng(10).integers(0, 3, size=(100, 4)))
    allowed = {(0, 0), (0, 1), (1, 0), (2, 1), (3, 2)}
    exact = brute_force(ds, 1000, 4, allowed)
    compressed, _ = exact_topk(ds, 5, allowed_items=allowed)
    selected = list(exact)[::3]
    estimates = [CandidateEstimate(key, 10.0, 0.0, ()) for key in selected]
    compressed.update({key: exact[key] for key in selected})
    assert evaluate_estimates(estimates, compressed, 5) == evaluate_estimates(estimates, exact, 5)


def test_transformed_datasets_have_their_own_exact_truth():
    from privfim.data import _merge_features, _reduce_domains

    source = dataset(np.random.default_rng(23).integers(0, 4, size=(31, 5)))
    merged, _, _ = _merge_features(source.data, 0.6, 2, 2026)
    reduced, _, _ = _reduce_domains(source.data, list(map(set, source.domains)), 0.5, 2026)
    for matrix in (merged, reduced):
        ds = dataset(matrix)
        actual, _ = exact_topk(ds, 15)
        assert list(actual.items()) == list(brute_force(ds, 15, 4).items())


def test_global_audit_counts_missing_topk_predictions(tmp_path):
    from workflows.evaluate import audit
    from privfim.types import CandidateEstimate
    ds = dataset(np.array([[0, 0], [0, 0], [1, 0], [1, 1]]))
    estimates = [CandidateEstimate(((1, 1),), 4., 0., ())]
    scores = audit(ds, estimates, 2, 1, 2, tmp_path)
    assert scores["f1"] == 0.
    assert scores["ncr"] == 0.
    assert scores["evaluation_scope"] == "global_exact"
