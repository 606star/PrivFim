import numpy as np
import pytest

from privfim.fo import (
    oue_column_sums,
    oue_encode,
    oue_estimate_count,
    oue_intersection_estimate,
    oue_positive_membership,
)


@pytest.mark.parametrize("budget", [1, 19, 10000])
def test_chunked_oue_matches_full_report_for_same_seed(budget):
    values = np.asarray([2, 0, 7, 2, 7, 0] * 17)
    domain = (0, 2, 4, 7)
    full = oue_encode(values, domain, epsilon=0.13, seed=2026)
    streamed = oue_column_sums(values, domain, 0.13, 2026, budget)
    np.testing.assert_array_equal(streamed, full.sum(axis=0))


def test_oue_count_is_close_to_true_count_with_large_population():
    values = np.concatenate([np.zeros(20_000, dtype=int), np.ones(10_000, dtype=int)])
    reports = oue_encode(values, (0, 1), epsilon=5.0, seed=11)
    estimate = oue_estimate_count(reports, epsilon=5.0, category_index=0)
    assert abs(estimate - 20_000) < 600


def test_aligned_oue_reports_estimate_intersection():
    n_rows = 40_000
    first = np.zeros(n_rows, dtype=bool)
    second = np.zeros(n_rows, dtype=bool)
    first[:22_000] = True
    second[:12_000] = True
    second[22_000:30_000] = True
    reports = [
        oue_positive_membership(first, epsilon=6.0, seed=2),
        oue_positive_membership(second, epsilon=6.0, seed=3),
    ]
    estimate = oue_intersection_estimate(reports, [6.0, 6.0], n_rows)
    assert abs(estimate - 12_000) < 900
