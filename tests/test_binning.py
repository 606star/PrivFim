import pytest

from privfim.binning import binned_itemset, private_quantile_bins, within_bin_probability


def test_private_quantile_bins_use_only_noisy_counts_and_cover_domain():
    domains = ((0, 1, 2, 3, 4, 5),)
    noisy_counts = {
        (0, 0): 50.0,
        (0, 1): 30.0,
        (0, 2): 10.0,
        (0, 3): 5.0,
        (0, 4): 3.0,
        (0, 5): 2.0,
    }

    plan = private_quantile_bins(domains, noisy_counts, bin_count=3)

    # Frequent values receive finer bins; contiguous bins still cover each public value exactly once.
    assert plan == (((0,), (1,), (2, 3, 4, 5)),)
    assert binned_itemset(((0, 4),), domains, 3, plan) == ((0, 2),)
    assert within_bin_probability(0, 2, noisy_counts, domains, 3, plan) == pytest.approx(
        10.0 / 20.0
    )


def test_private_quantile_bins_fall_back_when_private_mass_is_zero():
    plan = private_quantile_bins(((0, 1, 2, 3),), {}, bin_count=2)

    assert plan == (((0, 1), (2, 3)),)
