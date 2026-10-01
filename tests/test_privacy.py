import math

import pytest

from privfim.privacy import (
    dpfm_coordinate_epsilon,
    measurement_groups,
    overlap_rdp_account,
    public_measurement_group_overlap_bound,
    public_overlap_bound,
    split_budget,
    split_grouped_budget,
)


def test_public_overlap_bound_groups_incompatible_singleton_buckets():
    keys = (
        ((0, 0),),
        ((0, 1),),
        ((1, 0),),
        ((0, 1), (1, 0)),
        ((0, 0), (2, 3)),
    )

    # A record matches at most one attribute-0 singleton bin; count both joint keys conservatively.
    assert public_overlap_bound(keys) == 4


def test_split_budget_is_still_fixed_per_key_allocation():
    budget = split_budget(epsilon=2.0, delta=1e-5, key_count=4)

    assert budget.epsilon == 0.5
    assert budget.delta == pytest.approx(2.5e-6)


def test_measurement_group_budget_reuses_budget_for_mutually_exclusive_buckets():
    keys = (
        ((0, 0),),
        ((0, 1),),
        ((1, 0),),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 0)),
    )

    allocation = split_grouped_budget(epsilon=6.0, delta=3e-5, keys=keys)

    assert measurement_groups(keys) == {
        (0,): (((0, 0),), ((0, 1),)),
        (1,): (((1, 0),),),
        (0, 1): (((0, 0), (1, 0)), ((0, 1), (1, 0))),
    }
    # Split the budget across three projection groups, without splitting again across two bins in a group.
    assert all(
        budget.epsilon == pytest.approx(2.0)
        and budget.delta == pytest.approx(1e-5)
        for budget in allocation.budget_by_key.values()
    )
    assert public_measurement_group_overlap_bound(keys) == 3


def test_overlap_rdp_account_uses_weighted_hit_bound_and_optimized_order():
    eta_a = dpfm_coordinate_epsilon(0.2, 1e-6, 128)
    eta_b = dpfm_coordinate_epsilon(0.1, 1e-6, 128)
    account = overlap_rdp_account(
        repetitions=128,
        target_delta=1e-5,
        overlap_bounds={"a": 2, "b": 5},
        coordinate_epsilons={"a": eta_a, "b": eta_b},
    )

    assert account.weighted_overlap == pytest.approx(
        2 * eta_a**2 + 5 * eta_b**2
    )
    assert account.rdp_coefficient == pytest.approx(
        2 * 128 * account.weighted_overlap
    )
    assert account.rdp_order >= 2
    assert account.rdp_epsilon == pytest.approx(
        account.rdp_coefficient * account.rdp_order
        + math.log(1e5) / (account.rdp_order - 1.0)
    )


def test_overlap_rdp_account_rejects_mismatched_clients():
    with pytest.raises(ValueError, match="客户端集合"):
        overlap_rdp_account(
            repetitions=8,
            target_delta=1e-5,
            overlap_bounds={"a": 1},
            coordinate_epsilons={"b": 0.01},
        )
