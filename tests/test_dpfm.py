import numpy as np
import privfim.dpfm as dpfm

from privfim.dpfm import (
    RankOracle,
    _alpha_min_for_report,
    _bounded_candidate_grid,
    _joint_log_likelihood,
    intersection_bounds,
    map_intersection_estimate,
    map_nested_pair_cardinality_estimate,
    union_complement_estimate,
    union_complement_fm_estimate,
)
from privfim.types import AlphaReport


def test_rank_oracle_is_reproducible_and_private_alpha_has_expected_shape():
    first = RankOracle.build(n_rows=50, m=16, gamma=1.0, seed=7)
    second = RankOracle.build(n_rows=50, m=16, gamma=1.0, seed=7)
    assert np.array_equal(first.ranks, second.ranks)

    membership = np.arange(50) % 2 == 0
    alpha, phantom_count = first.private_alpha(
        membership=membership,
        epsilon=1.0,
        delta=1e-5,
        random_seed=9,
    )
    assert alpha.shape == (16,)
    assert np.all(alpha >= 1)
    assert phantom_count > 0


def test_joint_map_tracks_two_party_overlap_and_keeps_zero_in_domain():
    n_rows = 1000
    oracle = RankOracle.build(n_rows=n_rows, m=1024, gamma=1.0, seed=31)

    def report(client_id, attribute, membership, seed):
        alpha, phantom_count = oracle.private_alpha(
            membership=membership,
            epsilon=100.0,
            delta=1e-5,
            random_seed=seed,
        )
        return AlphaReport(
            client_id=client_id,
            key=((attribute, 1),),
            alpha=alpha,
            epsilon=100.0,
            delta=1e-5,
            phantom_count=phantom_count,
        )

    first = np.zeros(n_rows, dtype=bool)
    first[:600] = True
    second = np.zeros(n_rows, dtype=bool)
    second[:250] = True
    second[600:900] = True
    overlap_estimate = map_intersection_estimate(
        reports=[report("client_0", 0, first, 1), report("client_1", 1, second, 2)],
        n_rows=n_rows,
        gamma=1.0,
        map_step=1,
        max_map_points=n_rows,
    )
    assert abs(overlap_estimate - 250) <= 30

    left = np.zeros(n_rows, dtype=bool)
    left[:500] = True
    right = ~left
    empty_estimate = map_intersection_estimate(
        reports=[report("client_0", 0, left, 3), report("client_1", 1, right, 4)],
        n_rows=n_rows,
        gamma=1.0,
        map_step=1,
        max_map_points=n_rows,
    )
    assert empty_estimate <= 20


def test_intersection_bounds_use_marginal_set_counts():
    assert intersection_bounds(np.asarray([600.0, 550.0]), 1000) == (150, 550)
    assert intersection_bounds(np.asarray([800.0, 700.0, 600.0]), 1000) == (
        100,
        600,
    )


def test_nested_pair_map_fuses_singletons_and_local_joint_alpha():
    n_rows = 800
    oracle = RankOracle.build(n_rows=n_rows, m=4096, gamma=1.0, seed=47)
    left = np.zeros(n_rows, dtype=bool)
    right = np.zeros(n_rows, dtype=bool)
    left[:500] = True
    right[:300] = True
    right[500:650] = True

    reports = []
    for key, membership, seed in (
        (((0, 1),), left, 1),
        (((1, 1),), right, 2),
        (((0, 1), (1, 1)), left & right, 3),
    ):
        alpha, phantom_count = oracle.private_alpha(
            membership=membership,
            epsilon=1e4,
            delta=1e-5,
            random_seed=seed,
        )
        reports.append(
            AlphaReport(
                client_id="client_0",
                key=key,
                alpha=alpha,
                epsilon=1e4,
                delta=1e-5,
                phantom_count=phantom_count,
            )
        )

    estimate = map_nested_pair_cardinality_estimate(
        left_report=reports[0],
        right_report=reports[1],
        joint_report=reports[2],
        n_rows=n_rows,
        gamma=1.0,
        map_step=1,
        max_map_points=n_rows,
    )
    assert abs(estimate - 300) <= 35


def test_map_can_ablate_the_frechet_lower_bound(monkeypatch):
    reports = [
        AlphaReport(
            client_id=f"client_{index}",
            key=((index, 1),),
            alpha=np.asarray([3.0, 4.0, 5.0, 3.0]),
            epsilon=10.0,
            delta=1e-5,
            phantom_count=1.0,
        )
        for index in range(2)
    ]
    captured = []

    def capture_candidates(**kwargs):
        captured.append(kwargs["candidates"])
        return float(kwargs["candidates"][0])

    monkeypatch.setattr(dpfm, "_adaptive_joint_map", capture_candidates)
    arguments = {
        "reports": reports,
        "n_rows": 1000,
        "gamma": 1.0,
        "map_step": 1,
        "max_map_points": 1000,
        "block_counts": np.asarray([600.0, 550.0]),
    }
    map_intersection_estimate(**arguments)
    map_intersection_estimate(**arguments, use_frechet_lower_bound=False)

    assert captured[0][0] == 150.0
    assert captured[1][0] == 0.0
    assert captured[0][-1] == captured[1][-1] == 550.0


def test_adaptive_joint_map_matches_dense_grid_on_private_reports():
    n_rows = 600
    oracle = RankOracle.build(n_rows=n_rows, m=512, gamma=1.0, seed=73)
    memberships = []
    for start, stop in ((0, 400), (120, 470), (220, 520)):
        membership = np.zeros(n_rows, dtype=bool)
        membership[start:stop] = True
        memberships.append(membership)

    reports = []
    for index, membership in enumerate(memberships):
        alpha, phantom_count = oracle.private_alpha(
            membership=membership,
            epsilon=50.0,
            delta=1e-5,
            random_seed=100 + index,
        )
        reports.append(
            AlphaReport(
                client_id=f"client_{index}",
                key=((index, 1),),
                alpha=alpha,
                epsilon=50.0,
                delta=1e-5,
                phantom_count=phantom_count,
            )
        )

    block_counts = np.asarray([400.0, 350.0, 300.0])
    adaptive = map_intersection_estimate(
        reports=reports,
        n_rows=n_rows,
        gamma=1.0,
        map_step=1,
        max_map_points=n_rows,
        block_counts=block_counts,
    )

    lower, upper = intersection_bounds(block_counts, n_rows)
    candidates = _bounded_candidate_grid(lower, upper, 1, n_rows)
    observations = np.stack([report.alpha for report in reports], axis=1)
    unique_observations, occurrences = np.unique(
        observations, axis=0, return_counts=True
    )
    dense_likelihood = _joint_log_likelihood(
        observations=unique_observations,
        occurrences=occurrences,
        candidates=candidates,
        block_counts=block_counts,
        phantom_counts=np.asarray(
            [report.phantom_count for report in reports], dtype=np.float64
        ),
        alpha_mins=np.asarray(
            [_alpha_min_for_report(report, 1.0) for report in reports],
            dtype=np.float64,
        ),
        gamma=1.0,
    )
    dense = float(candidates[int(np.argmax(dense_likelihood))])
    assert adaptive == dense


def test_direct_map_rejects_complement_reports():
    report = AlphaReport(
        client_id="client_0",
        key=((0, 1),),
        alpha=np.asarray([3.0, 4.0, 3.0, 5.0]),
        epsilon=1.0,
        delta=1e-5,
        phantom_count=1.0,
        is_complement=True,
    )
    with np.testing.assert_raises_regex(ValueError, "正向 Alpha"):
        map_intersection_estimate(
            reports=[report],
            n_rows=10,
            gamma=1.0,
            map_step=1,
            max_map_points=10,
        )


def test_union_complement_recovers_intersection_with_private_complement_alphas():
    n_rows = 1000
    oracle = RankOracle.build(n_rows=n_rows, m=1024, gamma=1.0, seed=41)

    first = np.zeros(n_rows, dtype=bool)
    first[:600] = True
    second = np.zeros(n_rows, dtype=bool)
    second[:250] = True
    second[600:900] = True

    reports = []
    for client_id, membership, seed in (
        ("client_0", first, 5),
        ("client_1", second, 6),
    ):
        alpha, phantom_count = oracle.private_alpha(
            membership=~membership,
            epsilon=100.0,
            delta=1e-5,
            random_seed=seed,
        )
        reports.append(
            AlphaReport(
                client_id=client_id,
                key=((0, 1),),
                alpha=alpha,
                epsilon=100.0,
                delta=1e-5,
                phantom_count=phantom_count,
                is_complement=True,
            )
        )

    estimate = union_complement_estimate(
        reports=reports,
        n_rows=n_rows,
        gamma=1.0,
        map_step=1,
        max_map_points=n_rows,
    )
    assert abs(estimate - 250) <= 35

    fm_estimate = union_complement_fm_estimate(
        reports=reports,
        n_rows=n_rows,
        gamma=1.0,
    )
    assert 0 <= fm_estimate <= n_rows
