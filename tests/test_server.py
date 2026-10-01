import numpy as np
import pytest

from privfim.server import PrivFimServer
from privfim.types import (
    FM_FULL_ESTIMATOR,
    FM_OTHER_ESTIMATOR,
    FM_OTHER_VALUE,
    MAP_BOUNDS_ONLY_ESTIMATOR,
    MAP_NO_FRECHET_ESTIMATOR,
    AlphaReport,
    Candidate,
    NoisyCountReport,
    SINGLETON_ALPHA,
)


def _report(client_id, key):
    return AlphaReport(
        client_id=client_id,
        key=key,
        alpha=np.asarray([2.0, 3.0, 2.0, 4.0]),
        epsilon=1.0,
        delta=1e-5,
        phantom_count=0.0,
        is_complement=False,
    )


def test_server_averages_client_noisy_n_reports():
    reports = [
        NoisyCountReport("client_0", {}, 0.4, noisy_n=95.0, n_epsilon=0.05),
        NoisyCountReport("client_1", {}, 0.4, noisy_n=105.0, n_epsilon=0.05),
    ]

    assert PrivFimServer.aggregate_noisy_n(reports) == pytest.approx(100.0)


def test_server_requires_noisy_n_from_every_client():
    reports = [NoisyCountReport("client_0", {}, 0.4)]

    with pytest.raises(ValueError, match="每个客户端"):
        PrivFimServer.aggregate_noisy_n(reports)


def test_category_fm_unions_every_non_target_bucket():
    server = PrivFimServer(n_rows=10, partitions=((0,), (1,)))
    candidate = Candidate(
        itemset=((0, 1), (1, 1)), score=1.0, guessed_count=5.0
    )
    reports = [
        _report("client_0", ((0, 0),)),
        _report("client_0", ((0, 1),)),
        _report("client_1", ((1, 0),)),
        _report("client_1", ((1, 1),)),
    ]

    estimates = server.estimate_candidates(
        candidates=[candidate],
        reports=reports,
        mode=SINGLETON_ALPHA,
        gamma=1.0,
        map_step=1,
        max_map_points=10,
        estimator=FM_FULL_ESTIMATOR,
    )

    assert estimates[0].local_blocks == (((0, 0),), ((1, 0),))
    assert 0.0 <= estimates[0].estimated_count <= 10.0


def test_fm_other_unions_other_target_values_and_placeholder():
    server = PrivFimServer(n_rows=10, partitions=((0,),))
    candidate = Candidate(itemset=((0, 1),), score=1.0, guessed_count=5.0)
    reports = [
        _report("client_0", ((0, 1),)),
        _report("client_0", ((0, 2),)),
        _report("client_0", ((0, FM_OTHER_VALUE),)),
    ]

    estimate = server.estimate_candidates(
        candidates=[candidate],
        reports=reports,
        mode=SINGLETON_ALPHA,
        gamma=1.0,
        map_step=1,
        max_map_points=10,
        estimator=FM_OTHER_ESTIMATOR,
    )[0]

    assert estimate.local_blocks == (
        ((0, FM_OTHER_VALUE),),
        ((0, 2),),
    )


def test_map_server_routes_positive_reports_to_direct_intersection(monkeypatch):
    server = PrivFimServer(n_rows=10, partitions=((0,), (1,)))
    candidate = Candidate(
        itemset=((0, 1), (1, 1)), score=1.0, guessed_count=5.0
    )
    reports = [
        _report("client_0", ((0, 1),)),
        _report("client_1", ((1, 1),)),
    ]
    seen = {}

    def fake_marginal(**kwargs):
        return 6.0

    def fake_intersection(**kwargs):
        seen["reports"] = kwargs["reports"]
        seen["block_counts"] = kwargs["block_counts"]
        return 4.0

    monkeypatch.setattr("privfim.server.private_cardinality_map_estimate", fake_marginal)
    monkeypatch.setattr("privfim.server.map_intersection_estimate", fake_intersection)

    estimate = server.estimate_candidates(
        candidates=[candidate],
        reports=reports,
        mode=SINGLETON_ALPHA,
        gamma=1.0,
        map_step=1,
        max_map_points=10,
        estimator="map",
    )[0]

    assert all(not report.is_complement for report in seen["reports"])
    assert seen["block_counts"] == [6.0, 6.0]
    assert estimate.estimated_count == 4.0


def test_map_m_bin_recovers_exact_candidate_from_private_bin_marginals(monkeypatch):
    server = PrivFimServer(
        n_rows=10,
        partitions=((0,), (1,)),
        domains=((0, 1, 2, 3), (0, 1, 2, 3)),
    )
    candidate = Candidate(
        itemset=((0, 0), (1, 2)), score=1.0, guessed_count=5.0
    )
    reports = [
        _report("client_0", ((0, 0),)),
        _report("client_1", ((1, 1),)),
    ]

    monkeypatch.setattr(
        "privfim.server.private_cardinality_map_estimate", lambda **_: 8.0
    )
    monkeypatch.setattr(
        "privfim.server.map_intersection_estimate", lambda **_: 6.0
    )
    estimate = server.estimate_candidates(
        candidates=[candidate],
        reports=reports,
        mode="mixed_itemset_alpha",
        gamma=1.0,
        map_step=1,
        max_map_points=10,
        estimator="map",
        map_m_bin_count=2,
        round1_noisy_counts={(0, 0): 9.0, (0, 1): 1.0, (1, 2): 8.0, (1, 3): 2.0},
    )[0]

    # P(a0=0 | bin{0,1})=0.9，P(a1=2 | bin{2,3})=0.8。
    assert estimate.local_blocks == (((0, 0),), ((1, 1),))
    assert estimate.estimated_count == 6.0 * 0.9 * 0.8


def test_map_m_bin_uses_first_round_private_binning_plan(monkeypatch):
    server = PrivFimServer(
        n_rows=10,
        partitions=((0,), (1,)),
        domains=((0, 1, 2, 3), (0, 1, 2, 3)),
    )
    candidate = Candidate(
        itemset=((0, 2), (1, 0)), score=1.0, guessed_count=5.0
    )
    reports = [
        _report("client_0", ((0, 1),)),
        _report("client_1", ((1, 0),)),
    ]
    binning = (((0,), (1, 2, 3)), ((0, 1), (2, 3)))

    monkeypatch.setattr(
        "privfim.server.private_cardinality_map_estimate", lambda **_: 8.0
    )
    monkeypatch.setattr(
        "privfim.server.map_intersection_estimate", lambda **_: 6.0
    )
    estimate = server.estimate_candidates(
        candidates=[candidate],
        reports=reports,
        mode="mixed_binned_itemset_alpha",
        gamma=1.0,
        map_step=1,
        max_map_points=10,
        estimator="map",
        map_m_bin_count=2,
        map_m_binning=binning,
        round1_noisy_counts={(0, 1): 1.0, (0, 2): 8.0, (0, 3): 1.0, (1, 0): 6.0, (1, 1): 4.0},
    )[0]

    # The private plan maps a0=2 to {1,2,3} and a1=0 to {0,1}, with recovery
    # probabilities 8/10 and 6/10, verifying no fallback to equal-width public bins.
    assert estimate.local_blocks == (((0, 1),), ((1, 0),))
    assert estimate.estimated_count == pytest.approx(6.0 * 0.8 * 0.6)


def test_map_server_skips_candidate_when_top_k_hides_required_block():
    server = PrivFimServer(n_rows=10, partitions=((0,), (1,)))
    candidate = Candidate(
        itemset=((0, 1), (1, 1)), score=1.0, guessed_count=5.0
    )
    reports = [_report("client_0", ((0, 1),))]

    estimates = server.estimate_candidates(
        candidates=[candidate],
        reports=reports,
        mode=SINGLETON_ALPHA,
        gamma=1.0,
        map_step=1,
        max_map_points=10,
        estimator="map",
    )

    assert estimates == []


def test_map_component_ablation_removes_only_the_frechet_lower_bound(monkeypatch):
    server = PrivFimServer(n_rows=10, partitions=((0,), (1,)))
    candidate = Candidate(
        itemset=((0, 1), (1, 1)), score=1.0, guessed_count=5.0
    )
    reports = [
        _report("client_0", ((0, 1),)),
        _report("client_1", ((1, 1),)),
    ]
    seen = {}

    monkeypatch.setattr(
        "privfim.server.private_cardinality_map_estimate", lambda **_: 6.0
    )

    def fake_intersection(**kwargs):
        seen["use_frechet_lower_bound"] = kwargs["use_frechet_lower_bound"]
        return 4.0

    monkeypatch.setattr("privfim.server.map_intersection_estimate", fake_intersection)
    estimate = server.estimate_candidates(
        candidates=[candidate],
        reports=reports,
        mode=SINGLETON_ALPHA,
        gamma=1.0,
        map_step=1,
        max_map_points=10,
        estimator=MAP_NO_FRECHET_ESTIMATOR,
    )[0]

    assert seen["use_frechet_lower_bound"] is False
    assert estimate.estimated_count == 4.0


def test_map_bounds_only_ablation_uses_midpoint_without_joint_likelihood(monkeypatch):
    server = PrivFimServer(n_rows=10, partitions=((0,), (1,)))
    candidate = Candidate(
        itemset=((0, 1), (1, 1)), score=1.0, guessed_count=5.0
    )
    reports = [
        _report("client_0", ((0, 1),)),
        _report("client_1", ((1, 1),)),
    ]
    monkeypatch.setattr(
        "privfim.server.private_cardinality_map_estimate", lambda **_: 6.0
    )

    def should_not_run(**_):
        raise AssertionError("Bounds-only 消融不能调用联合 Alpha 似然")

    monkeypatch.setattr("privfim.server.map_intersection_estimate", should_not_run)
    estimate = server.estimate_candidates(
        candidates=[candidate],
        reports=reports,
        mode=SINGLETON_ALPHA,
        gamma=1.0,
        map_step=1,
        max_map_points=10,
        estimator=MAP_BOUNDS_ONLY_ESTIMATOR,
    )[0]

    assert estimate.estimated_count == 4.0
