import numpy as np
import pytest

from privfim.client import VerticalClient
from privfim.dpfm import RankOracle
from privfim.types import (
    FM_FULL_ESTIMATOR,
    FM_OTHER_ESTIMATOR,
    FM_OTHER_VALUE,
    Candidate,
    LOCAL_ITEMSET_ALPHA,
    LOCAL_TOP_ITEMSET_ALPHA,
    LOCAL_TOP_SINGLETON_ALPHA,
    LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
    MIXED_BINNED_ITEMSET_ALPHA,
    MIXED_ITEMSET_ALPHA,
    SINGLETON_ALPHA,
)


def test_enhanced_mode_adds_deduplicated_local_itemset_keys():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0, 1),
        local_data=np.asarray([[1, 2], [1, 3], [0, 2]]),
    )
    candidates = [
        Candidate(itemset=((0, 1), (1, 2)), score=1.0, guessed_count=2.0),
        Candidate(itemset=((0, 1), (1, 2), (2, 5)), score=0.8, guessed_count=1.0),
        Candidate(itemset=((0, 1), (2, 5)), score=0.7, guessed_count=1.0),
        Candidate(itemset=((1, 2), (2, 5)), score=0.6, guessed_count=1.0),
    ]

    singleton = set(client.report_keys(candidates, SINGLETON_ALPHA))
    enhanced = set(client.report_keys(candidates, LOCAL_ITEMSET_ALPHA))

    assert singleton == {((0, 1),), ((1, 2),)}
    # S requires a local joint projection and singleton projections for attributes 0 and 1.
    assert enhanced == {((0, 1),), ((1, 2),), ((0, 1), (1, 2))}

    oracle = RankOracle.build(n_rows=3, m=8, gamma=1.0, seed=11)
    baseline_reports = client.round2_reports(
        candidates, SINGLETON_ALPHA, 3.0, 1e-5, oracle, seed=4
    )
    enhanced_reports = client.round2_reports(
        candidates, LOCAL_ITEMSET_ALPHA, 3.0, 1e-5, oracle, seed=4
    )
    assert all(report.epsilon == 1.5 for report in baseline_reports)
    assert all(report.epsilon == 1.0 for report in enhanced_reports)


def test_map_l_uploads_exact_candidate_projections_without_component_extras():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0, 1),
        local_data=np.asarray([[1, 1], [1, 0], [0, 1]]),
    )
    candidates = [
        Candidate(itemset=((0, 1), (1, 1)), score=0.9, guessed_count=2.0),
        Candidate(itemset=((0, 0), (1, 1)), score=0.4, guessed_count=1.0),
    ]

    selected = set(client.report_keys(candidates, LOCAL_ITEMSET_ALPHA))
    selected_with_legacy_limit = set(
        client.report_keys(candidates, LOCAL_ITEMSET_ALPHA, report_key_limit=1)
    )

    assert selected == {((0, 0), (1, 1)), ((0, 1), (1, 1))}
    assert selected_with_legacy_limit == selected
    assert ((0, 0),) not in selected
    assert ((0, 1),) not in selected
    assert ((1, 1),) not in selected


def test_map_l_adds_singletons_only_when_s_contains_those_projections():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0, 1),
        local_data=np.asarray([[1, 15], [1, 13], [0, 15]]),
    )
    candidates = [
        Candidate(itemset=((0, 1), (1, 15), (2, 7)), score=1.0, guessed_count=3.0),
        Candidate(itemset=((0, 1),), score=0.8, guessed_count=2.0),
        Candidate(itemset=((1, 15), (2, 7)), score=0.7, guessed_count=1.5),
    ]

    assert set(client.report_keys(candidates, LOCAL_ITEMSET_ALPHA)) == {
        ((0, 1),),
        ((1, 15),),
        ((0, 1), (1, 15)),
    }


def test_map_m_top_k_jointly_ranks_singletons_and_local_itemsets():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0, 1),
        local_data=np.asarray([[1, 1], [1, 0], [0, 1]]),
    )
    candidates = [
        Candidate(itemset=((0, 1),), score=1.0, guessed_count=3.0),
        Candidate(itemset=((1, 1),), score=0.9, guessed_count=2.4),
        Candidate(itemset=((0, 0),), score=0.8, guessed_count=1.8),
        Candidate(itemset=((1, 0),), score=0.7, guessed_count=1.5),
        Candidate(itemset=((0, 1), (1, 1)), score=0.6, guessed_count=2.0),
        Candidate(itemset=((0, 0), (1, 1)), score=0.5, guessed_count=1.2),
    ]

    selected = client.report_keys(
        candidates,
        MIXED_ITEMSET_ALPHA,
        report_key_limit=3,
        noisy_singleton_counts={
            (0, 1): 3.0,
            (1, 1): 2.4,
            (0, 0): 1.8,
            (1, 0): 1.5,
        },
    )

    # The product guess for {0=1,1=1} is 3*2.4/3=2.4, tying {1=1}.
    # Joint keys compete directly with items rather than filling only leftover slots.
    assert selected == (
        ((0, 1),),
        ((0, 1), (1, 1)),
        ((1, 1),),
    )


def test_no_guess_ablation_modes_have_distinct_local_key_policies():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0, 1),
        local_data=np.asarray([[1, 1], [1, 0], [0, 1], [0, 1]]),
    )
    candidates = [
        Candidate(itemset=((0, 1),), score=1.0, guessed_count=0.0),
        Candidate(itemset=((1, 1),), score=0.9, guessed_count=0.0),
        Candidate(itemset=((0, 1), (1, 1)), score=0.8, guessed_count=0.0),
        Candidate(itemset=((0, 0), (1, 0)), score=0.7, guessed_count=0.0),
    ]
    item_keys = client.report_keys(
        candidates, LOCAL_TOP_SINGLETON_ALPHA, report_key_limit=1
    )
    component_keys = client.report_keys(
        candidates, LOCAL_TOP_ITEMSET_COMPONENT_ALPHA, report_key_limit=1
    )
    assert len(item_keys) == 1
    assert all(len(key) == 1 for key in item_keys)
    assert component_keys == (((1, 1),),)


def test_map_m_requires_a_public_limit():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0,),
        local_data=np.asarray([[1], [0]]),
    )
    candidate = Candidate(itemset=((0, 1),), score=1.0, guessed_count=1.0)

    try:
        client.report_keys(
            [candidate],
            MIXED_ITEMSET_ALPHA,
        )
    except ValueError as error:
        assert "report_key_limit" in str(error)
    else:
        raise AssertionError("top_k 策略没有拒绝缺失的公共键数上限")


def test_map_m_allocates_budget_by_attribute_projection_groups():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0, 1, 2),
        local_data=np.asarray(
            [[0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]]
        ),
    )
    candidates = [
        Candidate(itemset=((0, 0),), score=1.0, guessed_count=3.0),
        Candidate(itemset=((0, 1),), score=0.9, guessed_count=2.8),
        Candidate(itemset=((0, 0), (1, 1)), score=0.8, guessed_count=2.4),
        Candidate(itemset=((0, 1), (1, 1)), score=0.7, guessed_count=2.0),
        Candidate(itemset=((0, 0), (2, 1)), score=0.6, guessed_count=1.8),
        Candidate(itemset=((0, 1), (2, 1)), score=0.5, guessed_count=1.6),
    ]
    oracle = RankOracle.build(n_rows=4, m=8, gamma=1.0, seed=11)

    reports = client.round2_reports(
        candidates,
        MIXED_ITEMSET_ALPHA,
        epsilon=3.0,
        delta=1e-5,
        oracle=oracle,
        seed=4,
        report_key_limit=5,
        noisy_singleton_counts={
            (0, 0): 3.0,
            (0, 1): 2.8,
            (1, 1): 1.0,
            (2, 1): 0.8,
        },
        # v18 ignores the legacy per-key joint-item weight.
        local_joint_budget_weight=99.0,
    )

    assert {report.key for report in reports} == {
        ((0, 0),),
        ((0, 1),),
        ((1, 1),),
        ((2, 1),),
        ((0, 0), (1, 1)),
    }
    # Groups (0,), (1,), (2,), and (0,1) each receive epsilon=0.75. Group (0,)
    # contains two disjoint gender bins, avoiding the per-key allocation of epsilon=0.6.
    assert all(report.epsilon == 0.75 for report in reports)
    assert all(report.delta == pytest.approx(1e-5 / 4) for report in reports)


def test_map_m_can_disable_grouping_and_split_budget_equally_per_key():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0, 1, 2),
        local_data=np.asarray(
            [[0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]]
        ),
    )
    candidates = [
        Candidate(itemset=((0, 0),), score=1.0, guessed_count=3.0),
        Candidate(itemset=((0, 1),), score=0.9, guessed_count=2.8),
        Candidate(itemset=((0, 0), (1, 1)), score=0.8, guessed_count=2.4),
        Candidate(itemset=((0, 1), (1, 1)), score=0.7, guessed_count=2.0),
        Candidate(itemset=((0, 0), (2, 1)), score=0.6, guessed_count=1.8),
        Candidate(itemset=((0, 1), (2, 1)), score=0.5, guessed_count=1.6),
    ]
    oracle = RankOracle.build(n_rows=4, m=8, gamma=1.0, seed=11)

    reports = client.round2_reports(
        candidates,
        MIXED_ITEMSET_ALPHA,
        epsilon=3.0,
        delta=1e-5,
        oracle=oracle,
        seed=4,
        report_key_limit=5,
        noisy_singleton_counts={
            (0, 0): 3.0,
            (0, 1): 2.8,
            (1, 1): 1.0,
            (2, 1): 0.8,
        },
        local_joint_budget_weight=0.0,
    )

    assert len(reports) == 5
    assert all(report.epsilon == pytest.approx(3.0 / 5) for report in reports)
    assert all(report.delta == pytest.approx(1e-5 / 5) for report in reports)


def test_map_m_bin_merges_public_value_buckets_before_generating_alpha():
    class RecordingOracle:
        m = 4

        def __init__(self):
            self.memberships = []

        def private_alpha(self, membership, epsilon, delta, random_seed):
            self.memberships.append(membership.copy())
            return np.ones(self.m), 1.0

    client = VerticalClient(
        client_id="client_0",
        attributes=(0,),
        local_data=np.asarray([[0], [1], [2], [3]]),
        domains=((0, 1, 2, 3),),
    )
    candidates = [
        Candidate(itemset=((0, value),), score=1.0, guessed_count=4.0 - value)
        for value in range(4)
    ]
    oracle = RecordingOracle()

    reports = client.round2_reports(
        candidates,
        MIXED_BINNED_ITEMSET_ALPHA,
        epsilon=2.0,
        delta=1e-5,
        oracle=oracle,
        seed=9,
        report_key_limit=4,
        map_m_bin_count=2,
        map_m_binning=(((0, 1, 2), (3,)),),
    )

    assert [report.key for report in reports] == [((0, 0),), ((0, 1),)]
    assert [membership.tolist() for membership in oracle.memberships] == [
        [True, True, True, False],
        [False, False, False, True],
    ]
    # Both bins share a projection group and reuse epsilon=2 instead of splitting it into 1 each.
    assert all(report.epsilon == 2.0 for report in reports)


def test_map_l_top_keeps_singletons_and_caps_only_joint_keys():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0, 1),
        local_data=np.asarray([[1, 1], [1, 0], [0, 1]]),
    )
    candidates = [
        Candidate(itemset=((0, 1),), score=1.0, guessed_count=3.0),
        Candidate(itemset=((1, 1),), score=0.9, guessed_count=2.4),
        Candidate(itemset=((0, 1), (1, 1)), score=0.8, guessed_count=2.0),
        Candidate(itemset=((0, 0), (1, 1)), score=0.1, guessed_count=1.0),
    ]

    selected = client.report_keys(
        candidates,
        LOCAL_TOP_ITEMSET_ALPHA,
        local_projection_limit=1,
    )

    assert ((0, 1),) in selected
    assert ((1, 1),) in selected
    assert ((0, 1), (1, 1)) in selected
    assert ((0, 0), (1, 1)) not in selected


def test_fm_full_and_other_use_domain_buckets_and_actual_key_budget():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0, 1),
        local_data=np.asarray([[0, 0], [1, 0], [2, 1], [2, 1]]),
    )
    candidates = [
        Candidate(itemset=((0, 1), (1, 0)), score=1.0, guessed_count=2.0)
    ]

    full_keys = set(
        client.report_keys(
            candidates, SINGLETON_ALPHA, estimator=FM_FULL_ESTIMATOR
        )
    )
    other_keys = set(
        client.report_keys(
            candidates, SINGLETON_ALPHA, estimator=FM_OTHER_ESTIMATOR
        )
    )

    assert full_keys == {
        ((0, 0),),
        ((0, 1),),
        ((0, 2),),
        ((1, 0),),
        ((1, 1),),
    }
    assert other_keys == {
        ((0, 1),),
        ((0, FM_OTHER_VALUE),),
        ((1, 0),),
        ((1, FM_OTHER_VALUE),),
    }
    all_domain_targets = [
        Candidate(itemset=((1, 0),), score=1.0, guessed_count=2.0),
        Candidate(itemset=((1, 1),), score=0.9, guessed_count=2.0),
    ]
    assert ((1, FM_OTHER_VALUE),) in client.report_keys(
        all_domain_targets,
        SINGLETON_ALPHA,
        estimator=FM_OTHER_ESTIMATOR,
    )
    target_values = client._target_values(candidates)
    assert client._fm_membership(
        ((0, FM_OTHER_VALUE),), target_values
    ).tolist() == [True, False, True, True]

    oracle = RankOracle.build(n_rows=4, m=8, gamma=1.0, seed=11)
    full_reports = client.round2_reports(
        candidates,
        SINGLETON_ALPHA,
        4.0,
        1e-5,
        oracle,
        seed=4,
        estimator=FM_FULL_ESTIMATOR,
    )
    other_reports = client.round2_reports(
        candidates,
        SINGLETON_ALPHA,
        4.0,
        1e-5,
        oracle,
        seed=4,
        estimator=FM_OTHER_ESTIMATOR,
    )
    assert all(report.epsilon == 0.8 for report in full_reports)
    assert all(report.epsilon == 1.0 for report in other_reports)
    assert not any(report.is_complement for report in full_reports + other_reports)


def test_map_reports_target_positive_membership_only():
    class RecordingOracle:
        def __init__(self):
            self.memberships = []
            self.m = 4

        def private_alpha(self, membership, epsilon, delta, random_seed):
            self.memberships.append(membership.copy())
            return np.ones(4), 1.0

    client = VerticalClient(
        client_id="client_0",
        attributes=(0,),
        local_data=np.asarray([[0], [1], [1], [2]]),
    )
    candidate = Candidate(itemset=((0, 1),), score=1.0, guessed_count=2.0)
    oracle = RecordingOracle()

    reports = client.round2_reports(
        [candidate], SINGLETON_ALPHA, 1.0, 1e-5, oracle, seed=9
    )

    assert oracle.memberships[0].tolist() == [False, True, True, False]
    assert reports[0].key == ((0, 1),)
    assert reports[0].is_complement is False


def test_round1_reports_zero_support_values_from_public_domain():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0,),
        local_data=np.asarray([[0], [0], [1]]),
        domains=((0, 1, 2),),
    )

    report = client.round1_report(epsilon=1.0, seed=9)

    assert set(report.counts) == {(0, 0), (0, 1), (0, 2)}


def test_round1_releases_noisy_n_with_its_own_budget():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0,),
        local_data=np.asarray([[0], [1], [1], [0]]),
        domains=((0, 1),),
    )

    report = client.round1_report(epsilon=1.0, seed=9, n_epsilon=0.25)

    assert report.noisy_n is not None
    assert report.noisy_n != client.n_rows
    assert report.n_epsilon == pytest.approx(0.25)


def test_round1_rejects_nonpositive_noisy_n_budget():
    client = VerticalClient(
        client_id="client_0",
        attributes=(0,),
        local_data=np.asarray([[0], [1]]),
        domains=((0, 1),),
    )

    with pytest.raises(ValueError, match="n_epsilon"):
        client.round1_report(epsilon=1.0, seed=9, n_epsilon=0.0)
