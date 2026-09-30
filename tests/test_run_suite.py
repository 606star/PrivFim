from pathlib import Path

from experiments.run_suite import _config_for, build_cases
from privfim.types import (
    MIXED_BINNED_ITEMSET_ALPHA,
    MIXED_ITEMSET_ALPHA,
    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
    SINGLETON_ALPHA,
)


def test_dense_only_cases_cover_only_new_continuous_sweep_points():
    axes = {"k", "epsilon", "m", "candidate_pool", "budget_split"}
    cases = build_cases([3031], axes, dense_only=True)

    assert len(cases) == 16
    assert {case.axes[0] for case in cases} == axes
    assert {case.k for case in cases if case.axes == ("k",)} == {3, 8, 12, 18, 30}
    assert {case.epsilon for case in cases if case.axes == ("epsilon",)} == {
        0.125,
        0.75,
        1.5,
    }
    assert {case.m for case in cases if case.axes == ("m",)} == {128, 4096}


def test_controlled_tail_and_domain_axes_hold_other_generator_parameters_fixed():
    cases = build_cases([3031], {"tail", "domain"})
    tail = [case.dataset for case in cases if case.axes == ("tail",)]
    domain = [case.dataset for case in cases if case.axes == ("domain",)]

    assert {spec.zipf_exponent for spec in tail} == {0.0, 0.6, 1.2, 1.8}
    assert {spec.domain_size for spec in tail} == {8}
    assert {spec.correlation for spec in tail} == {0.75}
    assert {spec.domain_size for spec in domain} == {4, 8, 16, 32}
    assert {spec.zipf_exponent for spec in domain} == {1.2}
    assert {spec.correlation for spec in domain} == {0.75}


def test_binned_map_mode_is_configured_with_a_bin_count():
    case = next(
        case
        for case in build_cases([3031], {"dataset"})
        if case.dataset.name == "Adult"
    )
    modes = (
        SINGLETON_ALPHA,
        MIXED_ITEMSET_ALPHA,
        MIXED_BINNED_ITEMSET_ALPHA,
        MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
    )
    config = _config_for(
        case,
        Path("results/test"),
        map_modes=modes,
        map_m_bin_count=4,
    )

    assert config.protocol.modes == modes
    assert config.protocol.map_m_bin_count == 4
