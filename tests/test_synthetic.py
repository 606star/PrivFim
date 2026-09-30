import numpy as np

from experiments.generate_synthetic import SyntheticSpec, generate, validate


def test_synthetic_generator_controls_within_group_dependence():
    independent_spec = SyntheticSpec("ind", 4000, 8, 4, 4, 0.0, 0.0, 3)
    correlated_spec = SyntheticSpec("corr", 4000, 8, 4, 4, 0.9, 0.0, 4)
    independent = validate(generate(independent_spec), independent_spec)
    correlated = validate(generate(correlated_spec), correlated_spec)

    assert independent["shape"] == [4000, 8]
    assert correlated["mean_within_group_nmi"] > independent["mean_within_group_nmi"]
    assert correlated["mean_within_group_nmi"] > correlated["mean_across_group_nmi"]


def test_layout_changes_only_column_placement():
    grouped = SyntheticSpec("g", 500, 12, 5, 3, 0.7, 1.1, 9, "grouped")
    interleaved = SyntheticSpec(
        "i", 500, 12, 5, 3, 0.7, 1.1, 9, "interleaved"
    )
    grouped_data = generate(grouped)
    interleaved_data = generate(interleaved)
    expected_order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

    assert np.array_equal(interleaved_data, grouped_data[:, expected_order])
    grouped_validation = validate(grouped_data, grouped)
    interleaved_validation = validate(interleaved_data, interleaved)
    assert np.isclose(
        grouped_validation["mean_within_group_nmi"],
        interleaved_validation["mean_within_group_nmi"],
    )
