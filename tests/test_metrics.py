from privfim.metrics import evaluate_estimates
from privfim.types import CandidateEstimate


def test_frequency_errors_can_use_a_common_candidate_domain():
    first = ((0, 1), (1, 1))
    second = ((0, 1), (2, 1))
    estimates = [
        CandidateEstimate(first, 90.0, 0.0, (first,)),
        CandidateEstimate(second, 0.0, 0.0, (second,)),
    ]
    truth = {first: 100, second: 80}

    all_errors = evaluate_estimates(estimates, truth, k=1, n_rows=100)
    common_errors = evaluate_estimates(
        estimates,
        truth,
        k=1,
        n_rows=100,
        error_itemsets={first},
    )

    assert all_errors.mae == 45.0
    assert common_errors.mae == 10.0
    assert common_errors.nmae == 0.1


def test_missing_top_k_estimates_reduce_recall_instead_of_shrinking_k():
    first = ((0, 1),)
    second = ((1, 1),)
    third = ((2, 1),)
    estimates = [CandidateEstimate(first, 100.0, 0.0, (first,))]
    truth = {first: 100, second: 90, third: 80}

    metrics = evaluate_estimates(estimates, truth, k=2, n_rows=100)

    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.candidate_recall == 0.5
