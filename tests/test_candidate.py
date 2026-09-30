from privfim.candidate import (
    construct_itemset_first_candidates,
    construct_svsm_candidates,
)


def test_svsm_guess_uses_normalized_product_and_count_formula():
    candidates = construct_svsm_candidates(
        frequent_items=[((0, 1), 80.0), ((1, 2), 50.0), ((0, 0), 20.0)],
        n_rows=100,
        candidate_count=5,
        min_size=2,
        max_size=2,
    )

    assert candidates[0].itemset == ((0, 1), (1, 2))
    assert candidates[0].score == 0.9 * (0.9 * 50.0 / 80.0)
    assert candidates[0].guessed_count == 40.0
    assert all(len({item[0] for item in candidate.itemset}) == 2 for candidate in candidates)


def test_singletons_compete_inside_the_unified_candidate_limit():
    candidates = construct_svsm_candidates(
        frequent_items=[((0, 1), 80.0), ((1, 2), 50.0), ((2, 3), 40.0)],
        n_rows=100,
        candidate_count=2,
        min_size=1,
        max_size=3,
    )

    assert len(candidates) == 2
    assert candidates == sorted(
        candidates, key=lambda candidate: (-candidate.score, candidate.itemset)
    )
    assert all(len(candidate.itemset) == 1 for candidate in candidates)


def test_itemset_first_keeps_a_singleton_and_prioritizes_cross_items():
    candidates = construct_itemset_first_candidates(
        frequent_items=[((0, 1), 80.0), ((1, 2), 70.0), ((2, 3), 60.0)],
        n_rows=100,
        candidate_count=3,
        min_size=1,
        max_size=2,
    )

    assert len(candidates) == 3
    assert any(len(candidate.itemset) == 1 for candidate in candidates)
    assert sum(len(candidate.itemset) > 1 for candidate in candidates) == 2
