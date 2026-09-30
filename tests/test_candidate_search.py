import unittest
import numpy as np

from privfim.candidate import construct_svsm_candidates
from privfim.candidate_search import search_candidates


class CandidateSearchTest(unittest.TestCase):
    def test_matches_exhaustive_with_ties_conflicts_and_clipping(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            items = [((a, v), float(rng.choice([-5, 0, 20, 50, 100, 110])))
                     for a in range(5) for v in range(2)]
            rng.shuffle(items)
            for minimum in (1, 2):
                for limit in (1, 7, 30):
                    expected = construct_svsm_candidates(items, 100, limit, minimum, 4)
                    for pruning in (False, True):
                        actual, _ = search_candidates(items, 100, limit, minimum, 4, pruning=pruning)
                        self.assertEqual(actual, expected)

    def test_pruning_reduces_search_without_changing_output(self):
        items = [((a, 0), 100. - a * 3) for a in range(20)]
        full, fs = search_candidates(items, 100, 30, pruning=False)
        pruned, ps = search_candidates(items, 100, 30, pruning=True)
        self.assertEqual(full, pruned)
        self.assertLess(ps['visited_nodes'], fs['visited_nodes'])


if __name__ == '__main__':
    unittest.main()
