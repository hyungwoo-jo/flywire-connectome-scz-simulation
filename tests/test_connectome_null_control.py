"""Check null-model constraints rather than expected biological conclusions."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "pipelines/05_connectome_null_control.py"
SPEC = importlib.util.spec_from_file_location("null_control", SCRIPT)
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


class NullConstraints(unittest.TestCase):
    def test_degrees_weights_and_self_loops_survive_rewiring(self):
        n = 40
        rng = np.random.default_rng(5)
        pre = np.repeat(np.arange(n), 5)
        post = np.concatenate([
            rng.choice(np.delete(np.arange(n), i), 5, replace=False)
            for i in range(n)
        ])
        # Existing self-loops are retained; new ones must not be introduced.
        post[0] = 0
        weights = rng.uniform(.1, 3, len(pre)) * np.where(pre % 3 == 0, -1, 1)
        original = np.zeros((n, n))
        original[pre, post] = weights
        for seed in (99, 100, 101):
            with self.subTest(seed=seed):
                target, info = CONTROL.rewire(pre, post, n, seed, 10)
                shuffled = np.zeros((n, n))
                shuffled[pre, target] = weights
                np.testing.assert_array_equal((original != 0).sum(0), (shuffled != 0).sum(0))
                np.testing.assert_array_equal((original != 0).sum(1), (shuffled != 0).sum(1))
                np.testing.assert_allclose(original.sum(1), shuffled.sum(1))
                np.testing.assert_allclose(np.abs(original).sum(1), np.abs(shuffled).sum(1))
                np.testing.assert_array_equal((original < 0).sum(1), (shuffled < 0).sum(1))
                np.testing.assert_array_equal(np.diag(original), np.diag(shuffled))
                self.assertEqual(len(set(zip(pre, target))), len(pre))
                self.assertEqual(info["successful_swaps"], 10 * np.sum(pre != post))
                self.assertGreater(info["changed_edge_target_fraction"], .8)


if __name__ == "__main__":
    unittest.main()
