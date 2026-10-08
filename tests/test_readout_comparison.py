import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('readout',Path(__file__).resolve().parents[1]/'pipelines/10_readout_comparison.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class ReadoutTests(unittest.TestCase):
    def test_normalized_scores_ignore_uniform_neural_scaling(self):
        before=np.array([[.4,.6]]);after=np.array([[.1,.5]]);k=np.array([[.2,.5],[.7,.3]])
        one=model.read_scores(before,after,k);scaled=model.read_scores(before,after,k*.2)
        np.testing.assert_allclose(scaled['raw'],one['raw']*.2)
        np.testing.assert_allclose(scaled['activity_normalized'],one['activity_normalized'])
        np.testing.assert_allclose(scaled['counterfactual_fraction'],one['counterfactual_fraction'])

    def test_fraction_is_zero_without_learning_and_at_zero_input(self):
        w=np.array([[.4,.6]]);k=np.array([[0.,.5],[0.,.3]])
        scores=model.read_scores(w,w,k)
        np.testing.assert_array_equal(scores['counterfactual_fraction'],np.zeros(2))
        self.assertTrue(all(np.isfinite(s).all() for s in scores.values()))
        self.assertTrue(all(s[0]==0 for s in scores.values()))

    def test_fraction_distinguishes_depressed_and_unchanged_inputs(self):
        scores=model.read_scores(np.array([[.5,.5]]),np.array([[.1,.5]]),np.eye(2))
        np.testing.assert_allclose(scores['counterfactual_fraction'],[.8,0.])


if __name__=='__main__':unittest.main()
