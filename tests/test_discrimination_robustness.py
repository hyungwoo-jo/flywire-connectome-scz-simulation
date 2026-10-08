import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('robustness',Path(__file__).resolve().parents[1]/'pipelines/07_discrimination_robustness.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class StrengthNullTests(unittest.TestCase):
    def test_weighted_input_strengths_and_degrees_are_preserved(self):
        pre=np.repeat(np.arange(6),3)
        post=np.concatenate([np.array([i,i+1,i+3])%9 for i in range(6)])
        weights=np.tile([1.,2.,1.],6)
        for seed in range(3):
            target,info=model.strength_preserving_null(pre,post,weights,np.random.default_rng(seed),multiple=2)
            np.testing.assert_array_equal(np.bincount(post,minlength=9),np.bincount(target,minlength=9))
            np.testing.assert_allclose(np.bincount(post,weights=weights,minlength=9),np.bincount(target,weights=weights,minlength=9))
            self.assertEqual(len(set(zip(pre,target))),len(pre))
            self.assertEqual(info['swaps'],2*len(pre))
            self.assertGreater(info['changed_fraction'],0)

    def test_impossible_equal_weight_null_fails_explicitly(self):
        with self.assertRaises(ValueError):
            model.strength_preserving_null(np.array([0,1]),np.array([0,1]),np.array([1.,2.]),np.random.default_rng(0))


if __name__=='__main__':unittest.main()
