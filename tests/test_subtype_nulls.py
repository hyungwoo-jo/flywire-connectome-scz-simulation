import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('subtype',Path(__file__).resolve().parents[1]/'pipelines/12_subtype_nulls.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class SubtypeNullTests(unittest.TestCase):
    def test_subtype_input_counts_and_strengths_survive_swaps(self):
        pre=np.repeat(np.arange(6),3);post=np.concatenate([np.array([i,i+1,i+3])%9 for i in range(6)])
        weights=np.tile([1.,2.,1.],6);types=np.array(['a','a','a','b','b','b'])
        for seed in range(3):
            target,_=model.subtype_null(pre,post,weights,types,np.random.default_rng(seed),multiple=2)
            self.assertEqual(len(set(zip(pre,target))),len(pre))
            for subtype in ('a','b'):
                mask=types[pre]==subtype
                np.testing.assert_array_equal(np.bincount(post[mask],minlength=9),np.bincount(target[mask],minlength=9))
                np.testing.assert_allclose(np.bincount(post[mask],weights=weights[mask],minlength=9),np.bincount(target[mask],weights=weights[mask],minlength=9))

    def test_unmixable_type_is_preserved_and_reported(self):
        pre=np.array([0,0]);post=np.array([0,1]);weights=np.array([1.,2.]);types=np.array(['single'])
        target,info=model.subtype_null(pre,post,weights,types,np.random.default_rng(1))
        np.testing.assert_array_equal(target,post)
        self.assertEqual(info['groups'][0]['swaps'],0)
        self.assertEqual(info['changed_fraction'],0)


if __name__=='__main__':unittest.main()
