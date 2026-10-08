import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p19',Path(__file__).resolve().parents[1]/'pipelines/19_deficit_cause.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class SubtypeTargetNullTests(unittest.TestCase):
    def test_preserves_target_subtype_degree_and_strength(self):
        rng=np.random.default_rng(3)
        pre=np.repeat(np.arange(6),8);post=np.concatenate([rng.choice(20,8,replace=False) for _ in range(6)])
        w=np.ones(len(pre));kc_type=np.array(['g']*10+['ab']*10)
        t=p.subtype_target_null(pre,post,w,kc_type,np.random.default_rng(1))
        np.testing.assert_array_equal(kc_type[t],kc_type[post])
        np.testing.assert_array_equal(np.bincount(t,minlength=20),np.bincount(post,minlength=20))
        self.assertEqual(len(set(zip(pre,t))),len(pre))
        self.assertGreater(np.mean(t!=post),0)

    def test_centered_spearman_removes_group_offset(self):
        kind=np.array(['a']*20+['b']*20);s=np.r_[np.zeros(20),np.ones(20)]+np.random.default_rng(0).normal(0,1e-3,40)
        d=np.r_[np.zeros(20),np.ones(20)]+np.random.default_rng(1).normal(0,1,40)
        p.PERMUTATIONS=200
        rho,_=p.centered_spearman(s,d,kind,np.random.default_rng(2))
        self.assertLess(abs(rho),.5)


if __name__=='__main__':unittest.main()
