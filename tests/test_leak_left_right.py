import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p61',Path(__file__).resolve().parents[1]/'pipelines/61_leak_left_right.py')
p=importlib.util.module_from_spec(spec);sys.modules['p61']=p;spec.loader.exec_module(p)


class SwapTests(unittest.TestCase):
    def test_parts_sum_to_difference(self):
        rng=np.random.default_rng(0)
        mk=lambda:dict(crit=rng.random(4)*.5,le=[rng.random(rng.integers(5,20)) for _ in range(4)],n_trials=100)
        tl,tr=mk(),mk();s=p.swap_parts(tl,tr)
        self.assertAlmostEqual(s['crit_part']+s['noise_part'],s['LL']-s['RR'])

    def test_weights(self):
        w=np.array([0,1,2,3.]);rng=np.random.default_rng(1)
        u=p.weights(w,'uniform',rng);self.assertEqual(u[0],0);self.assertTrue(np.allclose(u[1:],1/3))
        s=p.weights(w,'shuffle',rng);self.assertEqual(s[0],0);self.assertAlmostEqual(s.sum(),1.)

    def test_gini(self):
        self.assertAlmostEqual(p.gini(np.ones(5)),0.)


if __name__=='__main__':unittest.main()
