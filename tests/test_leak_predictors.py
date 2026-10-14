import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p59',Path(__file__).resolve().parents[1]/'pipelines/59_leak_predictors.py')
p=importlib.util.module_from_spec(spec);sys.modules['p59']=p;spec.loader.exec_module(p)


class PredictorTests(unittest.TestCase):
    def test_loio_exact_line_has_zero_error(self):
        x=np.array([1.,2,3,4,5]);y=2+3*x
        self.assertAlmostEqual(p.loio_mae(x,y,['a','a','b','b','c']),0.,places=8)

    def test_candidates(self):
        w=np.array([0,1,1,2.]);s=np.array([1,2,3,4.]);mn=np.array([0,0,1,3.])
        c=p.candidates(w,s,mn,5.)
        self.assertAlmostEqual(c['cv_spont_read'],np.std([2,3,4])/3)
        self.assertEqual(c['top10'],.75);self.assertEqual(c['H'],5.)

    def test_seed_zero_is_baseline(self):
        self.assertEqual(p.seeds(0),p.BASE);self.assertNotEqual(p.seeds(1)['noise'],p.BASE['noise'])


if __name__=='__main__':unittest.main()
