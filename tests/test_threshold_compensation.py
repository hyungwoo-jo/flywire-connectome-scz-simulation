import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p27',Path(__file__).resolve().parents[1]/'pipelines/27_threshold_compensation.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class CompensationTests(unittest.TestCase):
    def test_homeostatic_thresholds_quantile_and_inputless(self):
        e=np.vstack([np.arange(10.),np.zeros(10),2*np.arange(10.)])
        th=p.homeostatic_thresholds(e)
        self.assertAlmostEqual(th[0],np.percentile(np.arange(10.),90))
        self.assertAlmostEqual(th[2],2*th[0]);self.assertAlmostEqual(th[1],(th[0]+th[2])/2)


if __name__=='__main__':unittest.main()
