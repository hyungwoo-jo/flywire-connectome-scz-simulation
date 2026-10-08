import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p23',Path(__file__).resolve().parents[1]/'pipelines/23_detection_false_alarms.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class DetectionTests(unittest.TestCase):
    def test_auc_extremes_and_ties(self):
        self.assertEqual(p.auc(np.array([2.,3.]),np.array([0.,1.])),1.)
        self.assertEqual(p.auc(np.array([0.,0.]),np.array([0.,0.])),.5)

    def test_sdt_clipping_and_symmetry(self):
        d,c=p.sdt(.5,.5,100,100);self.assertAlmostEqual(d,0.);self.assertAlmostEqual(c,0.)
        d,_=p.sdt(1.,0.,10,10);self.assertTrue(np.isfinite(d));self.assertGreater(d,0)


if __name__=='__main__':unittest.main()
