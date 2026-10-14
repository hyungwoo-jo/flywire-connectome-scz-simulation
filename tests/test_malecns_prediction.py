import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p58',Path(__file__).resolve().parents[1]/'pipelines/58_malecns_prediction.py')
p=importlib.util.module_from_spec(spec);sys.modules['p58']=p;spec.loader.exec_module(p)


class PredictionTests(unittest.TestCase):
    def test_rules_match_reference_fit(self):
        for k,(a,b) in p.RULES.items():
            bb,aa=np.polyfit(p.CV_REF,p.REF[k],1)
            self.assertAlmostEqual(a,aa,places=2);self.assertAlmostEqual(b,bb,places=2)

    def test_interval_brackets_point_and_widens_away_from_mean(self):
        mid=p.prediction('specific',float(p.CV_REF.mean()));far=p.prediction('specific',1.1)
        self.assertLess(mid[1],mid[0]);self.assertLess(mid[0],mid[2])
        self.assertGreater(far[2]-far[1],mid[2]-mid[1])
        self.assertAlmostEqual(p.prediction('specific',.80)[0],2.67,places=2)


if __name__=='__main__':unittest.main()
