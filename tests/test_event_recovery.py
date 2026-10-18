import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('recovery',Path(__file__).resolve().parents[1]/'pipelines/65y_event_recovery.py')
p=importlib.util.module_from_spec(spec);sys.modules['recovery']=p;spec.loader.exec_module(p)

class RecoveryTests(unittest.TestCase):
    def test_vector_detector_matches_reference(self):
        x=np.random.default_rng(10).random((5,7,20))>.5
        np.testing.assert_array_equal(p.count_two_runs(x),p.p.p64.runs(x,2))

    def test_zero_amplitude_preserves_trace(self):
        base=np.random.default_rng(4).normal(size=(20,10,20));B=np.arange(10)
        changed,event,_,prob=p.inject(base,B,.02,0,np.random.default_rng(8))
        np.testing.assert_array_equal(changed,base)
        self.assertTrue(np.all(np.diff(prob)>0))
        self.assertAlmostEqual(prob.mean(),.02)

    def test_injection_respects_trial_boundaries_and_missing_data(self):
        base=np.random.default_rng(4).normal(size=(30,10,20));base[0,0,0]=np.nan
        changed,event,start,_=p.inject(base,np.zeros(10),.5,8,np.random.default_rng(9))
        self.assertFalse(event[0,0])
        self.assertTrue(np.all(start+8<=20))
        for trial,cell in zip(*np.where(event)):
            self.assertGreater(changed[trial,cell,start[trial,cell]],base[trial,cell,start[trial,cell]])
