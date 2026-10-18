import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('obs',Path(__file__).resolve().parents[1]/'pipelines/65x_observation_model.py')
p=importlib.util.module_from_spec(spec);sys.modules['obs']=p;spec.loader.exec_module(p)

class ObservationTests(unittest.TestCase):
    def test_lowpass_step_and_impulse(self):
        a=np.exp(-p.DT/.3);x=np.ones((2,12))
        np.testing.assert_allclose(p.lowpass(x,.3),np.tile(1-a**np.arange(1,13),(2,1)))
        impulse=np.zeros((1,12));impulse[0,0]=1
        np.testing.assert_allclose(p.lowpass(impulse,.3)[0],(1-a)*a**np.arange(12))
        with self.assertRaises(ValueError):p.lowpass(x,0)

    def test_stationary_ar_noise(self):
        x=p.ar_noise((20000,12),.2,.6,np.random.default_rng(30))
        self.assertAlmostEqual(x.std(),.2,delta=.003)
        self.assertAlmostEqual(np.corrcoef(x[:,:-1].ravel(),x[:,1:].ravel())[0,1],.6,delta=.01)

    def test_same_event_detector_rejects_isolated_frame(self):
        base=np.tile(np.array([-1.,0.,1.,0.]),(10,1,6))
        base[0,0,10:12]=20
        pos,neg=p.p65.event_rates_nan(base,3.)
        self.assertEqual(pos.item(),.1)
        self.assertEqual(neg.item(),0)
        base[0,0,11]=0
        pos,_=p.p65.event_rates_nan(base,3.)
        self.assertEqual(pos.item(),0)
