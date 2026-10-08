import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('odor',Path(__file__).resolve().parents[1]/'pipelines/14_observed_odor_geometry.py')
odor=importlib.util.module_from_spec(spec);spec.loader.exec_module(odor)

class OdorGeometryTests(unittest.TestCase):
    def test_silent_readout_does_not_invent_identity(self):
        labels=np.repeat(np.arange(3),4);train=np.tile([True,True,False,False],3)
        result=odor.classify(np.zeros((5,12)),labels,train)
        self.assertAlmostEqual(result['accuracy'],1/3)
        self.assertEqual(result['tied_test_fraction'],1.)
        self.assertEqual(result['silent_trial_fraction'],1.)

    def test_known_separable_multiclass(self):
        labels=np.repeat(np.arange(3),4);train=np.tile([True,True,False,False],3)
        self.assertEqual(odor.classify(np.eye(3)[:,labels],labels,train)['accuracy'],1.)

    def test_baseline_subtraction_explicitly_loses_inhibition(self):
        values=np.array([[.1,.5],[.4,.1]]);sfr=np.array([.2,.2])
        np.testing.assert_allclose(odor.transfer(values,sfr,'positive_delta'),[[0,.3],[.2,0]])
        np.testing.assert_array_equal(odor.transfer(values,sfr,'absolute'),values)
        np.testing.assert_array_equal(odor.transfer(sfr[:,None],sfr,'positive_delta'),np.zeros((2,1)))
