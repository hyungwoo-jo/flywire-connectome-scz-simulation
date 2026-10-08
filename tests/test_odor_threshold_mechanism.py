import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('mechanism',Path(__file__).resolve().parents[1]/'pipelines/15_odor_threshold_mechanism.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ThresholdMechanismTests(unittest.TestCase):
    def test_errors_partition_without_conditioning_denominator(self):
        result=m.error_components(np.array([0.,1.,.5,1.]),np.array([True,False,False,False]))
        self.assertEqual(result['accuracy'],.625)
        self.assertEqual(result['silent_error_mass'],.25)
        self.assertEqual(result['active_error_mass'],.125)
        self.assertAlmostEqual(result['accuracy_if_active'],5/6)
        self.assertAlmostEqual(1-result['accuracy'],result['silent_error_mass']+result['active_error_mass'])

    def test_silent_centroids_receive_chance_score(self):
        labels=np.repeat(np.arange(3),4);train=np.tile([True,True,False,False],3)
        scores,silent=m.trial_scores(np.zeros((4,12)),labels,train)
        np.testing.assert_allclose(scores,1/3)
        self.assertTrue(silent.all())
        self.assertIsNone(m.error_components(scores,silent)['accuracy_if_active'])

    def test_gain_threshold_scaling_before_saturation(self):
        e=np.array([[.2,.3],[.15,.35]])
        inh=np.ones(2);drive=np.ones(2)/2
        low,_=m.base.response(e,inh,drive,.1/4,1.)
        high,_=m.base.response(4*e,inh,drive,.1,1.)
        self.assertLess(high.max(),1.)
        np.testing.assert_allclose(high,4*low,atol=1e-9)

    def test_zero_bound_uses_class_imbalance_without_training_predictor(self):
        labels=np.array([0,0,0,1,2,2]);zero=np.array([True,True,True,True,False,False])
        self.assertAlmostEqual(m.zero_error_bound(zero,labels),1/6)
        self.assertEqual(m.zero_error_bound(np.zeros(6,dtype=bool),labels),0.)

    def test_apl_changes_magnitude_without_changing_any_response(self):
        e=np.array([[.05,.2,.8],[.08,.3,.7]])
        inh=np.ones(2);drive=np.ones(2)/2
        absent,_=m.base.response(e,inh,drive,.1,0.)
        normal,_=m.base.response(e,inh,drive,.1,1.)
        np.testing.assert_array_equal(np.max(normal,axis=0)>1e-8,[False,True,True])
        np.testing.assert_array_equal(np.max(normal,axis=0)>1e-8,np.max(absent,axis=0)>1e-8)
        self.assertLess(normal.mean(),absent.mean())
