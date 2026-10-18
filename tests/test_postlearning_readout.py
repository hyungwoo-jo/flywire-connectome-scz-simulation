import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('readout',Path(__file__).resolve().parents[1]/'pipelines/67_postlearning_readout.py')
p=importlib.util.module_from_spec(spec);sys.modules['readout']=p;spec.loader.exec_module(p)

class PostlearningTests(unittest.TestCase):
    def test_threshold_fits_known_order_without_test_labels(self):
        scores=np.array([.1,.2,.8,.9]);labels=np.array([False,False,True,True])
        threshold=p.fit_threshold(scores,labels)
        np.testing.assert_array_equal(scores>threshold,labels)
        self.assertGreater(threshold,.2);self.assertLess(threshold,.8)
        with self.assertRaises(ValueError):p.fit_threshold(scores,np.ones(4,dtype=bool))

    def test_shuffle_preserves_weight_drop_and_subtype_constraints(self):
        profile=np.array([.1,.2,.3,.4,.5,.7]);weights=np.array([1,1,2,2,1,1])/8
        types=np.array(['a','a','a','a','b','b'])
        shuffled,eligible=p.matched_profile(profile,weights,types,np.random.default_rng(7))
        self.assertAlmostEqual(eligible,1.)
        self.assertAlmostEqual(weights@np.exp(-p.p25.ETA*profile),weights@np.exp(-p.p25.ETA*shuffled))
        np.testing.assert_array_equal(np.sort(shuffled[:2]),np.sort(profile[:2]))
        np.testing.assert_array_equal(np.sort(shuffled[2:4]),np.sort(profile[2:4]))
        np.testing.assert_array_equal(np.sort(shuffled[4:]),np.sort(profile[4:]))

    def test_actual_scores_use_only_current_output_and_declared_channel(self):
        result=p.actual_scores(np.array([.2,.4]),np.array([1.,2.]))
        np.testing.assert_allclose(result['raw_after'],[-.2,-.4])
        np.testing.assert_allclose(result['pn_normalized_after'],[-.2,-.2])
        self.assertEqual(set(result),{'raw_after','pn_normalized_after'})
        np.testing.assert_array_equal(p.privileged_score(np.zeros(2),np.zeros(2)),np.zeros(2))

    def test_auc_ties_and_perfect_separation(self):
        self.assertEqual(p.auc(np.ones(5),np.ones(3)),.5)
        self.assertEqual(p.auc(np.array([2.,3.]),np.array([0.,1.])),1.)

    def test_no_learning_fraction_ignores_roundoff(self):
        before=np.array([.1,.4,1e-9])
        after=np.nextafter(before,np.zeros(3))
        np.testing.assert_array_equal(p.privileged_score(before,after),np.zeros(3))
