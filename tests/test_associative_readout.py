import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('learning',Path(__file__).resolve().parents[1]/'pipelines/08_associative_readout.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class LearningTests(unittest.TestCase):
    def test_gate_off_preserves_weights_and_anatomical_zeros(self):
        w=np.array([[.3,0.,.7],[0.,1.,0.]])
        np.testing.assert_array_equal(model.learn(w,np.ones(3),20,gate=0),w)
        after=model.learn(w,np.array([1.,.5,0.]),5)
        self.assertTrue(np.all(after[w==0]==0))
        self.assertTrue(np.all(after>=0))
        self.assertTrue(np.all(after<=w))

    def test_specific_training_changes_only_active_synapse(self):
        w=np.array([[.5,.5]])
        after=model.learn(w,np.array([1.,0.]),2)
        np.testing.assert_allclose(after,np.array([[.5*np.exp(-2),.5]]))
        self.assertLess((after@np.array([1.,0.])).item(),(w@np.array([1.,0.])).item())
        self.assertEqual((after@np.array([0.,1.])).item(),(w@np.array([0.,1.])).item())

    def test_auc_ties_and_reversed_association(self):
        labels=np.array([0,0,1,1])
        self.assertEqual(model.auc(np.ones(4),labels),.5)
        self.assertEqual(model.auc(np.array([3.,2.,1.,0.]),labels),1.)
        self.assertEqual(model.auc(np.array([0.,1.,2.,3.]),labels),0.)

    def test_training_threshold_separates_known_scores(self):
        s=np.array([-.1,-.2,-.7,-.8]);y=np.array([0,0,1,1])
        t=model.decision_threshold(s,y)
        np.testing.assert_array_equal(s>t,y==0)
        self.assertTrue(0>t)

    def test_negative_learning_rate_is_rejected(self):
        with self.assertRaises(ValueError):model.learn(np.ones((1,1)),np.ones(1),-1)


if __name__=='__main__':unittest.main()
