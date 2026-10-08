import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('calibration',Path(__file__).resolve().parents[1]/'pipelines/11_readout_calibration.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class CalibrationTests(unittest.TestCase):
    def test_context_recovers_known_offset_without_test_labels(self):
        feature=np.repeat([0.,10.,20.],4)
        scores=np.tile([2.,3.,0.,1.],3)+feature
        labels=np.tile([0,0,1,1],3)
        fit=model.fit_calibration(scores,labels,feature,context=True)
        np.testing.assert_array_equal(model.margins(scores,feature,fit)>0,labels==0)
        result=model.margins(np.array([2.,22.]),np.array([-100.,100.]),fit)
        self.assertTrue(np.isfinite(result).all())

    def test_one_class_bins_use_global_fallback(self):
        scores=np.arange(12,dtype=float);feature=scores.copy();labels=np.r_[np.zeros(6,dtype=int),np.ones(6,dtype=int)]
        fit=model.fit_calibration(scores,labels,feature,context=True)
        fallback=model.learning.decision_threshold(scores,labels)
        self.assertEqual(fit['thresholds'][0],fallback)
        self.assertEqual(fit['thresholds'][2],fallback)

    def test_global_threshold_does_not_change_auc(self):
        scores=np.array([.2,.8,.4,.1]);labels=np.array([0,0,1,1]);feature=np.ones(4)
        fit=model.fit_calibration(scores,labels,feature)
        self.assertEqual(model.learning.auc(scores,labels),model.learning.auc(model.margins(scores,feature,fit),labels))


if __name__=='__main__':unittest.main()
