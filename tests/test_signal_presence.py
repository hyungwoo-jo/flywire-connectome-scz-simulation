import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('presence',Path(__file__).resolve().parents[1]/'pipelines/09_signal_presence.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class PresenceTests(unittest.TestCase):
    def test_pattern_identity_matches_training_generator(self):
        x,y,_=model.base.patterns(138,.9,0.,40201,80)
        bases=model.pattern_bases(138,.9,40201)
        np.testing.assert_array_equal(x,bases[:,y])

    def test_independent_blank_false_detection_is_near_calibration_tail(self):
        threshold=model.presence_threshold(.4,32,701)
        blank=np.maximum(0,np.random.default_rng(702).normal(0,.4,(32,5000)))
        fraction=np.mean(blank.sum(axis=0)>threshold)
        self.assertGreater(fraction,.025)
        self.assertLess(fraction,.075)
        self.assertGreater(threshold,0)
        self.assertFalse(np.zeros(32).sum()>threshold)


if __name__=='__main__':unittest.main()
