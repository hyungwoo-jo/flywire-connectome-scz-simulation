import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p24',Path(__file__).resolve().parents[1]/'pipelines/24_excitability_expectation.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class IdentifyTests(unittest.TestCase):
    def test_cosine_identification_is_scale_invariant(self):
        t=np.array([[1.,0.],[0.,1.],[1.,1.]]);t=t/np.linalg.norm(t,axis=0)
        k=np.array([[5.,0.],[0.,.1],[5.,.1]])
        np.testing.assert_array_equal(p.identify(k,t),[0,1])


if __name__=='__main__':unittest.main()
