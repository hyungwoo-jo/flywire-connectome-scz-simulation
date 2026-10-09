import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p25',Path(__file__).resolve().parents[1]/'pipelines/25_learned_value_leakage.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class ValueSignalTests(unittest.TestCase):
    def test_value_signal_only_on_learned_kcs(self):
        w=np.array([.5,.5]);learn=np.array([1.,0.])
        k=np.array([[1.,0.,0.],[0.,1.,0.]])
        L=p.value_signal(w,learn,k)
        self.assertAlmostEqual(L[0],1-np.exp(-p.ETA));self.assertEqual(L[1],0.);self.assertEqual(L[2],0.)


if __name__=='__main__':unittest.main()
