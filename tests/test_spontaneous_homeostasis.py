import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p56',Path(__file__).resolve().parents[1]/'pipelines/56_spontaneous_homeostasis.py')
p=importlib.util.module_from_spec(spec);sys.modules['p56']=p;spec.loader.exec_module(p)


class SpontTests(unittest.TestCase):
    def test_z_hits_target_without_inhibition(self):
        rng=np.random.default_rng(0);n=500;E=rng.random((n,40))
        c=dict(inh=np.zeros(n),drive=np.zeros((1,n)))
        z=p.z_for_target(E,c,np.full(n,.2),np.full(n,.1),target=.25)
        frac=np.mean(p.p55.respond(E,c,np.full((n,1),.2+z*.1))>1e-8)
        self.assertAlmostEqual(frac,.25,delta=.01)


if __name__=='__main__':unittest.main()
