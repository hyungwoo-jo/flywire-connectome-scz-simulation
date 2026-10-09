import importlib.util
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse

spec=importlib.util.spec_from_file_location('p37',Path(__file__).resolve().parents[1]/'pipelines/37_kc_recurrence.py')
p=importlib.util.module_from_spec(spec);sys.modules['p37']=p;spec.loader.exec_module(p)


class RecurrenceTests(unittest.TestCase):
    def setUp(self):
        self.c=dict(inh=np.ones(3),drive=np.ones(3)/3)
        self.E=np.array([[.8],[.2],[.5]])

    def test_zero_gain_matches_base_solver(self):
        c=dict(self.c,kk=(sparse.csr_matrix(np.ones((3,3))-np.eye(3)),0.))
        k0,_=p.base.response(self.E,c['inh'],c['drive'],.3,1.,gain=p.cal.APL_GAIN);k1,_=p.respond(self.E,c,.3)
        np.testing.assert_allclose(k0,k1)

    def test_excitatory_recurrence_does_not_reduce_activity_and_converges(self):
        W=sparse.csr_matrix(np.ones((3,3))-np.eye(3))
        k0,_=p.respond(self.E,dict(self.c),.3);c=dict(self.c,kk=(W,.3));k1,_=p.respond(self.E,c,.3)
        self.assertTrue(c['_converged']);self.assertGreaterEqual(k1.sum(),k0.sum()-1e-9)


if __name__=='__main__':unittest.main()
