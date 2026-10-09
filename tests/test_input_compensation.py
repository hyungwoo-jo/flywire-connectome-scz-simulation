import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse

spec=importlib.util.spec_from_file_location('p29',Path(__file__).resolve().parents[1]/'pipelines/29_input_compensation.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class InputCompensationTests(unittest.TestCase):
    def test_alpha_endpoints(self):
        m=sparse.csr_matrix(np.array([[1.,3.],[0.,0.],[2.,0.]]))
        np.testing.assert_allclose(p.compensate(m,0.).toarray(),m.toarray())
        e=p.compensate(m,1.).toarray();self.assertAlmostEqual(e[0].sum(),e[2].sum());self.assertEqual(e[1].sum(),0.)

    def test_interior_optimum(self):
        self.assertTrue(p.interior_optimum([3,2,1,2,3],p.ALPHAS,'min')[0])
        self.assertFalse(p.interior_optimum([5,4,3,2,1],p.ALPHAS,'min')[0])


if __name__=='__main__':unittest.main()
