import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse

spec=importlib.util.spec_from_file_location('p26',Path(__file__).resolve().parents[1]/'pipelines/26_hub_normalization.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class HubTests(unittest.TestCase):
    def test_equalize_rows_keeps_ratios_and_zero_rows(self):
        m=sparse.csr_matrix(np.array([[1.,3.],[0.,0.],[2.,0.]]))
        e=p.equalize_rows(m).toarray()
        self.assertAlmostEqual(e[0].sum(),e[2].sum());self.assertEqual(e[1].sum(),0.)
        self.assertAlmostEqual(e[0,1]/e[0,0],3.)

    def test_gini(self):
        self.assertAlmostEqual(p.gini([1,1,1,1]),0.)
        self.assertGreater(p.gini([0,0,0,10]),.7)


if __name__=='__main__':unittest.main()
