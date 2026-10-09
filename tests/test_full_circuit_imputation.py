import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('p28',Path(__file__).resolve().parents[1]/'pipelines/28_full_circuit_imputation.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class ImputationTests(unittest.TestCase):
    def test_lateral_sum_uses_real_channels_only(self):
        orn=np.array([[50.],[0.],[1000.]])
        a=p.olsen_partial_sum(orn,2);b=p.olsen_partial_sum(orn[:2],2)
        self.assertAlmostEqual(a[0,0],b[0,0])

    def test_imputed_channels_one_per_unmapped_type_preserve_values(self):
        pns=pd.DataFrame(dict(glomerulus=['A','A','B','C'],receptor=['Or1','Or1','','']))
        delta=np.arange(12.).reshape(3,4);sfr=np.array([1.,2.,3.])
        types,rows,s=p.imputed_channels(pns,delta,sfr,np.random.default_rng(0))
        self.assertEqual(types,['B','C']);self.assertEqual(rows.shape,(2,4))
        for r in rows:self.assertTrue(any(np.array_equal(np.sort(r),np.sort(d)) for d in delta))


if __name__=='__main__':unittest.main()
