import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('p20',Path(__file__).resolve().parents[1]/'pipelines/20_label_permutation.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class LabelPermutationTests(unittest.TestCase):
    def test_permutation_stays_within_pn_count_class_and_is_not_identity(self):
        gl=['A','B','C','D','E'];count={'A':1,'B':1,'C':2,'D':2,'E':3}
        for seed in range(30):
            m=p.class_permutation(gl,count,np.random.default_rng(seed))
            self.assertEqual(sorted(m.values()),gl)
            self.assertTrue(all(count[a]==count[b] for a,b in m.items()))
            self.assertTrue(any(a!=b for a,b in m.items()))
            self.assertEqual(m['E'],'E')

    def test_receptor_rows_follow_mapping(self):
        mapped=pd.DataFrame(dict(glomerulus=['A','A','B'],receptor=['Or1','Or1','Or2']))
        rec=['Or2','Or1']
        np.testing.assert_array_equal(p.receptor_rows(mapped,rec,{'A':'A','B':'B'}),[1,1,0])
        np.testing.assert_array_equal(p.receptor_rows(mapped,rec,{'A':'B','B':'A'}),[0,0,1])

    def test_mantel_detects_identical_matrices(self):
        rng=np.random.default_rng(0);x=rng.normal(size=(8,8));s=(x+x.T)/2
        rho,pv=p.mantel(s,s,np.random.default_rng(1),500)
        self.assertAlmostEqual(rho,1.);self.assertLess(pv,.05)


if __name__=='__main__':unittest.main()
