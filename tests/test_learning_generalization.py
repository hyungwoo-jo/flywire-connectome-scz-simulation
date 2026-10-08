import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p21',Path(__file__).resolve().parents[1]/'pipelines/21_learning_generalization.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class GeneralizationTests(unittest.TestCase):
    def test_disjoint_odors_do_not_generalize_and_identical_fully_do(self):
        k=np.array([[1.,1.,0.],[0.,0.,1.]]);w=np.ones(2)/2
        gi=p.generalization(k,w,5.)
        self.assertAlmostEqual(gi[0,1],1.);self.assertAlmostEqual(gi[0,2],0.)
        self.assertTrue(np.isnan(gi[0,0]))

    def test_class_index_signs(self):
        cls=np.array(['a','a','b','b'])
        gi=np.array([[np.nan,1,0,0],[1,np.nan,0,0],[0,0,np.nan,1],[0,0,1,np.nan]],dtype=float)
        self.assertAlmostEqual(p.class_index(gi,cls),1.)
        self.assertAlmostEqual(p.class_index(gi,cls,'a'),1.)


if __name__=='__main__':unittest.main()
