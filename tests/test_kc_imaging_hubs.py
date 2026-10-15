import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('p63',Path(__file__).resolve().parents[1]/'pipelines/63_kc_imaging_hubs.py')
p=importlib.util.module_from_spec(spec);sys.modules['p63']=p;spec.loader.exec_module(p)


class MeasureTests(unittest.TestCase):
    def test_model_breadth(self):
        n_odor=2;k=np.zeros((3,(n_odor+1)*p.N_TRIAL))
        k[0,:3]=1;k[0,6:9]=1  # KC0 reliable on both odors
        k[1,:2]=1             # KC1 below MIN_REL
        k[2,12:]=1            # KC2 blank only
        B,S=p.model_breadth(k,n_odor)
        self.assertEqual(B.tolist(),[2,0,0]);self.assertEqual(S.tolist(),[0.,0.,1.])

    def test_real_measures(self):
        odors=list(p.ODORS)+[p.BLANK];head=pd.DataFrame(dict(odor=np.repeat(odors,p.N_TRIAL),fp=.5))
        n=len(head);tr=np.zeros((n,2,30));sig=np.ones((n,2))
        tr[:,0,15:19]=1.  # cell 0 responds in window (t 7.5-9.0) on every trial
        sig[:,0]=0.;sig[(head.odor==p.BLANK).to_numpy(),1]=0.
        m=p.real_measures(head,tr,sig)
        self.assertEqual(m['B'].tolist(),[6,0]);self.assertEqual(m['S'].tolist(),[1.,1.]);self.assertAlmostEqual(m['S_r'][0],1.)

    def test_perm(self):
        rng=np.random.default_rng(0);B=np.arange(30);obs,pv=p.perm_test(B,B+rng.random(30),rng,n=200)
        self.assertGreater(obs,.9);self.assertLess(pv,.01)


if __name__=='__main__':unittest.main()
