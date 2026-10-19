import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy import sparse

spec=importlib.util.spec_from_file_location('p72',Path(__file__).resolve().parents[1]/'pipelines/72_joint_objective.py')
p=importlib.util.module_from_spec(spec);sys.modules['p72']=p;spec.loader.exec_module(p)


class ProxyTests(unittest.TestCase):
    def test_proxy_sign(self):
        kt=np.zeros((10,3));kt[:3]=1;kn=np.full((10,5),.1)
        self.assertAlmostEqual(p.proxy(kt,kn),0.,places=9)
        kn[:3]=.5;self.assertGreater(p.proxy(kt,kn),0.)
        kn=np.full((10,5),.1);kn[3:]=.5;self.assertLess(p.proxy(kt,kn),0.)

    def test_finite_difference(self):
        rng=np.random.default_rng(0);n_kc,n_pn=30,8
        m=sparse.random(n_kc,n_pn,density=.4,format='coo',random_state=1);ij=(m.row,m.col);w=m.data+.5
        c=dict(inh=np.zeros(n_kc),drive=np.ones(n_kc)/n_kc);xo=rng.random((n_pn,6));xs=rng.random((n_pn,9))*.8;theta=.3
        D,gw=p.penalty_grad(w,ij,m.shape,xo,xs,c,theta);eps=1e-6
        for e in rng.choice(len(w),6,replace=False):
            wp=w.copy();wp[e]+=eps;wm=w.copy();wm[e]-=eps
            fd=(p.penalty_grad(wp,ij,m.shape,xo,xs,c,theta)[0]-p.penalty_grad(wm,ij,m.shape,xo,xs,c,theta)[0])/(2*eps)
            self.assertAlmostEqual(gw[e],fd,places=5)


class LamStarTests(unittest.TestCase):
    def test_interpolation(self):
        self.assertAlmostEqual(p.lam_star([.1,1,10],[2.,1.,.5],1.),1.)
        self.assertAlmostEqual(p.lam_star([1,100],[2.,0.],1.),10.)
        self.assertIsNone(p.lam_star([.1,1],[2.,1.5],1.))


if __name__=='__main__':unittest.main()
