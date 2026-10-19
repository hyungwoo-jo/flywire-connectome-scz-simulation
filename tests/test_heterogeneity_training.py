import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy import sparse

spec=importlib.util.spec_from_file_location('p71',Path(__file__).resolve().parents[1]/'pipelines/71_heterogeneity_and_training.py')
p=importlib.util.module_from_spec(spec);sys.modules['p71']=p;spec.loader.exec_module(p)


class DialTests(unittest.TestCase):
    def test_dial_endpoints_and_monotone(self):
        m=sparse.random(300,40,density=.1,format='csr',random_state=0)
        np.testing.assert_allclose(p.dial(m,0.).toarray(),m.toarray())
        np.testing.assert_allclose(p.dial(m,1.).toarray(),p.p26.equalize_rows(m).toarray(),atol=1e-12)
        cvs=[p.cv(p.dial(m,a)) for a in p.ALPHAS];self.assertTrue(all(np.diff(cvs)<0));self.assertLess(cvs[-1],1e-10)


class GradTests(unittest.TestCase):
    def test_finite_difference(self):
        rng=np.random.default_rng(0);n_kc,n_pn,B,n_cls=30,8,12,3
        m=sparse.random(n_kc,n_pn,density=.4,format='coo',random_state=1);ij=(m.row,m.col);w=m.data+.5
        c=dict(inh=np.zeros(n_kc),drive=np.ones(n_kc)/n_kc);x=rng.random((n_pn,B));y=rng.integers(0,n_cls,B)
        R=rng.normal(size=(n_cls,n_kc));b=rng.normal(size=n_cls);theta=.3
        loss,gw,gR,gb=p.loss_grad(w,ij,m.shape,x,y,R,b,c,theta);eps=1e-6
        for e in rng.choice(len(w),5,replace=False):
            wp=w.copy();wp[e]+=eps;wm=w.copy();wm[e]-=eps
            fd=(p.loss_grad(wp,ij,m.shape,x,y,R,b,c,theta)[0]-p.loss_grad(wm,ij,m.shape,x,y,R,b,c,theta)[0])/(2*eps)
            self.assertAlmostEqual(gw[e],fd,places=5)
        Rp=R.copy();Rp[1,3]+=eps;Rm=R.copy();Rm[1,3]-=eps
        fd=(p.loss_grad(w,ij,m.shape,x,y,Rp,b,c,theta)[0]-p.loss_grad(w,ij,m.shape,x,y,Rm,b,c,theta)[0])/(2*eps)
        self.assertAlmostEqual(gR[1,3],fd,places=5)


if __name__=='__main__':unittest.main()
