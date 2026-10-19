import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p68',Path(__file__).resolve().parents[1]/'pipelines/68_interpretability_ground_truth.py')
p=importlib.util.module_from_spec(spec);sys.modules['p68']=p;spec.loader.exec_module(p)


def sparse_data(seed=0,d=60,K=6,n=4000):
    """Nonnegative sparse sources mixed by nonnegative, nearly orthogonal atoms."""
    rng=np.random.default_rng(seed);A=np.abs(rng.normal(size=(d,K)))*(rng.random((d,K))<.3)
    S=rng.exponential(1.,(K,n))*(rng.random((K,n))<.15);return A,A@S+.01*rng.normal(size=(d,n)),rng


class MetricTests(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(p.auc(np.array([3.,2,1,0]),np.array([1,1,0,0])),1.)
        self.assertEqual(p.auc(np.array([0.,1,2,3]),np.array([1,1,0,0])),0.)
        self.assertAlmostEqual(p.auc(np.ones(4),np.array([1,0,1,0])),.5)
        self.assertTrue(np.isnan(p.auc(np.ones(3),np.array([1,1,1]))))

    def test_recovery_and_coherence(self):
        A=np.eye(5)[:, :3];self.assertAlmostEqual(p.recovery(A,A)[0],1.)
        self.assertAlmostEqual(p.recovery(A,-2*A)[0],1.);self.assertAlmostEqual(p.coherence(A)[0],0.)

    def test_ridge(self):
        rng=np.random.default_rng(1);X=rng.normal(size=(2000,4));y=X@np.array([2.,0,0,-1])+.01*rng.normal(size=2000)
        b=p.ridge(X,y[:,None],1e-6)[:,0];self.assertGreater(b[0],1.5);self.assertLess(b[3],-.5);self.assertLess(abs(b[1]),.05)


class MethodTests(unittest.TestCase):
    def test_pca_orthonormal(self):
        A,Y,_=sparse_data();U,_=p.pca_atoms(Y,6);np.testing.assert_allclose(U.T@U,np.eye(6),atol=1e-10)

    def test_ica_recovers(self):
        A,Y,rng=sparse_data();self.assertGreater(p.recovery(A,p.ica_atoms(Y,6,rng))[0],.9)

    def test_nmf_recovers(self):
        A,Y,rng=sparse_data();self.assertGreater(p.recovery(A,p.nmf_atoms(np.maximum(Y,0),6,rng))[0],.85)

    def test_sae_recovers_and_beats_chance(self):
        A,Y,rng=sparse_data();W,info=p.sae_atoms(Y,12,p.SAE_LAMS[0],rng)
        self.assertGreater(p.recovery(A,W)[0],.8);self.assertGreater(p.recovery(A,W)[0],p.recovery(A,rng.normal(size=(60,12)))[0]+.3)
        np.testing.assert_allclose(np.linalg.norm(W,axis=0),1.,atol=1e-6);self.assertIn('dead',info)


class CenteredSaeTest(unittest.TestCase):
    def test_centered_sae_recovers(self):
        A,Y,rng=sparse_data(3);W,info=p.sae_atoms(Y+5.,12,p.SAE_LAMS[0],rng,center=True)  # large offset must not matter
        self.assertGreater(p.recovery(A,W)[0],.8);self.assertLess(info['dead'],.6)


if __name__=='__main__':unittest.main()
