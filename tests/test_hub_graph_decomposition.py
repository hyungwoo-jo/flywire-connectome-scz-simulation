import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p54',Path(__file__).resolve().parents[1]/'pipelines/54_hub_graph_decomposition.py')
p=importlib.util.module_from_spec(spec);sys.modules['p54']=p;spec.loader.exec_module(p)


class HubGraphTests(unittest.TestCase):
    def test_hfac_matches_step43(self):
        rng=np.random.default_rng(0);w=rng.random(50);m=rng.random(50);k=rng.random(50)
        _,H,_=p.p43.index_terms(w,k,m)
        self.assertAlmostEqual(p.hfac(w,m,1-np.exp(-p.p25.ETA*k)),H)

    def test_share_bounds(self):
        rng=np.random.default_rng(1);s=rng.random(200);w=np.ones(200)
        m=s**4;D=np.stack([s**2,s**3],1)
        _,_,sh=p.shares(w,m,D,s)
        self.assertTrue(np.all(sh>.8))                     # m, d depend on s only
        _,_,sh0=p.shares(w,m,D,rng.permutation(s))
        self.assertTrue(np.all(np.abs(sh0)<.3))            # unrelated strength explains little


if __name__=='__main__':unittest.main()
