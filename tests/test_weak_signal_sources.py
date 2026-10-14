import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('p62',Path(__file__).resolve().parents[1]/'pipelines/62_weak_signal_sources.py')
p=importlib.util.module_from_spec(spec);sys.modules['p62']=p;spec.loader.exec_module(p)


def toy():
    mapped=pd.DataFrame(dict(pn_index=[0,1,2],glomerulus=['A','A','B']))
    return dict(pre=np.array([0,1,2,2]),post=np.array([0,1,1,2]),w=np.array([1.,2.,3.,1.]),
                inh=np.array([.5,1.,1.5]),drive=np.array([.2,.3,.5]),mapped=mapped)


class VariantTests(unittest.TestCase):
    def test_glom_matched_hits_target(self):
        c=toy();v=p.variant(c,'glom_matched',dict(A=6.,B=2.))
        self.assertEqual(p.glom_totals(v),dict(A=6.,B=2.))
        self.assertAlmostEqual(v['w'][1]/v['w'][0],2.)  # within-glomerulus ratio kept
        self.assertEqual(c['w'][0],1.)  # original untouched

    def test_apl_and_binary(self):
        c=toy()
        self.assertFalse(p.variant(c,'no_apl')['inh'].any())
        u=p.variant(c,'uniform_apl');self.assertTrue(np.allclose(u['inh'],1));self.assertAlmostEqual(u['drive'].sum(),1.)
        b=p.variant(c,'binary_pn');self.assertEqual(len(set(b['w'])),1)

    def test_overlap_product(self):
        rng=np.random.default_rng(0);w=rng.random(30);learn=np.where(rng.random(30)<.3,rng.random(30),0);kw=rng.random((30,1))*(rng.random((30,1))<.5)
        P,D=p.overlap_terms(w,learn,kw);L=p.p25.value_signal(w,learn,kw)[0]
        self.assertAlmostEqual(P*D,L)


if __name__=='__main__':unittest.main()
