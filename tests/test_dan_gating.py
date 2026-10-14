import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p60',root/'pipelines/60_dan_gating.py')
p=importlib.util.module_from_spec(spec);sys.modules['p60']=p;spec.loader.exec_module(p)


class GateTests(unittest.TestCase):
    def test_gamma_weighted_mean_is_one(self):
        g=np.array([0,1,3,4.]);w=np.array([1,2,0,1.])
        gm=p.gamma(g,w);self.assertAlmostEqual((w*gm).sum()/w.sum(),1.)

    def test_shuffle_keeps_class_multisets(self):
        g=np.arange(10.);cls=np.array(list('gggggabbbb'))
        s=p.shuffle_within(g,cls,np.random.default_rng(0))
        for c in 'gab':self.assertEqual(sorted(s[cls==c]),sorted(g[cls==c]))

    def test_kc_class(self):
        f=p.lib.kc_class
        self.assertEqual([f('KCg-m'),f('KCab-p'),f("KCa'b'-ap1"),f('KCapbp-m'),f('KC')],['g','ab','apbp','apbp','other'])

    def test_gate_patch_restores(self):
        orig=p.p25.value_signal;p.set_gate(np.ones(3));self.assertIsNot(p.p25.value_signal,orig)
        p.set_gate(None);self.assertIs(p.p25.value_signal,orig)


if __name__=='__main__':unittest.main()
