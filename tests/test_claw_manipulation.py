import importlib.util
import sys
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p39',Path(__file__).resolve().parents[1]/'pipelines/39_claw_manipulation.py')
p=importlib.util.module_from_spec(spec);sys.modules['p39']=p;spec.loader.exec_module(p)


def toy():
    pre=np.array([0,1,2,0,3]);post=np.array([0,0,0,1,1]);w=np.array([6.,7.,1.,8.,5.])*.1
    return dict(pre=pre,post=post,w=w,inh=np.ones(2),info=dict(pn=5))


class ClawTests(unittest.TestCase):
    def test_counts_recovered_from_unit(self):
        cnt,unit=p.counts_of(toy());np.testing.assert_array_equal(cnt,[6,7,1,8,5]);self.assertAlmostEqual(unit,.1)

    def test_remove_keeps_one_claw_per_kc(self):
        out,_=p.remove_claws(toy(),np.random.default_rng(0))
        self.assertEqual(sorted(out['post'].tolist()),[0,1]);self.assertTrue((p.counts_of(toy())[1]*0+out['w']>=.5-1e-12).all())

    def test_add_claws_creates_new_unique_partners(self):
        c=toy();out,n=p.add_claws(c,np.random.default_rng(1))
        self.assertGreater(n,0);pairs=list(zip(out['pre'].tolist(),out['post'].tolist()));self.assertEqual(len(pairs),len(set(pairs)))


if __name__=='__main__':unittest.main()
