import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from scipy import sparse

spec=importlib.util.spec_from_file_location('p69',Path(__file__).resolve().parents[1]/'pipelines/69_wiring_flatness.py')
p=importlib.util.module_from_spec(spec);sys.modules['p69']=p;spec.loader.exec_module(p)


class PerturbTests(unittest.TestCase):
    def test_mean_one_and_pattern(self):
        m=sparse.random(200,50,density=.2,format='csr',random_state=0);mp=p.perturb(m,.3,np.random.default_rng(0))
        np.testing.assert_array_equal((m!=0).toarray(),(mp!=0).toarray())
        self.assertAlmostEqual(float(np.mean(mp.data/m.data)),1.,delta=.02)
        np.testing.assert_allclose(p.perturb(m,0.,np.random.default_rng(1)).toarray(),m.toarray())
        self.assertIsNot(p.perturb(m,.1,np.random.default_rng(0)),m)


class VerdictTests(unittest.TestCase):
    def frame(self,real_s,flat_s,null_s):
        rows=[]
        for i,n in enumerate(p.NAMES):
            rows.append(dict(dataset=n,graph='real',sigma=.3,ident0=.8,mem0=.7,S_ident=real_s[i],S_mem=real_s[i]))
            rows.append(dict(dataset=n,graph='flat',sigma=.3,ident0=.8,mem0=.7,S_ident=flat_s[i],S_mem=flat_s[i]))
            for j,v in enumerate(null_s):rows.append(dict(dataset=n,graph=f'null_{j}',sigma=.3,ident0=.8-.01*j,mem0=.7-.01*j,S_ident=v,S_mem=v))
        return pd.DataFrame(rows)

    def test_flatter_and_sharper(self):
        v,_=p.verdicts(self.frame([.01]*5,[.2]*5,list(np.linspace(.05,.1,10))));self.assertEqual(v['F1'],'flatter')
        self.assertEqual(v['F3_ident'],'equalizing sharpens');self.assertEqual(v['F4']['ident'],'negative')
        v,_=p.verdicts(self.frame([.5]*5,[.01]*5,list(np.linspace(.05,.1,10))));self.assertEqual(v['F2'],'sharper')
        self.assertEqual(v['F3_mem'],'equalizing flattens')
        v,_=p.verdicts(self.frame([.01,.01,.5,.5,.075],[.1]*5,list(np.linspace(.05,.1,10))));self.assertEqual(v['F1'],'no difference')


if __name__=='__main__':unittest.main()
