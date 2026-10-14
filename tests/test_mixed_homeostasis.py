import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('p57',Path(__file__).resolve().parents[1]/'pipelines/57_mixed_homeostasis.py')
p=importlib.util.module_from_spec(spec);sys.modules['p57']=p;spec.loader.exec_module(p)


class MixedTests(unittest.TestCase):
    def test_both_goals(self):
        sp=pd.Series([.001,.004,.002],index=list('abc'))
        d=pd.DataFrame(dict(diff=[0.,.01,-.03],se=[.01,.01,.01]),index=list('abc'))
        self.assertEqual(p.both_goals(sp,d).tolist(),[True,False,False])

    def test_total_homeostasis_equalizes_activity(self):
        rng=np.random.default_rng(1);n=200
        Eo=rng.random((n,30))*rng.uniform(.5,1.5,n)[:,None];En=rng.normal(.2,.05,(n,300))*rng.uniform(.5,2,n)[:,None]
        c=dict(inh=np.zeros(n),drive=np.zeros((1,n)));tune=np.ones(n,bool)
        th,cv=p.mixed_homeostasis(Eo,En,c,.6,tune,.5)
        ko=p.p55.respond(Eo,c,np.full((n,1),.6)).mean(1);kn=p.p55.respond(En,c,np.full((n,1),.6)).mean(1)
        cv0=(.5*ko+.5*kn).std()/(.5*ko+.5*kn).mean()
        self.assertLess(cv,cv0/3)


if __name__=='__main__':unittest.main()
