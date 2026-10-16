import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p64',Path(__file__).resolve().parents[1]/'pipelines/64_kc_spontaneous_events.py')
p=importlib.util.module_from_spec(spec);sys.modules['p64']=p;spec.loader.exec_module(p)


class EventTests(unittest.TestCase):
    def test_runs(self):
        m=np.array([[1,1,0,1,0,1,1,1],[0,0,0,0,0,0,0,1]],bool)
        self.assertEqual(p.runs(m,2).tolist(),[2,0]);self.assertEqual(p.runs(m,1).tolist(),[3,1])

    def test_event_rates(self):
        rng=np.random.default_rng(0);base=rng.normal(0,.01,(4,2,17))
        base[0,0,5:8]=1.;base[2,0,10:12]=1.;base[1,1,3:5]=-1.
        pos,neg=p.event_rates(base)
        self.assertEqual(pos[0],.5);self.assertEqual(neg[1],.25);self.assertEqual(pos[1],0.)


if __name__=='__main__':unittest.main()
