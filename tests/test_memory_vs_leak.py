import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p55',Path(__file__).resolve().parents[1]/'pipelines/55_memory_vs_leak.py')
p=importlib.util.module_from_spec(spec);sys.modules['p55']=p;spec.loader.exec_module(p)


class MemoryTests(unittest.TestCase):
    def test_disjoint_odors_are_learned_perfectly(self):
        n=6;kt=np.eye(n);lab=np.repeat(np.arange(n),3);kte=kt[:,lab]
        self.assertEqual(p.memory(kt,kte,lab,n,n),[1.]*p.N_DRAW)

    def test_silent_trials_score_chance(self):
        n=4;kt=np.eye(n);lab=np.repeat(np.arange(n),2);kte=np.zeros((n,len(lab)))
        self.assertEqual(p.memory(kt,kte,lab,n,n),[.5]*p.N_DRAW)


if __name__=='__main__':unittest.main()
