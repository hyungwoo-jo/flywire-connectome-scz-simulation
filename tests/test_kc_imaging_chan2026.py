import importlib.util
import io
import pickle
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('p65',Path(__file__).resolve().parents[1]/'pipelines/65_kc_imaging_chan2026.py')
p=importlib.util.module_from_spec(spec);sys.modules['p65']=p;spec.loader.exec_module(p)


class Evil:
    def __reduce__(self):return (print,('should not run',))


class LoaderTests(unittest.TestCase):
    def test_restricted_npz(self):
        with tempfile.TemporaryDirectory() as d:
            ok=Path(d)/'ok.npz';np.savez(ok,tracedict=np.array({0:{'oil':np.ones((2,3))}},dtype=object),before_onset=np.array(20))
            r=p.load_npz(ok);self.assertEqual(r['tracedict'][0]['oil'].shape,(2,3));self.assertEqual(int(r['before_onset']),20)
            bad=Path(d)/'bad.npz';np.savez(bad,x=np.array([Evil()],dtype=object))
            with self.assertRaises(pickle.UnpicklingError):p.load_npz(bad)


class MeasureTests(unittest.TestCase):
    def test_normalize_and_respond(self):
        rng=np.random.default_rng(0);tr=100+rng.normal(0,1,(2,29));tr[0,22:25]+=50
        x=p.normalize(tr,20);self.assertAlmostEqual(np.nanmean(x[1,:21]),0,places=6)
        self.assertEqual(p.responds(x,20,9).tolist(),[True,False])
        npl=np.full((2,29),10.);self.assertTrue((p.normalize(tr,20,npl)>=0).all())

    def test_rho_guards(self):
        self.assertTrue(np.isnan(p.rho(np.zeros(10),np.arange(10.))))
        self.assertAlmostEqual(p.rho(np.arange(10.),np.arange(10.)),1.)


if __name__=='__main__':unittest.main()
