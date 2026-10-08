import importlib.util
from pathlib import Path
import unittest
import numpy as np

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p22',root/'pipelines/22_left_replication.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


@unittest.skipUnless((root/'data/banc_888_meta.feather').exists(),'BANC data not present')
class SideCircuitTests(unittest.TestCase):
    def test_right_side_matches_original_loader(self):
        odors,_,_=p.cal.load_hallem(root);rec=list(odors.columns)
        c=p.load_side_circuit(root,'right','720575941482622627',rec)
        pre,post,w,inh,drive,info=p.base.load_circuit(root)
        np.testing.assert_array_equal(c['pre'],pre);np.testing.assert_array_equal(c['post'],post)
        np.testing.assert_allclose(c['w'],w);np.testing.assert_allclose(c['inh'],inh);np.testing.assert_allclose(c['drive'],drive)
        orig=p.cal.load_partial_circuit(root,rec)['mapped']
        self.assertEqual(list(c['mapped'].root_888),list(orig.root_888))
        self.assertEqual(list(c['mapped'].receptor),list(orig.receptor))


if __name__=='__main__':unittest.main()
