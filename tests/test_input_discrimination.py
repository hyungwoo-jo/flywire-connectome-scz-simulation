import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('experiment',Path(__file__).resolve().parents[1]/'pipelines/06_input_discrimination.py')
experiment=importlib.util.module_from_spec(spec);spec.loader.exec_module(experiment)


class CircuitTests(unittest.TestCase):
    def test_scalar_feedback_has_known_solution(self):
        k,res=experiment.response(np.array([[.8]]),np.array([1.]),np.array([1.]),.1,1,gain=2)
        self.assertAlmostEqual(float(k[0,0]),.7/3,places=9)
        self.assertLess(res,1e-9)

    def test_zero_inhibition_is_feedforward(self):
        e=np.array([[.2,2.],[.9,.1]])
        k,res=experiment.response(e,np.ones(2),np.array([.5,.5]),.3,0)
        np.testing.assert_allclose(k,np.clip(e-.3,0,1))
        self.assertLess(res,1e-9)

    def test_null_preserves_bipartite_degrees_and_source_weights(self):
        pre=np.repeat(np.arange(6),3);post=np.concatenate([np.array([i,i+1,i+3])%9 for i in range(6)])
        for seed in range(3):
            target,_=experiment.rewire(pre,post,np.random.default_rng(seed),multiple=2)
            np.testing.assert_array_equal(np.bincount(target,minlength=9),np.bincount(post,minlength=9))
            self.assertEqual(len(set(zip(pre,target))),len(pre))

    def test_id_release_and_verified_transmitters_match(self):
        root=Path(__file__).resolve().parents[1]
        pre,post,w,inh,drive,info=experiment.load_circuit(root)
        self.assertEqual(info['id_column'],'root_888')
        self.assertEqual(info['pn_verified_transmitters'],{'acetylcholine':info['pn']})
        self.assertEqual(info['apl_verified_transmitter'],['gaba'])
        self.assertGreater(info['apl_kc_edges'],2000)
        self.assertGreater(info['kc_apl_edges'],2000)
        self.assertEqual(info['kc_verified_transmitters'],{'acetylcholine':info['kc']})


if __name__=='__main__':unittest.main()
