import importlib.util
from pathlib import Path
import unittest
import numpy as np

root=Path(__file__).resolve().parents[1]


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,root/'pipelines'/file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


cal=load('cal16','16_hallem_calibration.py')
com=load('com17','17_pn_community.py')
disc=load('disc18','18_community_discrimination.py')


class OlsenTransformTests(unittest.TestCase):
    def test_matches_published_equation(self):
        orn=np.zeros((24,1));orn[0,0]=50.;orn[1,0]=30.
        s=10.63*80/190
        expected=165*50**1.5/(50**1.5+12**1.5+s**1.5)
        self.assertAlmostEqual(cal.olsen(orn)[0,0],expected)

    def test_negative_changes_are_clipped_and_lateral_input_suppresses(self):
        orn=np.zeros((24,2));orn[0]=[40.,40.];orn[1]=[-20.,200.]
        pn=cal.olsen(orn)
        self.assertEqual(pn[1,0],0.)
        self.assertLess(pn[0,1],pn[0,0])

    def test_poisson_noise_is_unbiased_in_rate(self):
        delta=np.array([[30.],[0.]]);sfr=np.array([10.,5.])
        x=cal.sample_orn(delta,sfr,np.zeros(20000,int),.5,np.random.default_rng(0))
        self.assertAlmostEqual(x[0].mean(),30.,delta=.3)
        self.assertAlmostEqual(x[1].mean(),0.,delta=.2)

    def test_split_is_fixed_and_sized(self):
        a=cal.split_odors(list(range(110)));b=cal.split_odors(list(range(110)))
        np.testing.assert_array_equal(a,b);self.assertEqual(a.sum(),37)


class CommunityTests(unittest.TestCase):
    def test_co_convergence_counts_shared_kcs(self):
        pre=np.array([0,1,2,0]);post=np.array([0,0,1,1]);types=np.array([0,1,1])
        c=com.co_convergence(pre,post,types,2,2)
        self.assertEqual(c[0,1],2.);self.assertEqual(c[0,0],2.)

    def test_components_minimum_size(self):
        adj=np.zeros((5,5),bool);adj[0,1]=adj[1,0]=adj[1,2]=adj[2,1]=True;adj[3,4]=adj[4,3]=True
        self.assertEqual(com.components(adj),[[0,1,2]])


class DiscriminationTests(unittest.TestCase):
    def test_dprime_cap_only_for_zero_variance(self):
        # Odor 0 trials project identically: zero variance and nonzero difference.
        g=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]]);lt=np.array([0,0,1,1])
        self.assertEqual(disc.pair_dprime(g,lt,np.array([0,1])),disc.DPRIME_CAP)
        g2=np.array([[3.,0.],[1.,0.],[0.,1.],[0.,3.]])
        self.assertAlmostEqual(disc.pair_dprime(g2,lt,np.array([0,1])),4/np.sqrt(2))

    def test_accuracy_ties_scored_uniformly(self):
        cent=np.zeros((2,3));kt=np.zeros((2,3));lt=np.array([0,1])
        self.assertAlmostEqual(disc.accuracy(kt,cent,lt,np.array([0,1])),.5)

    def test_d_test_identical_groups_gives_zero(self):
        import pandas as pd
        rng=np.random.default_rng(1);v=rng.normal(size=11)
        f=pd.DataFrame(dict(graph=np.arange(11),X_HIGH_dprime=v,X_LOW_dprime=v))
        r=disc.d_test(f,'X')
        self.assertAlmostEqual(r['D'],0.);self.assertGreater(r['p_one_sided'],.5)


if __name__=='__main__':unittest.main()
