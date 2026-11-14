import unittest
import numpy as np
from scipy.linalg import toeplitz
import protocol_core as c

class Contracts(unittest.TestCase):
    def test_boundaries_and_limit(self):
        t=np.arange(121.)
        np.testing.assert_allclose(c.curve('SE',t,[.1,1,30,1]),c.curve('E1',t,[.1,1,30]))
        np.testing.assert_allclose(c.curve('E2',t,[.1,1,.2,30,30]),c.curve('E1',t,[.1,1,30]))
        self.assertLess(np.max(np.abs(c.curve('PL',t,[0,1,30000,1000])-np.exp(-t/30))),.0003)
        with self.assertRaises(ValueError):c.curve('E2',t,[0,1,.5,30,2])
    def test_time_not_reset(self):
        full=c.curve('E1',np.arange(121),[0,1,30])
        np.testing.assert_array_equal(full[5:],c.curve('E1',np.arange(5,121),[0,1,30]))
        self.assertNotEqual(full[5],1.)
    def test_normalization_and_undershoot(self):
        z=c.normalize(np.full(250,20.),40.,[40.,20.,10.])
        np.testing.assert_allclose(z,[1,0,-1],atol=1e-14)
        with self.assertRaises(ValueError):c.normalize(np.full(249,20.),40.,[40.])
        with self.assertRaises(ValueError):c.normalize(np.full(250,20.),20.,[40.])
    def test_cooldown_and_first_tie(self):
        dates=np.datetime64('2020-01-01')+np.arange(80)
        v=np.full(80,20.); v[1:3]=30; v[31]=30;v[32]=40
        self.assertEqual(c.detect(dates,v),[(1,1),(32,32)])
    def test_grid_and_allocation(self):
        self.assertEqual(len(c.GRID),9)
        self.assertEqual(c.allocation(10,20,30),63)
        self.assertEqual(c.allocation(1,1,1),100)
        self.assertEqual(c.allocation(20000,20000,20000),0)
        with self.assertRaises(ValueError):c.allocation(0,1,1)
    def test_seed(self):
        self.assertEqual(c.seed('S02b','OU',1),c.seed('S02b','OU',1))
        self.assertNotEqual(c.seed('timing','OU',1),c.seed('S02b','OU',1))
    def test_covariance(self):
        for d in (.2,.4):
            r=c.arfima_cov(100,d)
            self.assertAlmostEqual(r[1]/r[0],d/(1-d))
            self.assertGreater(np.linalg.eigvalsh(toeplitz(r)).min(),0)
    def test_overlap_transitive(self):
        np.testing.assert_array_equal(c.overlap_groups([(1,3),(5,7),(3,5),(9,10)]),[0,0,0,1])
    def test_predictor_and_logo_no_target_leakage(self):
        self.assertAlmostEqual(c.buildup(np.exp(3+.02*np.arange(20))),.02)
        x=np.column_stack([np.arange(30),np.sin(np.arange(30))]); y=np.cos(np.arange(30))
        groups=np.repeat(np.arange(10),3)
        p=c.logo_predictions(x,y,groups); changed=y.copy();changed[:3]+=10000
        np.testing.assert_array_equal(p[:3],c.logo_predictions(x,changed,groups)[:3])
    def test_benchmark_recursion_and_no_test_argument(self):
        pre=np.log(20)+.01*np.sin(np.arange(400)); origin=pre[-40:]
        np.testing.assert_array_equal(c.forecast_benchmark(pre,origin,20,'RW'),np.repeat(origin[-1],20))
        coef=np.linalg.lstsq(np.column_stack([np.ones(399),pre[:-1]]),pre[1:],rcond=None)[0]
        first=coef@[1,origin[-1]]
        p=c.forecast_benchmark(pre,origin,20,'AR1')
        self.assertAlmostEqual(p[0],first);self.assertAlmostEqual(p[1],coef@[1,first])
        self.assertEqual(len(c.forecast_benchmark(pre,origin,20,'HAR')),20)
        with self.assertRaises(ValueError):c.forecast_benchmark(pre[:200],origin,20,'HAR')
    def test_noiseless_fits(self):
        fixtures={'E1':[.1,.9,30],'E2':[.1,.9,.3,2,40],'PL':[.1,.9,3,.6],'SE':[.1,.9,30,.6]}
        for family,p in fixtures.items():
            t=np.arange(1.,121.);y=c.curve(family,t,p)
            result=c.fit(family,t,y,('S01','fixture',family))
            self.assertEqual(len(result['starts']),32)
            self.assertIsNotNone(result['best'])
            self.assertLess(result['best']['sse'],1e-8)
            self.assertEqual(result['k'],len(p))
    def test_stage_guard(self):
        with self.assertRaises(RuntimeError):c.stage_runner()

if __name__=='__main__':unittest.main(verbosity=2)
