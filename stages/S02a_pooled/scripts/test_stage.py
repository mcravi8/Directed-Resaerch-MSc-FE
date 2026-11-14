"""Independent numerical checks, no empirical outcomes."""
import math
import unittest
import numpy as np
import pandas as pd
from scipy.special import gammaln
import pipeline as p

class StageChecks(unittest.TestCase):
    def test_stationary_covariance(self):
        for d in (.2,.4):
            k=np.arange(15000)
            independent=.16*np.exp(gammaln(k+d)-gammaln(k+1-d)+gammaln(1-d)-gammaln(d))
            r=p.core.arfima_cov(15000,d)
            np.testing.assert_allclose(r,independent,rtol=1e-10)
            embedding=np.r_[r,r[-2:0:-1]]
            eigen=np.fft.fft(embedding).real
            self.assertGreaterEqual(eigen.min(),0)
            np.testing.assert_allclose(np.fft.ifft(eigen).real[:len(r)],r,atol=1e-14)
    def test_exponential_limit(self):
        t=np.arange(251); rate=.03
        errors=[np.max(np.abs(np.exp(-rate*t)-(1+t/c)**(-rate*c))) for c in (100,1000,10000)]
        self.assertTrue(errors[0]>errors[1]>errors[2])
        self.assertLess(errors[-1],.001)
    def test_cohort_dates_and_truncation(self):
        dates=pd.bdate_range('2000-01-03',periods=600)
        values=np.full(600,20.); values[300:]=30.
        ts=pd.DataFrame({'VIX30D':values},index=dates)
        ev=pd.DataFrame([dict(event_id='a',onset_date=dates[290],peak_date=dates[300]),dict(event_id='b',onset_date=dates[450],peak_date=dates[455])])
        z,iv,rows=p.panels(ts,ev,'VIX30D',120)
        self.assertIn('a',z); self.assertEqual(z['a'].shape,(121,))
        np.testing.assert_allclose(z['a'],1)
        z,_,_=p.panels(ts,ev,'VIX30D',250); self.assertNotIn('a',z)
        ts.iloc[320,0]=np.nan
        z,_,_=p.panels(ts,ev,'VIX30D',120); self.assertNotIn('a',z)
    def test_noiseless_optimizer(self):
        for family,pars in [('E1',[.1,.8,20]),('E2',[.1,.8,.4,3,70]),('PL',[.1,.8,5,.8]),('SE',[.1,.8,20,.6])]:
            t=np.arange(1,121); y=p.core.curve(family,t,pars)
            out=p.core.fit(family,t,y,['S02a','synthetic',family])
            self.assertEqual(len(out['starts']),32)
            self.assertIsNotNone(out['best'])
            self.assertLess(out['best']['sse'],1e-10)
    def test_allocation(self):
        self.assertEqual(p.core.allocation(100,200,300),math.floor(21600/(2*(300+800+600))))
        self.assertEqual(p.core.allocation(5000,5000,5000),0)

if __name__=='__main__': unittest.main(verbosity=2)
