import unittest
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from research_v2.src.analysis.evaluate_nacc_external import WeightedRank,rr_from_counts

class NaccStatisticsTests(unittest.TestCase):
    def test_weighted_ties_and_zero_bootstrap_multiplicity_match_sklearn(self):
        y=np.array([1,0,1,0,1,0]);p=np.array([.5,.5,.8,.2,.2,.9]);w=np.array([1,.5,1,.3,.8,.1]);ids=np.array([0,1,0,2,2,1])
        rank=WeightedRank(y,p,w,ids)
        for mult in [np.array([1,1,1]),np.array([2,0,1]),np.array([0,1,2])]:
            a,ap=rank.score(mult);weight=w*mult[ids]
            self.assertAlmostEqual(a,roc_auc_score(y,p,sample_weight=weight),places=12)
            self.assertAlmostEqual(ap,average_precision_score(y,p,sample_weight=weight),places=12)
    def test_no_error_or_all_error_is_not_a_valid_auc(self):
        rank=WeightedRank([1,0],[.9,.1],[1,1],np.array([0,1]))
        self.assertTrue(np.isnan(rank.score(np.array([1,0]))[0]))
    def test_rr_uses_weighted_denominators(self):
        rr,rd,a,b=rr_from_counts(np.array([20.,100.,10.,200.]))
        self.assertAlmostEqual(rr,4);self.assertAlmostEqual(rd,.15)
        self.assertAlmostEqual(a,.2);self.assertAlmostEqual(b,.05)

if __name__=='__main__':unittest.main()
