import unittest
from aboot.schema import Record
from aboot.models import TrainableRetriever
from aboot.diagnostics import diagnose, ranking_stats

class DiagnosticsTests(unittest.TestCase):
    def test_known_ranking(self):
        r=ranking_stats([1,0,1,0],[1,2,100])
        self.assertAlmostEqual(r['average_precision'],5/6)
        self.assertEqual(r['2']['recall'],.5)
        self.assertEqual(r['100']['effective_k'],4)
    def test_empty_support(self):
        self.assertIsNone(ranking_stats([0,0],[1])['1']['recall'])
    def test_retrieval_misses_count_as_end_to_end_false_negatives(self):
        class Ranking:
            def score(self,p): return {'x':3,'y':2,'z':1}[p.id]
        class Perfect:
            def predict(self,p): return 1.
        rows=[Record(i,i,i,{'a':1},'test') for i in ['x','y','z']]
        r=diagnose(Ranking(),Perfect(),rows,{'a':'d'},[1])['per_code']['a']
        self.assertEqual(r['classifier_on_retrieved']['1']['recall'],1.)
        self.assertEqual(r['end_to_end']['1']['recall'],1/3)
        self.assertEqual(r['end_to_end']['1']['fn'],2)
    def test_unknown_labels_excluded(self):
        class Model:
            def score(self,p): return 0.
            def predict(self,p): return .5
        r=diagnose(Model(),Model(),[Record('x','g','t',{},'t')],{'a':'d'},[10])
        self.assertEqual(r['per_code']['a']['retrieval']['corpus_size'],0)

if __name__=='__main__': unittest.main()
