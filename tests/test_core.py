import copy, json, random, tempfile, unittest
from pathlib import Path
from aboot.schema import Record, Pair, pairs, binary
from aboot.models import LinearClassifier, TrainableRetriever, group_advantages
from aboot.selection import sample_actions, select_teacher
from aboot.splits import split_records, validate_splits
from aboot.teacher import OracleTeacher, GPTTeacher
from aboot.loop import run, balanced_take
from aboot.readers import ODDReader, SBDHReader, OurReader
from aboot.metrics import evaluate

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.pos=Pair('p','a','opioid use','opioid use pain')
        self.neg=Pair('n','a','opioid use','denies opioid use')

    def test_missing_label_not_negative(self):
        for x in ('',None,'possible',2):
            with self.assertRaises(ValueError): binary(x)

    def test_gold_not_in_pair(self):
        p=pairs([Record('p','g','text',{'a':1},'test')],{'a':'definition'})[0]
        self.assertFalse(hasattr(p,'labels')); self.assertFalse(hasattr(p,'group_id'))

    def test_all_strategies_exclude_endpoints_and_cap(self):
        gs=[(Pair(str(i),'a','d','t'),[1]*i+[0]*(8-i),.5) for i in range(9)]
        for strategy in ('uniform','uniform_bins','central'):
            selected=select_teacher(gs,4,strategy,random.Random(42))
            self.assertEqual(len(selected),4)
            self.assertTrue(all(0<sum(g[1])<8 for g in selected))
            self.assertEqual(len({g[0].id for g in selected}),4)

    def test_bin_uniformity(self):
        gs=[(Pair(f'{k}-{i}','a','d','t'),[1]*k+[0]*(8-k),.5) for k in range(1,8) for i in range(20)]
        selected=select_teacher(gs,14,'uniform_bins',random.Random(1))
        from collections import Counter
        self.assertEqual(set(Counter(sum(a) for _,a,_ in selected).values()),{2})

    def test_sft_direction(self):
        c=LinearClassifier(['a']); before=c.predict(self.pos)
        c.sft([(self.pos,1.,1.)]); self.assertGreater(c.predict(self.pos),before)

    def test_grpo_direction_both_labels(self):
        for label in (0,1):
            c=LinearClassifier(['a']); ref=copy.deepcopy(c)
            c.grpo([(self.pos,[0,1]*4,.5,label)],ref)
            self.assertGreater((c.predict(self.pos)-.5)*(2*label-1),0)

    def test_constant_group_zero_signal(self):
        self.assertEqual(group_advantages([1]*8,0),[0.]*8)
        c=LinearClassifier(['a']); before=c.state()['b'].copy()
        c.grpo([(self.pos,[1]*8,.5,0)],copy.deepcopy(c))
        self.assertEqual(c.b,before)

    def test_retriever_margin_improves(self):
        r=TrainableRetriever({'a':'opioid use'})
        before=r.score(self.pos)-r.score(self.neg)
        r.train([(self.pos,1),(self.neg,0)],random.Random(1),epochs=4)
        self.assertGreater(r.score(self.pos)-r.score(self.neg),before)

    def test_split_patient_and_duplicate_integrity(self):
        rs=[Record(str(i),str(i),'same' if i<2 else f'text {i}',{'a':i%2},'t') for i in range(30)]
        split_records(rs); validate_splits(rs)
        self.assertEqual(rs[0].split,rs[1].split)
        other=copy.deepcopy(rs); other[1].split='test' if other[0].split!='test' else 'train'
        with self.assertRaises(ValueError): validate_splits(other)

    def test_missing_support_is_not_zero_recall(self):
        result=evaluate(LinearClassifier(['a']),[Record('x','g','x',{'a':0},'t')],{'a':'d'})
        self.assertIsNone(result['per_code']['a']['recall'])

    def test_odd_and_mapped_readers(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.jsonl'
            path.write_text(json.dumps({'rid':'x','patient':'p','body':'text','y':1})+'\n')
            mapping={'id':'rid','group_id':'patient','text':'body','labels':{'a':'y'}}
            for cls in (ODDReader,SBDHReader,OurReader):
                r=cls().read(path,mapping)[0]; self.assertEqual(r.labels,{'a':1})

    def test_teacher_cache_uses_text_and_definition(self):
        with tempfile.TemporaryDirectory() as d:
            t=GPTTeacher('test',Path(d)/'cache')
            changed=Pair('p','a','changed',self.pos.text)
            self.assertNotEqual(t.cache_key(self.pos),t.cache_key(changed))

    def test_oracle_only_has_given_train_labels(self):
        t=OracleTeacher([Record('p','g','x',{'a':1},'t')])
        t.label(self.pos); t.label(self.pos); self.assertEqual(t.calls,1)
        with self.assertRaises(KeyError): t.label(self.neg)

    def test_all_modes_budget_training_and_no_test_output(self):
        rs=[Record(str(i),str(i),('opioid pain' if i%2 else 'no medication')+f' case {i}',{'a':i%2},'t') for i in range(45)]
        split_records(rs)
        for mode in ('sft','grpo','soft','hybrid'):
            with tempfile.TemporaryDirectory() as d:
                teacher=OracleTeacher([r for r in rs if r.split=='train'])
                cfg=dict(mode=mode,rounds=3,seed=3,rollouts=8,top_k_per_code=20,explore_per_code=10,
                         teacher_budget_total=9,teacher_budget_round=4,self_threshold=.51,stable_rounds=1)
                logs=run(cfg,rs,{'a':'opioid medication'},teacher,Path(d)/'run')
                self.assertLessEqual(teacher.calls,9)
                self.assertTrue((Path(d)/'run/best_classifier.json').exists())
                self.assertTrue(all('test' not in row for row in logs))
                if mode=='soft': self.assertEqual(teacher.calls,logs[0]['teacher_requests'])
                if mode in ('soft','hybrid'):
                    self.assertGreater(sum(row['self_training_pairs'] for row in logs),0)
                else:
                    self.assertEqual(sum(row['self_training_pairs'] for row in logs),0)

if __name__=='__main__': unittest.main()
