"""修正回归：可控目标替身验证状态提交和分解更新，不评价正式指标。"""
import copy
import math
import unittest
from unittest.mock import patch
from tests import test_comparison_algorithms as support
CLASSES,optimizer,artificial=support.CLASSES,support.optimizer,support.artificial
from src.algorithms.comparison_common import crowding,environmental,Solution
from src.algorithms.moead import MOEAD
from src.algorithms.mpmoga import MPMOGA
from src.algorithms.comparison_operators import pox


def spawned_run(name):
    support.AlgorithmsTests.setUpClass()
    cls=next(c for c in CLASSES if c.__name__==name)
    a=optimizer(cls,support.AlgorithmsTests.data,FE_max=31,snapshot_interval_fe=10)
    a.run()
    return a.A,a.snapshots,a.diagnostics


class CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        support.AlgorithmsTests.setUpClass();cls.data=support.AlgorithmsTests.data

    def test_exact_callbacks_and_contents(self):
        for cls in CLASSES:
            for budget in (6,10,12,30,31,40,41):
                with self.subTest(cls=cls,budget=budget):
                    a=optimizer(cls,self.data,FE_max=budget,snapshot_interval_fe=10)
                    seen=[]
                    def cb(s):seen.append({'fe':s.n_evaluations,'archive':[list(x.objectives) for x in s.A],'archive_size':len(s.A)})
                    out=a.run(cb)
                    expected=sorted(set(range(10,budget+1,10))|{budget})
                    self.assertEqual([s['fe'] for s in seen],expected)
                    self.assertEqual(seen,a.snapshots)
                    self.assertEqual(seen[-1]['archive'],[list(x.objectives) for x in out])
                    self.assertEqual(len(seen),len(set(x['fe'] for x in seen)))
                    for snap in seen:self.assertTrue(all(math.isfinite(v) for p in snap['archive'] for v in p))

    def test_moead_only_used_mutation(self):
        for cp in (0,1):
            for mp in (0,1):
                a=optimizer(MOEAD,self.data,population_size=10,FE_max=11,crossover_probability=cp,mutation_probability=mp)
                with patch('src.algorithms.comparison_operators.mutate',wraps=__import__('src.algorithms.comparison_operators',fromlist=['mutate']).mutate) as mut:
                    a.run();self.assertEqual(mut.call_count,mp)
                self.assertEqual(a.diagnostics['mutations'],mp)
                self.assertEqual(a.diagnostics['crossovers'],cp)
                self.assertEqual(a.diagnostics['candidates'],1)
                self.assertEqual(a.n_evaluations,11)

    def test_moead_scopes_deterministic(self):
        for local in (0,1):
            a=optimizer(MOEAD,self.data,FE_max=7,neighborhood_mating_probability=local)
            scopes=[];compared=[];bounds=[];replacement_scopes=[]
            original_sample=a.rng.sample
            def sample(scope,k):scopes.append(list(scope));return original_sample(scope,k)
            original=a.scalarize
            def scalar(x,w):
                compared.append(a.weights.index(w));bounds.append((tuple(a.ideal),tuple(a.upper)))
                return original(x,w)
            # 强制候选更差，使遍历完整范围，不受限次替换提前退出影响。
            child=artificial([(1e10,1e10)])[0]
            with patch.object(a.rng,'sample',side_effect=sample),patch.object(a,'scalarize',side_effect=scalar),patch.object(a,'generate_single_offspring',return_value=child),patch.object(a,'evaluate',wraps=a.evaluate) as ev:
                # evaluate 仍需合法染色体，因此单候选入口返回真实父代的副本。
                def offspring(x,y):return x.copy()
                with patch.object(a,'generate_single_offspring',side_effect=offspring),patch.object(a.rng,'shuffle',side_effect=lambda order:replacement_scopes.append(list(order))):
                    a.run()
            expected=a.neighborhoods[0] if local else list(range(6))
            self.assertEqual(scopes[-1],expected)
            self.assertEqual([scope for scope in replacement_scopes if all(type(i) is int for i in scope)],[expected])
            self.assertTrue(set(compared)<=set(expected))
            self.assertEqual(a.diagnostics['local_updates'],local)
            self.assertEqual(a.diagnostics['global_updates'],1-local)
            self.assertEqual(len(set(bounds)),1)
            self.assertLessEqual(a.diagnostics['replacement_counts'][0],2)

    def test_crowding_representatives_and_classic_values(self):
        pop=artificial([(0,10),(0,10),(1,9),(2,8),(10,0)])
        d=crowding(pop,list(range(5)))
        self.assertTrue(math.isinf(d[0]));self.assertEqual(d[1],0)
        self.assertTrue(math.isinf(d[4]))
        unique=[pop[i] for i in (0,2,3,4)]
        standard=crowding(unique,list(range(4)))
        self.assertEqual([d[i] for i in (0,2,3,4)],list(standard.values()))
        self.assertAlmostEqual(d[2],.4);self.assertAlmostEqual(d[3],1.8)
        self.assertEqual(list(crowding(artificial([(1,1)]*3),[0,1,2]).values()),[0,0,0])
        self.assertEqual(len(environmental(pop,4)),4)
        # AUS 同样通过公共环境选择，并保留容量与不同基因型。
        a=optimizer(MPMOGA,self.data,FE_max=6,archive_size=4)
        a.run();self.assertLessEqual(len(a.archive),4)

    def test_initial_failure_retry_and_invalid_genotypes(self):
        for cls in CLASSES:
            for failures,fe,success in ((0,6,True),(1,7,True),(2,7,False),(7,7,False)):
                a=optimizer(cls,self.data,FE_max=fe)
                decode=a.decoder.decode;count=[0]
                def decoder(*args,**kwargs):
                    count[0]+=1
                    if count[0]<=failures:raise RuntimeError('死锁：test')
                    return decode(*args,**kwargs)
                with patch.object(a.decoder,'decode',side_effect=decoder):
                    if success:a.run()
                    else:
                        with self.assertRaisesRegex(RuntimeError,'初始化失败'):a.run()
                self.assertEqual(a.n_evaluations,fe)
                self.assertEqual(count[0],fe)
            a=optimizer(cls,self.data)
            with self.assertRaises(ValueError):a.evaluate(Solution([],{}))
            os=a.encoder.generate_random_os()
            with self.assertRaises(ValueError):a.evaluate(Solution(os,{k:'invalid' for k in a.encoder.ms_index_order}))
            self.assertEqual(a.n_evaluations,0)

    def test_mpmoga_budget_in_all_phases(self):
        for budget,cp,mp,phase in ((7,1,0,'ast'),(7,0,1,'mutation'),(7,0,0,'aus')):
            a=optimizer(MPMOGA,self.data,FE_max=budget,crossover_probability=cp,mutation_probability=mp,snapshot_interval_fe=1)
            seen=[];a.run(lambda s:seen.append(s.n_evaluations))
            self.assertEqual(seen,list(range(1,8)))
            self.assertEqual(a.n_evaluations,7)
            self.assertEqual(a.diagnostics['aus_attempts'],int(phase=='aus'))
            if phase=='mutation':self.assertEqual(a.diagnostics['mutations'],1)

    def test_spawn_independence(self):
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        names=[cls.__name__ for cls in CLASSES]
        expected=[spawned_run(n) for n in names]
        with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
            actual=list(pool.map(spawned_run,names))
        self.assertEqual(actual,expected)

    def test_zero_span_scalarization(self):
        a=optimizer(MOEAD,self.data)
        point=artificial([(1,2)])[0];a.update_bounds(point)
        for w in ((0,1),(1,0),(.5,.5)):
            self.assertEqual(a.scalarize(point,w),0)
        a.update_bounds(artificial([(2,4)])[0])
        self.assertTrue(math.isfinite(a.scalarize(point,(0,1))))

    def test_medium_budget_observer_equivalence(self):
        for cls in CLASSES:
            a=optimizer(cls,self.data,FE_max=203,snapshot_interval_fe=10)
            b=optimizer(cls,self.data,FE_max=203,snapshot_interval_fe=10)
            events=[];a.run(lambda s:events.append(s.n_evaluations));b.run()
            self.assertEqual(a.snapshots,b.snapshots);self.assertEqual(a.diagnostics,b.diagnostics)
            self.assertEqual(a.rng.getstate(),b.rng.getstate())
            self.assertEqual(a.A,b.A)
            self.assertEqual(events,list(range(10,201,10))+[203])

if __name__=='__main__':unittest.main()
