import copy
import json
import math
from pathlib import Path
import random
import unittest
from unittest.mock import patch
from src.algorithms.comparison_common import Solution,fronts,crowding,environmental,native,dominates,genotype
from src.algorithms.comparison_operators import explicit_os_to_job_sequence,job_sequence_to_explicit_os,pox,insertion,machine_mutation
from src.algorithms.nsga2 import NSGAII
from src.algorithms.moead import MOEAD
from src.algorithms.spea2 import SPEA2,fitness,archive_selection
from src.algorithms.mpmoga import MPMOGA
from src.solution.decoder import StageBufferWIPScheduler
from src.solution.encoder import Encoder

ROOT=Path(__file__).resolve().parents[1]
CLASSES=(NSGAII,MOEAD,SPEA2,MPMOGA)

def optimizer(cls,data,**kwargs):
    params=dict(FE_max=43,seed=12,snapshot_interval_fe=7)
    if cls is MPMOGA: params.update(makespan_population_size=3,shortage_population_size=3,archive_size=8)
    else: params['population_size']=6
    if cls is SPEA2: params['archive_size']=8
    if cls is MOEAD: params['neighborhood_size']=3
    params.update(kwargs)
    return cls(data['operations'],data['buffers'],**params)


def artificial(values):
    return [Solution([('J',i)],{('J',i):'M'},x,y) for i,(x,y) in enumerate(values)]


class AlgorithmsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/'data/final_benchmark/instances/WIPHFSP_S1_P1_B1.json').read_text())

    def test_conversions_unequal_and_single_job(self):
        ops={'a':[{},{}],'b':[{}]}
        seq=['a','b','a']
        self.assertEqual(explicit_os_to_job_sequence(job_sequence_to_explicit_os(seq,ops)),seq)
        for bad in (['a','b'],['a','b','b'],['x']):
            with self.assertRaises(ValueError):job_sequence_to_explicit_os(bad,ops)
        with self.assertRaises(ValueError):explicit_os_to_job_sequence([('a',1)])
        one={'a':[{'machines':{'m':1}}]}
        a=Solution([('a',0)],{('a',0):'m'})
        self.assertEqual(pox(a,a,one,random.Random(0))[0],a)
        self.assertEqual(machine_mutation(a,one,random.Random(0)),a)

    def test_operators_and_ms_inheritance(self):
        ops=self.data['operations']; e=Encoder(ops,rng=random.Random(1))
        a=Solution(e.generate_random_os(),e.build_ms_map(e.generate_random_ms()))
        b=Solution(e.generate_random_os(),e.build_ms_map(e.generate_random_ms()))
        saved=copy.deepcopy((a,b))
        for seed in range(15):
            rng=random.Random(seed)
            selected=set(rng.sample(list(ops),rng.randrange(1,len(ops))))
            c,d=pox(a,b,ops,random.Random(seed))
            for child,first,second in ((c,a,b),(d,b,a)):
                e.validate_os(child.os_seq)
                for key,value in child.ms_map.items():
                    self.assertEqual(value,(first if key[0] in selected else second).ms_map[key])
            e.validate_os(insertion(a,ops,rng).os_seq)
            m=machine_mutation(a,ops,rng)
            self.assertEqual(sum(m.ms_map[k]!=a.ms_map[k] for k in m.ms_map),1)
        self.assertEqual((a,b),saved)

    def test_evaluation_matches_decoder(self):
        for cls in CLASSES:
            opt=optimizer(cls,self.data)
            for _ in range(3):
                a=Solution(opt.encoder.generate_random_os(),opt.encoder.build_ms_map(opt.encoder.generate_random_ms()))
                out=opt.evaluate(a)
                decoder=StageBufferWIPScheduler(self.data['operations'],self.data['buffers'])
                m,s,t=decoder.decode(a.os_seq,a.ms_map)
                self.assertEqual(out.objectives,(m,decoder.analyze(s,t,m)['shortage']['total_shortage_area']))

    def test_sorting_crowding_selection(self):
        pop=artificial([(1,3),(2,2),(3,1),(4,4),(2,2)])
        self.assertEqual(set(fronts(pop)[0]),{0,1,2,4})
        self.assertEqual(fronts(pop)[1],[3])
        self.assertTrue(dominates(pop[0],pop[3]))
        d=crowding(pop,fronts(pop)[0]);self.assertTrue(math.isinf(d[0]))
        self.assertEqual(len(environmental(pop,3)),3)
        self.assertEqual(len(native(pop)),3)
        self.assertEqual(list(crowding(artificial([(1,1),(1,1)]),[0,1]).values()),[0,0])
        opt=optimizer(NSGAII,self.data)
        for _ in range(10):self.assertIs(opt.tournament(pop,{0:0,1:1,2:1,3:1,4:1},{i:0 for i in range(5)}) in pop,True)

    def test_moead_scales_weights(self):
        opt=optimizer(MOEAD,self.data)
        self.assertEqual(opt.weights[0],(0,1));self.assertEqual(opt.weights[-1],(1,0))
        a,b=artificial([(2,30),(4,10)])
        opt.update_bounds(a);opt.update_bounds(b)
        self.assertEqual(opt.ideal,[2,10]);self.assertEqual(opt.upper,[4,30])
        self.assertAlmostEqual(opt.scalarize(a,(.5,.5)),.5)
        self.assertIn(0,opt.neighborhoods[0])
        opt.run()
        self.assertTrue(all(v<=2 for v in opt.diagnostics['replacement_counts']))
        self.assertEqual([x.objectives for x in opt.A],[x.objectives for x in native(opt.population)])

    def test_spea2_strength_and_truncation(self):
        pop=artificial([(1,1),(2,2),(3,3)])
        strength,raw,density,f,_=fitness(pop)
        self.assertEqual(strength,[2,1,0]);self.assertEqual(raw,[0,2,3])
        self.assertTrue(all(0<=x<=.5 for x in density))
        self.assertEqual(archive_selection(pop,2),pop[:2])
        line=artificial([(0,4),(1,3),(2,2),(3,1),(4,0)])
        selected=archive_selection(line,2)
        self.assertEqual([x.objectives for x in selected],[(0,4),(4,0)])

    def test_mpmoga_ast_branches_ties(self):
        opt=optimizer(MPMOGA,self.data,crossover_probability=0)
        p,a=artificial([(2,20),(1,30)])
        opt.archive=[a]
        self.assertEqual(opt.ast([p],0)[0].objectives,a.objectives)
        self.assertEqual(opt.ast([p],1)[0].objectives,p.objectives)
        opt.crossover_probability=1
        children=artificial([(1,40),(3,1)])
        with patch('src.algorithms.mpmoga.pox',return_value=children),patch.object(opt,'evaluate',side_effect=lambda x:x):
            self.assertEqual(opt.ast([p],0)[0].objectives,(1,40))
            self.assertEqual(opt.ast([p],1)[0].objectives,(3,1))
        tie=artificial([(2,1)])[0]
        opt.crossover_probability=0;opt.archive=[tie]
        self.assertEqual(opt.ast([p],0)[0].objectives,p.objectives)

    def test_mpmoga_aus_all_archive_os_only(self):
        opt=optimizer(MPMOGA,self.data,mutation_probability=0)
        opt.archive=opt.initial();opt.makespan_population=opt.archive[:3];opt.shortage_population=opt.archive[3:]
        n=len(opt.archive);before=opt.n_evaluations
        from src.algorithms.comparison_operators import insertion as original
        with patch('src.algorithms.mpmoga.insertion',wraps=original) as im:
            opt.aus()
            self.assertEqual(im.call_count,n)
        self.assertEqual(opt.n_evaluations-before,n)
        self.assertEqual(opt.diagnostics['aus_attempts'],n)
        pop=artificial([(1,1),(2,2)])
        self.assertEqual(len(environmental(pop,8)),2) # AUS 可以保留被支配个体

    def test_real_runs_reproducibility_callbacks_and_budget(self):
        summary=[]
        for name in ('WIPHFSP_S1_P1_B1','WIPHFSP_S4_P2_B3','WIPHFSP_S8_P4_B2'):
            data=json.loads((ROOT/f'data/final_benchmark/instances/{name}.json').read_text())
            for cls in CLASSES:
                a,b=optimizer(cls,data),optimizer(cls,data)
                seen=[]
                def callback(state):
                    before=state.rng.getstate();seen.append((state.n_evaluations,[x.objectives for x in state.A]));self.assertEqual(before,state.rng.getstate())
                with patch.object(a.decoder,'decode',wraps=a.decoder.decode) as decode:
                    a.run(callback);self.assertEqual(decode.call_count,43)
                b.run()
                self.assertEqual(a.n_evaluations,43)
                self.assertEqual(a.snapshots,b.snapshots)
                self.assertEqual(a.rng.getstate(),b.rng.getstate())
                self.assertEqual(a.diagnostics,b.diagnostics)
                self.assertEqual([genotype(x) for x in a.A],[genotype(x) for x in b.A])
                self.assertTrue(a.A)
                self.assertEqual(a.snapshots[-1]['archive'],[list(x.objectives) for x in a.A])
                self.assertEqual([s['fe'] for s in a.snapshots],[7,14,21,28,35,42,43])
                for x in a.A:a.encoder.validate_os(x.os_seq);self.assertTrue(all(math.isfinite(v) for v in x.objectives))
                summary.append((name,cls.__name__,a.n_evaluations,len(a.A),a.diagnostics))
        Path('/private/tmp/comparison-smoke-summary.json').write_text(json.dumps(summary,indent=2))

    def test_mutation_strict_objective_and_probability(self):
        opt=optimizer(MPMOGA,self.data,mutation_probability=1)
        p,c=artificial([(2,20),(2,1)])
        opt.archive=[p]
        with patch('src.algorithms.mpmoga.mutate',return_value=c) as mutation,patch.object(opt,'evaluate',side_effect=lambda x:x):
            self.assertEqual(opt.mutate_population([p],0)[0].objectives,p.objectives)
            self.assertEqual(opt.mutate_population([p],1)[0].objectives,c.objectives)
            self.assertEqual(mutation.call_count,2)
        opt.mutation_probability=0
        with patch('src.algorithms.mpmoga.mutate') as mutation:
            opt.mutate_population([p],0);mutation.assert_not_called()

    def test_default_parameters_and_initial_budget(self):
        for cls in CLASSES:
            opt=cls(self.data['operations'],self.data['buffers'])
            self.assertEqual(opt.population_size,210 if cls is MPMOGA else 100)
            self.assertEqual((opt.crossover_probability,opt.mutation_probability),(.9,.1))
            with self.assertRaises(ValueError):optimizer(cls,self.data,FE_max=1)
        for cls in CLASSES:
            opt=optimizer(cls,self.data,FE_max=6,snapshot_interval_fe=2)
            opt.run()
            self.assertEqual(opt.n_evaluations,6);self.assertTrue(opt.A)

    def test_deadlock_and_program_errors_are_distinct(self):
        opt=optimizer(NSGAII,self.data)
        a=Solution(opt.encoder.generate_random_os(),opt.encoder.build_ms_map(opt.encoder.generate_random_ms()))
        with patch.object(opt.decoder,'decode',side_effect=RuntimeError('死锁：test')):
            self.assertIsNone(opt.evaluate(a));self.assertEqual(opt.n_evaluations,1)
        with patch.object(opt.decoder,'decode',side_effect=RuntimeError('其他程序错误')):
            with self.assertRaises(RuntimeError):opt.evaluate(a)
        self.assertEqual(opt.diagnostics['infeasible'],1)

if __name__=='__main__':unittest.main()
