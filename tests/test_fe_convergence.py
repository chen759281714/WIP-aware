"""Correctness of a read-only observer and predeclared convergence arithmetic."""
import copy
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from experiments import run_fe_convergence as audit
from tests.hlns_test_support import make


def small_run(seed=2, observed=True, fe=66):
    engine, _ = make(seed=seed, fe=fe)
    observer = audit.SnapshotObserver('fixture', seed, interval=6)
    engine.run(diagnostic_callback=observer if observed else None)
    def state(ind):
        return (engine.genotype_key(ind), ind.makespan, ind.shortage,
                ind.schedule, ind.buffer_trace, ind.stats, ind.provenance)
    return dict(pop_m=[state(x) for x in engine.P_M], pop_s=[state(x) for x in engine.P_S],
                archive=[state(x) for x in engine.A], rng=engine.rng.getstate(),
                fe=engine.n_evaluations, diagnostics=engine.diagnostics,
                snapshots=observer.snapshots)


class ObserverTests(unittest.TestCase):
    def test_exact_search_state_and_rng_equivalence(self):
        for seed in (0, 2):
            plain, observed = small_run(seed,False), small_run(seed,True)
            for key in plain:
                if key != 'snapshots':
                    self.assertEqual(plain[key],observed[key],key)

    def test_frequency_final_and_independent_copy(self):
        engine,_=make(fe=30)
        observer=audit.SnapshotObserver('fixture',0,interval=6)
        engine.run(diagnostic_callback=observer)
        self.assertEqual([r['FE'] for r in observer.snapshots],list(range(6,31,6)))
        self.assertEqual(observer.snapshots[-1]['archive_objectives'],audit.objectives(engine))
        saved=copy.deepcopy(observer.snapshots)
        before=engine.rng.getstate()
        observer(engine)
        self.assertEqual(before,engine.rng.getstate())
        engine.A[0].makespan=-999
        self.assertEqual(observer.snapshots,saved)

    def test_long_budget_prefix_equals_short_termination(self):
        shorter,_=make(seed=3,fe=30)
        longer,_=make(seed=3,fe=66)
        shorter.run()
        expected=(shorter.rng.getstate(),copy.deepcopy(shorter.diagnostics),
                  [shorter.genotype_key(x) for x in shorter.P_M+shorter.P_S+shorter.A],
                  [(x.makespan,x.shortage) for x in shorter.P_M+shorter.P_S+shorter.A])
        captured=[]
        def callback(e):
            if e.n_evaluations==30:
                captured.append((e.rng.getstate(),copy.deepcopy(e.diagnostics),
                                 [e.genotype_key(x) for x in e.P_M+e.P_S+e.A],
                                 [(x.makespan,x.shortage) for x in e.P_M+e.P_S+e.A]))
        longer.run(diagnostic_callback=callback)
        self.assertEqual(captured,[expected])

    def test_spawn_matches_serial(self):
        serial=small_run()
        with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn')) as pool:
            spawned=pool.submit(small_run).result()
        for key in serial:
            if key!='snapshots':
                self.assertEqual(serial[key],spawned[key],key)
        for a,b in zip(serial['snapshots'],spawned['snapshots']):
            self.assertEqual({k:v for k,v in a.items() if k!='elapsed_seconds'},
                             {k:v for k,v in b.items() if k!='elapsed_seconds'})

    def test_formal_parameters_and_identity(self):
        data=json.loads((audit.ROOT/'data/final_benchmark/instances'/ (audit.INSTANCES[0]+'.json')).read_text())
        e=audit.WIPGraphDualPopulation(data['operations'],data['buffers'],variant='HLNS_FULL',**dict(audit.PARAMS,seed=0))
        self.assertEqual((e.N,e.N_A,e.FE_max),(50,100,60000))
        self.assertEqual((e.pm_specialized_kind,e.ps_specialized_kind,e.pm_archive_guidance_enabled,e.ps_archive_guidance_enabled),('MGS','HLNS',True,False))
        self.assertEqual(tuple(audit.metrics.SELECTED_INSTANCES),audit.INSTANCES)


class ArithmeticTests(unittest.TestCase):
    def test_nonmonotone_future_best(self):
        rows=[dict(FE=f,nhv=h,igd=i,makespan=c,shortage=s) for f,h,i,c,s in
              [(10000,.8,.2,100,10),(15000,.9,.1,98,9),(20000,.85,.15,99,9.5)]]
        result=audit.residual(rows,0)
        self.assertAlmostEqual(result['R_nhv'],.125)
        self.assertAlmostEqual(result['R_igd'],.5)
        self.assertAlmostEqual(result['R_makespan'],.02)
        self.assertAlmostEqual(result['R_shortage'],.1)
        self.assertFalse(result['pass'])

    def test_zero_and_terminal(self):
        rows=[dict(FE=60000,nhv=0,igd=0,makespan=1,shortage=0)]
        r=audit.residual(rows,0)
        self.assertTrue(r['pass'])
        self.assertTrue(all(r['R_'+k]==0 for k in audit.LIMITS))

    def test_window_checks_all_interior_checkpoints(self):
        rows=[dict(FE=f,nhv=h,igd=.1,makespan=1,shortage=0) for f,h in [(10000,.5),(11000,.8),(15000,.5),(16000,.9)]]
        self.assertAlmostEqual(audit.residual(rows,0,15000)['R_nhv'],.6)

    def test_metric_parity_and_fixed_scale(self):
        m=audit.metrics
        self.assertAlmostEqual(m.compute_hv_2d([(0,0)])/1.21,1)
        self.assertEqual(m.compute_igd([(0,0)],[(0,0)]),0)
        bounds=m.pooled_bounds([(10,20),(20,10),(30,30)])
        self.assertEqual(m.normalize_point((20,10),bounds),(.5,0))

    def test_large_group_and_instance_guard(self):
        rows=[]
        for name in audit.INSTANCES:
            for seed in audit.SEEDS:
                rows.append(dict(instance=name,size=name.split('_')[1],seed=seed,
                                 **{'pass':True},R_nhv=0,R_igd=0))
        self.assertTrue(audit.coverage_ok(rows)[0])
        for r in rows:
            if r['size']=='S8' and r['seed']==0:
                r['pass']=False
                r['R_nhv']=.2
        # Overall 93% and S8 80%, but per-instance remaining gain is too large.
        self.assertFalse(audit.coverage_ok(rows)[0])

    def test_sources_unchanged_and_snapshot_no_decode(self):
        from unittest.mock import patch
        before=audit.source_hashes()
        engine,_=make(fe=6)
        engine.run()
        observer=audit.SnapshotObserver('fixture',0)
        fe=engine.n_evaluations
        with patch.object(engine.decoder,'decode',side_effect=AssertionError('decode')), \
             patch.object(engine.decoder,'analyze',side_effect=AssertionError('analyze')):
            observer(engine)
        self.assertEqual(fe,engine.n_evaluations)
        self.assertEqual(before,audit.source_hashes())


class RecordAndAnalysisTests(unittest.TestCase):
    def record(self, name, seed):
        hashes=audit.source_hashes()
        expected=audit.identity(name,seed,hashes)
        snaps=[dict(instance=name,seed=seed,FE=fe,archive_size=1,
                    best_makespan=10.,best_shortage=0.,archive_objectives=[[10.,0.]],
                    elapsed_seconds=float(fe)) for fe in [100]+list(range(1000,60001,1000))]
        return dict(identity=expected,actual_FE=60000,actual_parameters=dict(audit.PARAMS,seed=seed),
                    population_sizes={'P_M':50,'P_S':50},mechanisms=['MGS','HLNS',True,False],
                    snapshots=snaps,final_archive=[[10.,0.]],runtime_seconds=60000.)

    def test_invalid_or_partial_records_rejected(self):
        record=self.record(audit.INSTANCES[0],0)
        audit.validate_record(record,record['identity'])
        for field,value in [('actual_FE',59000),('final_archive',[[11.,0.]]),
                            ('population_sizes',{'P_M':100,'P_S':50}),
                            ('snapshots',record['snapshots'][:-1])]:
            damaged=copy.deepcopy(record);damaged[field]=value
            with self.assertRaises(ValueError):
                audit.validate_record(damaged,record['identity'])

    def test_end_to_end_constant_front_earliest_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in audit.INSTANCES:
                for seed in audit.SEEDS:
                    audit.atomic_json(root/'raw'/name/('seed_%d.json'%seed),self.record(name,seed))
            with patch.object(audit,'make_figures'):
                decision=audit.analyze(root,audit.source_hashes())
            self.assertEqual(decision['recommended_FE'],10000)
            self.assertEqual(decision['formal_FE'],3600000)
            self.assertEqual(decision['late_event_count'],0)
            self.assertTrue((root/'analysis/fe_convergence_report.md').exists())
            self.assertTrue(all(r['threshold_pass_rate']==1 for r in decision['candidates']))


if __name__=='__main__':
    unittest.main()
