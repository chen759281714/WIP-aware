"""Isolated metric and real-optimizer smoke tests, never writes formal results."""
import copy
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments import run_formal_comparison as f


class MetricTests(unittest.TestCase):
    def test_front_and_duplicates(self):
        self.assertEqual(f.nondominated([(1,3),(1,3),(2,4),(3,1),(1,4)]),[(1,3),(3,1)])

    def test_exact_hv(self):
        self.assertAlmostEqual(f.compute_hv([(0.2,0.8),(0.8,0.2)]),.45)
        self.assertAlmostEqual(f.compute_hv([(0.2,0.8),(0.8,0.2),(2,0),(0,2),(0.2,0.8)]),.45)
        self.assertEqual(f.compute_hv([(2,0)]),0)
        self.assertAlmostEqual(f.compute_hv([(-.1,.2)]),1.08)

    def test_igd_direction(self):
        self.assertAlmostEqual(f.compute_igd([(0,0)],[(0,0),(0,2)]),1)

    def test_normalization_and_degenerate(self):
        b=f.bounds_for([(10,5),(20,5)])
        self.assertEqual(f.normalize([(15,5)],b),[(.5,0)])
        self.assertEqual(f.normalize([(30,7)],b),[(2,2)])
        self.assertTrue(b['degenerate'][1])
        self.assertTrue(f.bounds_for([(1,1),(1+1e-13,2)])['degenerate'][0])

    def test_bad_points(self):
        for p in ([],[(math.nan,0)],[(1,math.inf)],[(1,)],[(True,1)],[("1",2)]):
            with self.subTest(p=p),self.assertRaises(ValueError):
                f.compute_hv(p)
        with self.assertRaises(ValueError):
            f.bounds_for([(-1,2)])
        with self.assertRaises(ValueError):
            f.compute_igd([(1,1)],[])

    def test_sample_sd_levels(self):
        rows=f.summarize([{'a':'x','hv':1},{'a':'x','hv':3}],['a'],['hv'],'seeds')
        self.assertAlmostEqual(rows[0]['sample_sd_hv'],math.sqrt(2))
        self.assertEqual(rows[0]['sd_level'],'seeds')

    def test_defaults_and_analysis_import_isolation(self):
        c=f.setup()
        self.assertEqual(len(f.tasks(c,c['run_algorithms'])),3840)
        self.assertEqual(c['parameters'],dict(N=10,N_A=20,p_mut=.35,rho=.95,T_coop=3))
        self.assertEqual(c['checkpoints'],list(range(2000,40001,2000)))
        code="from experiments import run_formal_comparison as f; import sys; f.setup(); assert not any(k.startswith('src.algorithms') for k in sys.modules)"
        subprocess.run([sys.executable,'-c',code],cwd=f.ROOT,check=True)


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='formal-comparison-test-')
        cls.root=Path(cls.temp.name)
        cls.overrides={
            'SELECTED_INSTANCES':['WIPHFSP_S1_P1_B1'], 'SEEDS':[10], 'FE_MAX':43,
            'PROPOSED_PARAMS':dict(N=2,N_A=4), 'CONVERGENCE_INTERVAL_FE':12,
            'RESULTS_DIR':str(cls.root/'single'), 'MAKE_FIGURES':False,
            'USE_MULTIPROCESSING':False,
        }
        cls.patches=[patch.object(f,k,v) for k,v in cls.overrides.items()]
        for p in cls.patches:p.start()
        cls.config=f.setup()
        with patch.dict(os.environ,PYTHONHASHSEED='0'):
            f.run_mode(cls.config)

    @classmethod
    def tearDownClass(cls):
        for p in reversed(cls.patches):p.stop()
        cls.temp.cleanup()

    def test_all_variants_saved_and_exact_snapshots(self):
        for t in f.tasks(self.config,self.config['run_algorithms']):
            r=f.load_record(t,self.config)
            self.assertEqual(r['actual_fe'],43)
            self.assertEqual([s['fe'] for s in r['convergence']],[12,24,36,43])
            self.assertEqual(r['convergence'][-1]['archive'],r['final_archive'])

    def test_observation_preserves_rng_and_search(self):
        data=f.validate_instance(Path(self.config['instance_dir'])/'WIPHFSP_S1_P1_B1.json')
        for alg in self.config['run_algorithms']:
            module,clsname,variant,_=f.REGISTRY[alg]
            cls=getattr(importlib.import_module('src.algorithms.'+module),clsname)
            kw=dict(self.config['parameters'],FE_max=43,seed=10)
            if alg=='full':kw['variant']=variant
            plain=cls(data['operations'],data['buffers'],**kw)
            observed=cls(data['operations'],data['buffers'],**kw)
            plain.run()
            snaps=[]
            original=observed.decoder.decode
            with patch.object(observed.decoder,'decode',wraps=original) as decode:
                def callback(state):
                    before=state.rng.getstate()
                    snaps.append((state.n_evaluations,[(x.makespan,x.shortage) for x in state.A]))
                    self.assertEqual(before,state.rng.getstate())
                observed.run(diagnostic_callback=callback)
                self.assertEqual(decode.call_count,43)
            self.assertEqual(plain.rng.getstate(),observed.rng.getstate())
            self.assertEqual([(x.os_seq,x.ms_map,x.makespan,x.shortage) for x in plain.A],
                             [(x.os_seq,x.ms_map,x.makespan,x.shortage) for x in observed.A])
            record=f.load_record(dict(algorithm=alg,instance='WIPHFSP_S1_P1_B1',seed=10),self.config)
            for snap in record['convergence']:
                actual=dict(snaps)[snap['fe']]
                self.assertEqual(snap['archive'],[list(x) for x in actual])

    def test_skip_resume_and_invalid_protection(self):
        config=copy.deepcopy(self.config)
        config['output']=str(self.root/'resume')
        with patch.dict(os.environ,PYTHONHASHSEED='0'):
            f.run_mode(config)
            task=f.tasks(config,['full'])[0]
            path=f.raw_path(task,config)
            saved=path.read_bytes()
            f.run_mode(config)
            self.assertEqual(path.read_bytes(),saved)
            path.unlink()
            f.run_mode(config)
            self.assertEqual(json.loads(saved)['final_archive'],f.load_record(task,config)['final_archive'])
            path.write_text('{broken')
            with self.assertRaises(ValueError):f.run_mode(config)
            self.assertEqual(path.read_text(),'{broken')

    def test_analyze_is_offline_and_missing_rejected(self):
        from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation
        with patch.object(WIPGraphDualPopulation,'__init__',side_effect=AssertionError('optimizer construction')),patch.object(WIPGraphDualPopulation,'evaluate',side_effect=AssertionError('evaluation')):
            folder=f.analyze_mode(self.config)
            self.assertTrue((folder/'run_metrics.csv').exists())
            self.assertTrue((folder/'convergence_summary.csv').exists())
            config=copy.deepcopy(self.config)
            config['seeds']=[10,11]
            with self.assertRaises(ValueError):f.analyze_mode(config)

    def test_conflicts_and_checkpoints(self):
        task=f.tasks(self.config,['full'])[0]
        record=f.load_record(task,self.config)
        for field in ('parameters','sources','checkpoints'):
            bad=copy.deepcopy(record)
            bad['identity'][field]={}
            bad['payload_sha256']=f.digest({k:v for k,v in bad.items() if k!='payload_sha256'})
            with self.assertRaises(ValueError):f.validate_record(bad,f.source_identity(task,self.config))
        bad=copy.deepcopy(record);bad['convergence'][0]['fe']=13
        bad['payload_sha256']=f.digest({k:v for k,v in bad.items() if k!='payload_sha256'})
        with self.assertRaises(ValueError):f.validate_record(bad,record['identity'])
        with patch.object(f,'CONVERGENCE_INTERVAL_FE',5):
            with self.assertRaises(ValueError):f.setup()

    def test_figures_and_evaluation_scope(self):
        config=copy.deepcopy(self.config)
        config['output']=str(self.root/'figures')
        # Reuse genuine smoke results, do not fabricate optimization records.
        import shutil
        shutil.copytree(Path(self.config['output'])/'raw',Path(config['output'])/'raw')
        with patch.object(f,'MAKE_FIGURES',True):
            folder=f.analyze_mode(config)
        target=Path(config['output'])/'figures'/folder.name
        self.assertEqual(len(list(target.rglob('*.png'))),5)
        self.assertEqual(len(list(target.rglob('*.pdf'))),5)
        changed=copy.deepcopy(config)
        changed['analysis_algorithms']=['full']
        with patch.object(f,'MAKE_FIGURES',False):
            other=f.analyze_mode(changed)
        self.assertNotEqual(folder,other)

    def test_real_2000_fe_boundary(self):
        config=copy.deepcopy(self.config)
        config.update(output=str(self.root/'boundary2000'),fe_max=2003,checkpoints=[2000,2003],
                      parameters=dict(N=10,N_A=20,p_mut=.35,rho=.95,T_coop=3),run_algorithms=['full'])
        with patch.dict(os.environ,PYTHONHASHSEED='0'):
            f.run_mode(config)
        record=f.load_record(f.tasks(config,['full'])[0],config)
        self.assertEqual([s['fe'] for s in record['convergence']],[2000,2003])
        self.assertEqual(record['actual_fe'],2003)

    def test_lock_and_save_disabled(self):
        root=self.root/'lock'
        with f.run_lock(root):
            with self.assertRaises(RuntimeError):
                with f.run_lock(root):pass
        self.assertFalse((root/'.session.lock').exists())
        config=copy.deepcopy(self.config)
        config['output']=str(self.root/'no-convergence')
        config['checkpoints']=[]
        config['run_algorithms']=['full']
        with patch.dict(os.environ,PYTHONHASHSEED='0'):
            f.run_mode(config)
        t=f.tasks(config,['full'])[0]
        off=f.load_record(t,config)
        on=f.load_record(t,self.config)
        self.assertEqual(off['final_archive'],on['final_archive'])
        self.assertEqual(off['convergence'],[])
        # Worker/analysis selection does not enter task identity.
        changed=copy.deepcopy(config)
        changed['execution']['max_workers']=9
        changed['analysis_algorithms']=['full']
        self.assertEqual(f.source_identity(t,changed),f.source_identity(t,config))

    def test_spawn_same_results(self):
        config=copy.deepcopy(self.config)
        config['output']=str(self.root/'spawn')
        with patch.object(f,'USE_MULTIPROCESSING',True),patch.object(f,'MAX_WORKERS',2),patch.dict(os.environ,PYTHONHASHSEED='0'):
            f.run_mode(config)
        for t in f.tasks(config,config['run_algorithms']):
            a,b=f.load_record(t,self.config),f.load_record(t,config)
            self.assertEqual(a['final_archive'],b['final_archive'])
            self.assertEqual(a['convergence'],b['convergence'])

    def test_batch_algorithms_and_modes(self):
        config=copy.deepcopy(self.config)
        config['output']=str(self.root/'batch')
        config['run_algorithms']=['wo_ac']
        with patch.dict(os.environ,PYTHONHASHSEED='0'):
            f.run_mode(config)
            with self.assertRaises(ValueError):f.analyze_mode(config)
            config['run_algorithms']=['full','wo_mgs','wo_hlns']
            f.run_mode(config)
            self.assertTrue(f.analyze_mode(config).exists())
        for mode in ('run','analyze','all'):
            with patch.object(sys,'argv',['runner','--mode',mode]),patch.object(f,'setup',return_value=config),patch.object(f,'run_mode') as run,patch.object(f,'analyze_mode') as analyze,patch.dict(os.environ,PYTHONHASHSEED='0'):
                f.main()
                self.assertEqual(run.call_count,int(mode in ('run','all')))
                self.assertEqual(analyze.call_count,int(mode in ('analyze','all')))


if __name__=='__main__':
    unittest.main()
