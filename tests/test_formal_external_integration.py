"""外部算法集成与旧身份兼容性；所有写操作仅发生在临时目录。"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import run_formal_comparison as f


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='formal-eight-test-')
        cls.root=Path(cls.tmp.name)
        cls.params={
            'nsga2':dict(population_size=6,crossover_probability=.9,mutation_probability=.1),
            'moead':dict(population_size=6,crossover_probability=.9,mutation_probability=.1,neighborhood_size=3,neighborhood_mating_probability=.9,max_replacements=2),
            'spea2':dict(population_size=6,archive_size=8,crossover_probability=.9,mutation_probability=.1),
            'mpmoga':dict(makespan_population_size=3,shortage_population_size=3,archive_size=8,crossover_probability=.9,mutation_probability=.1)}
        cls.patches=[patch.object(f,k,v) for k,v in dict(
            SELECTED_INSTANCES=['WIPHFSP_S1_P1_B1','WIPHFSP_S4_P2_B3'],SEEDS=[10],
            SELECTED_ALGORITHMS=list(f.REGISTRY),ANALYSIS_ALGORITHMS=None,EXTERNAL_PARAMETERS=cls.params,
            PROPOSED_PARAMS=dict(N=2,N_A=4),FE_MAX=43,CONVERGENCE_INTERVAL_FE=12,
            RESULTS_DIR=str(cls.root/'single'),USE_MULTIPROCESSING=False,MAKE_FIGURES=False).items()]
        for p in cls.patches:p.start()
        cls.config=f.setup()
        with patch.dict(os.environ,PYTHONHASHSEED='0'):f.run_mode(cls.config)

    @classmethod
    def tearDownClass(cls):
        for p in reversed(cls.patches):p.stop()
        cls.tmp.cleanup()

    def test_saved_native_and_snapshots(self):
        for task in f.tasks(self.config,list(f.REGISTRY)):
            r=f.load_record(task,self.config)
            self.assertEqual(r['actual_fe'],43)
            self.assertEqual([s['fe'] for s in r['convergence']],[12,24,36,43])
            self.assertEqual(r['convergence'][-1]['archive'],r['final_archive'])

    def test_eight_algorithm_offline_analysis(self):
        with patch('importlib.import_module',side_effect=AssertionError('analysis imported optimizer')):
            folder=f.analyze_mode(self.config)
        import csv
        with (folder/'run_metrics.csv').open(encoding='utf-8-sig') as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),16)
        bounds=json.loads((folder/'normalization_bounds.json').read_text())
        self.assertEqual(len(bounds),2)
        self.assertEqual(set(r['algorithm'] for r in rows),set(f.REGISTRY))

    def test_spawn_resume_and_reproducibility(self):
        config=copy.deepcopy(self.config);config['output']=str(self.root/'spawn')
        config['run_algorithms']=list(self.params)
        with patch.object(f,'USE_MULTIPROCESSING',True),patch.object(f,'MAX_WORKERS',2),patch.dict(os.environ,PYTHONHASHSEED='0'):
            f.run_mode(config)
            hashes={str(f.raw_path(t,config)):f.file_hash(f.raw_path(t,config)) for t in f.tasks(config,list(self.params))}
            f.run_mode(config)
            self.assertTrue(all(f.file_hash(p)==h for p,h in hashes.items()))
        for t in f.tasks(config,list(self.params)):
            a,b=f.load_record(t,self.config),f.load_record(t,config)
            self.assertEqual(a['final_archive'],b['final_archive']);self.assertEqual(a['convergence'],b['convergence'])
        t=f.tasks(config,['nsga2'])[0];f.raw_path(t,config).unlink()
        with patch.object(f,'USE_MULTIPROCESSING',False),patch.dict(os.environ,PYTHONHASHSEED='0'):f.run_mode(config)
        self.assertEqual(f.load_record(t,config)['final_archive'],f.load_record(t,self.config)['final_archive'])

    def test_conflicts_invalid_and_worker_failure(self):
        t=f.tasks(self.config,['nsga2'])[0]
        config=copy.deepcopy(self.config);config['external_parameters']['nsga2']['mutation_probability']=.8
        with self.assertRaises(ValueError):f.load_record(t,config)
        r=f.load_record(t,self.config)
        for field,value in [('actual_fe',42),('final_archive',[[1,2],[1,2]])]:
            bad=copy.deepcopy(r);bad[field]=value;bad['payload_sha256']=f.digest({k:v for k,v in bad.items() if k!='payload_sha256'})
            with self.assertRaises(ValueError):f.validate_record(bad,r['identity'])
        c=copy.deepcopy(self.config);c['output']=str(self.root/'failure')
        with patch.dict(os.environ,PYTHONHASHSEED='0'),patch('importlib.import_module',side_effect=RuntimeError('worker failure')):
            status=f.run_task(t,c)
        self.assertEqual(status['status'],'failed');self.assertFalse(f.raw_path(t,c).exists())
        with f.run_lock(self.root/'locked'):
            with self.assertRaises(RuntimeError):
                with f.run_lock(self.root/'locked'):pass

    def test_external_checkpoint_not_full_multiple(self):
        with patch.object(f,'SELECTED_ALGORITHMS',list(self.params)),patch.object(f,'CONVERGENCE_INTERVAL_FE',7):
            c=f.setup();self.assertEqual(c['checkpoints'],[7,14,21,28,35,42,43])
        with patch.object(f,'CONVERGENCE_INTERVAL_FE',7):
            with self.assertRaises(ValueError):f.setup()

    def test_real_history_samples(self):
        root=f.ROOT/'experiments/results/formal_comparison'
        if not (root/'config.json').exists():self.skipTest('本地无历史正式记录')
        c=json.loads((root/'config.json').read_text())
        c.update(output=str(root),instance_dir=str(f.ROOT/'data/final_benchmark/instances'))
        for alg in f.LEGACY_ALGORITHMS:
            t=dict(algorithm=alg,instance=c['instances'][0],seed=c['seeds'][0])
            r=f.load_record(t,c)
            self.assertEqual(r['identity'],f.source_identity(t,c))

    def test_external_2000_boundary(self):
        c=copy.deepcopy(self.config);c.update(output=str(self.root/'boundary'),fe_max=2003,
                                            checkpoints=[2000,2003],external_snapshot_interval_fe=2000)
        c['run_algorithms']=list(self.params)
        with patch.dict(os.environ,PYTHONHASHSEED='0'):f.run_mode(c)
        summary=[]
        for t in f.tasks(c,list(self.params)):
            r=f.load_record(t,c)
            self.assertEqual([s['fe'] for s in r['convergence']],[2000,2003])
            summary.append({**t,'fe':r['actual_fe'],'archive_size':r['archive_size']})
        Path('/private/tmp/integration-boundary-summary.json').write_text(json.dumps(summary,indent=2))

    def test_legacy_identity_unchanged(self):
        # 首阶段保存的可信旧脚本快照，仅用于对照，不参与运行。
        path=Path('/private/tmp/formal-runner-before.py')
        if not path.exists():self.skipTest('旧脚本快照不可用；历史JSON兼容性由独立只读验证覆盖')
        spec=importlib.util.spec_from_file_location('formal_before',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        old.ROOT=f.ROOT;old.STARTUP_SOURCE_HASHES=f.STARTUP_SOURCE_HASHES
        for task in f.tasks(self.config,list(f.LEGACY_ALGORITHMS)):
            self.assertEqual(old.source_identity(task,self.config),f.source_identity(task,self.config))

if __name__=='__main__':unittest.main()
