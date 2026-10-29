"""正式启动前只读/接口核验：禁止优化、解码；联合分析使用明确标记的合成夹具。"""
import copy
import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import run_formal_comparison as f


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=f.setup()
        # 五算法默认配置只登记所需算法；八算法核验显式补入其余注册信息。
        cls.config["registry"].update(f.REGISTRY)
        cls.sample='WIPHFSP_S1_P1_B1'
        cls.data=f.validate_instance(Path(cls.config['instance_dir'])/(cls.sample+'.json'))

    def test_formal_task_selection(self):
        c=self.config
        self.assertEqual(c['run_algorithms'],['nsga2','moead','spea2','mpmoga'])
        self.assertEqual(c['analysis_algorithms'],['full','nsga2','moead','spea2','mpmoga'])
        self.assertEqual(len(c['instances']),96);self.assertEqual(c['seeds'],list(range(10,20)))
        batch=f.tasks(c,c['run_algorithms']);self.assertEqual(len(batch),3840)
        self.assertTrue(all(t['algorithm'] not in f.LEGACY_ALGORITHMS for t in batch))
        self.assertEqual(c['fe_max'],40000);self.assertEqual(c['checkpoints'],list(range(2000,40001,2000)))
        self.assertEqual(len({str(f.raw_path(t,c)) for t in batch}),3840)

    def test_construct_actual_parameters_without_evaluation(self):
        expected={
            'nsga2':dict(population_size=100,crossover_probability=.9,mutation_probability=.1),
            'moead':dict(population_size=100,crossover_probability=.9,mutation_probability=.1,neighborhood_size=20,neighborhood_mating_probability=.9,max_replacements=2),
            'spea2':dict(population_size=100,archive_size=100,crossover_probability=.9,mutation_probability=.1),
            'mpmoga':dict(makespan_population_size=105,shortage_population_size=105,archive_size=210,crossover_probability=.9,mutation_probability=.1)}
        self.assertEqual(f.EXTERNAL_PARAMETERS,expected)
        results={}
        for a in expected:
            t=dict(algorithm=a,instance=self.sample,seed=10);i=f.source_identity(t,self.config)
            module,cls,_,_=f.REGISTRY[a]
            typ=getattr(importlib.import_module('src.algorithms.'+module),cls)
            with patch.object(typ,'run',side_effect=AssertionError('禁止搜索')):
                engine=typ(self.data['operations'],self.data['buffers'],**i['parameters'])
            for k,v in i['parameters'].items():
                if k!='seed':self.assertEqual(getattr(engine,k),v)
            self.assertEqual(engine.n_evaluations,0)
            if a=='mpmoga':self.assertEqual(engine.population_size,210)
            results[a]=i['parameters']
        Path('/private/tmp/preflight-parameters.json').write_text(json.dumps(results,indent=2))

    def test_batch_identity_invariance_and_dependencies(self):
        for a in f.REGISTRY:
            t=dict(algorithm=a,instance=self.sample,seed=10);i=f.source_identity(t,self.config)
            c=copy.deepcopy(self.config);c.update(run_algorithms=['nsga2'],analysis_algorithms=list(f.REGISTRY))
            self.assertEqual(i,f.source_identity(t,c))
            if a in f.LEGACY_ALGORITHMS:
                self.assertFalse(any('comparison_' in s for s in i['sources']))
            else:
                self.assertIn('src/algorithms/comparison_common.py',i['sources'])
                self.assertIn('src/algorithms/comparison_operators.py',i['sources'])
                c['external_parameters'][a]['mutation_probability']=.2
                self.assertNotEqual(i,f.source_identity(t,c))
        self.assertEqual(self.config['parameters'],dict(N=10,N_A=20,p_mut=.35,rho=.95,T_coop=3))

    def test_five_and_eight_fixture_analysis_no_search(self):
        # 真实旧记录只复制读取；外部记录是接口夹具，绝不作为算法性能结果。
        with tempfile.TemporaryDirectory(prefix='formal-preflight-') as temp:
            c=copy.deepcopy(self.config);c.update(output=temp,instances=[self.sample],seeds=[10],subset=True)
            source=f.load_record(dict(algorithm='full',instance=self.sample,seed=10),self.config)
            for a in f.REGISTRY:
                t=dict(algorithm=a,instance=self.sample,seed=10)
                if a in f.LEGACY_ALGORITHMS:r=f.load_record(t,self.config)
                else:
                    identity=f.source_identity(t,c);p=identity['parameters'];r=copy.deepcopy(source)
                    final=[list(x) for x in f.nondominated(r['final_archive'])]
                    r.update(identity=identity,fingerprint=f.digest(identity),actual_parameters=p,
                             algorithm_variant=identity['variant'],final_archive=final,archive_size=len(final))
                    r.pop('mechanisms',None)
                    for snap in r['convergence']:
                        snap['archive']=[list(x) for x in f.nondominated(snap['archive'])];snap['archive_size']=len(snap['archive'])
                    r['convergence'][-1]['archive']=final;r['convergence'][-1]['archive_size']=len(final)
                    r['diagnostics']=dict(evaluations=40000,infeasible=0,infeasible_reasons={},crossovers=0,mutations=0)
                    state=copy.deepcopy(final)
                    if a in ('nsga2','moead'):state=[final[j%len(final)] for j in range(100)]
                    r['native_state']={'objectives':state}
                    if a=='moead':
                        weights=[(j/99,1-j/99) for j in range(100)]
                        r['native_state'].update(weights=[list(w) for w in weights],neighborhoods=[sorted(range(100),key=lambda j:(math.dist(w,weights[j]),j))[:20] for w in weights])
                        r['diagnostics'].update(candidates=39900,replacement_counts=[0]*39900,replacements=0,local_updates=39900,global_updates=0)
                    if a=='mpmoga':
                        r['native_state']['population_sizes']=[105,105]
                        r['diagnostics'].update(ast_crossovers=0,ast_no_crossovers=0,aus_attempts=0,mutation_accepted=0)
                    r['test_fixture']='合成接口夹具，不是外部算法真实运行结果'
                    r['payload_sha256']=f.digest({k:v for k,v in r.items() if k!='payload_sha256'})
                f.validate_record(r,f.source_identity(t,c));f.atomic_json(f.raw_path(t,c),r)
            # 禁止分析路径导入优化器；绘图同样只读保存数据。
            with patch('importlib.import_module',side_effect=AssertionError('禁止导入优化器')),patch.object(f,'run_task',side_effect=AssertionError('禁止优化')),patch.object(f,'MAKE_FIGURES',False):
                five=f.analyze_mode(c)
                c['analysis_algorithms']=list(f.REGISTRY);eight=f.analyze_mode(c)
                self.assertNotEqual(five,eight)
                for folder in (five,eight):
                    for n in ('run_metrics.csv','instance_summary.csv','overall_summary.csv','group_summary.csv','convergence_runs.csv','convergence_summary.csv','normalization_bounds.json','reference_fronts.json','evaluation_config.json'):
                        self.assertTrue((folder/n).exists(),n)
                f.raw_path(dict(algorithm='nsga2',instance=self.sample,seed=10),c).unlink()
                with self.assertRaises(ValueError):f.analyze_mode(c)

if __name__=='__main__':unittest.main()
