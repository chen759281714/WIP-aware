"""Core gate atomic fidelity, fallback, cooperation and read-only observation."""

import copy
import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import core_ps_variants as v
from experiments.diagnostics import diagnosis_subproblem_viability as d
from src.algorithms.wip_graph_dual_population import Individual, ShortageDiagnosis, WIPGraphDualPopulation
from tests.performance_trajectory_regression import population_state, signature
from tests.test_wip_graph_dual_population import fixture


def make_engine(core=True, observer=False, seed=0, fe=60):
    ops,buffers,seq,ms=fixture()
    engine=v.CorePSGateAlgorithm(ops,buffers,N=3,N_A=12,FE_max=fe,seed=seed,T_coop=2,
                                core_ps_enabled=core,gate_observer=observer)
    parent=Individual(seq,ms,operation_order=tuple(engine.encoder.ms_index_order))
    engine.evaluate(parent)
    return engine,parent


def test_restricted_equivalence():
    engine,parent=make_engine()
    for trial in range(100):
        a,b=random.Random(trial),random.Random(trial)
        formal=d.selection.exact_basic(engine,parent,a)
        restricted,meta=d.restricted_basic_mutation(engine,parent,set(engine.encoder.ms_index_order),b)
        if engine.genotype_key(formal)!=engine.genotype_key(restricted) or a.getstate()!=b.getstate():
            raise AssertionError("D=O child/RNG equality failed")
        d.assert_restricted(parent,restricted,set(engine.encoder.ms_index_order),meta)


class CoreGateTests(unittest.TestCase):
    def test_H_restricted_target_always_in_H(self):
        engine,parent=make_engine(observer=True)
        diagnosis=engine.diagnose_shortage(parent)
        high=engine.high_influence_set(parent,diagnosis)
        for _ in range(30):
            with patch.object(engine.rng,"random",return_value=.9):
                child=engine.vary_shortage(parent,[])
            meta=engine._gate_pending[id(child)]
            self.assertTrue(meta["restricted"])
            self.assertIn(tuple(meta["operation"]),high)

    def test_global_mutation_branch_never_calls_core(self):
        engine,parent=make_engine(observer=True)
        with patch.object(engine.rng,"random",return_value=0), patch.object(engine,"_core_child",side_effect=AssertionError("Core on mutation")):
            child=engine.vary_shortage(parent,[])
        self.assertEqual(engine._gate_pending[id(child)]["branch"],"mutation")
        self.assertFalse(engine._gate_pending[id(child)]["restricted"])

    def fallback(self,engine,parent,reason):
        before=engine.n_evaluations
        with patch.object(engine.rng,"random",return_value=.9):
            child=engine.vary_shortage(parent,[],archive_assisted=True)
        self.assertEqual(engine.n_evaluations,before)
        self.assertEqual(engine._gate_pending[id(child)]["fallback_reason"],reason)
        self.assertIsNone(child.shortage)
        engine.evaluate(child)
        self.assertEqual(engine.n_evaluations,before+1)

    def test_empty_H_fallback(self):
        engine,parent=make_engine(observer=True)
        with patch.object(engine,"high_influence_set",return_value=set()):
            self.fallback(engine,parent,"H_empty")

    def test_zero_shortage_fallback(self):
        engine,parent=make_engine(observer=True)
        parent.shortage=0
        with patch.object(engine,"diagnose_shortage",side_effect=AssertionError("No diagnosis at zero")):
            self.fallback(engine,parent,"zero_shortage")

    def test_invalid_diagnosis_fallback(self):
        engine,parent=make_engine(observer=True)
        diagnosis=engine.diagnose_shortage(parent)
        diagnosis.unanchored_intervals=1
        with patch.object(engine,"diagnose_shortage",return_value=diagnosis):
            self.fallback(engine,parent,"diagnosis_invariant")

    def test_invalid_phi_mass_fallback(self):
        engine,parent=make_engine(observer=True)
        diagnosis=engine.diagnose_shortage(parent)
        diagnosis.phi[next(iter(diagnosis.phi))]+=1
        with patch.object(engine,"diagnose_shortage",return_value=diagnosis):
            self.fallback(engine,parent,"phi_mass_invalid")

    def test_no_action_fallback(self):
        engine,parent=make_engine(observer=True)
        child=Individual(parent.os_seq[:],parent.ms_map.copy(),operation_order=parent.operation_order)
        with patch.object(d,"restricted_basic_mutation",return_value=(child,{"exhausted":True})):
            self.fallback(engine,parent,"no_action")

    def test_one_child_one_FE_no_intermediate_decode(self):
        engine,parent=make_engine()
        before=engine.n_evaluations
        original=engine.decoder.decode
        with patch.object(engine.rng,"random",return_value=.9), patch.object(engine.decoder,"decode",wraps=original) as decoded:
            child=engine.vary_shortage(parent,[])
            self.assertEqual(decoded.call_count,0)
            engine.evaluate(child)
            self.assertEqual(decoded.call_count,1)
        self.assertEqual(engine.n_evaluations,before+1)

    def test_no_SIS_or_PS_archive_guide(self):
        engine,parent=make_engine()
        with patch.object(engine,"choose_guide",side_effect=AssertionError("SIS guide")), \
             patch.object(engine,"choose_archive_guide_for_shortage",side_effect=AssertionError("PS archive guide")), \
             patch.object(engine,"archive_guide",side_effect=AssertionError("Archive accessed by PS")), \
             patch.object(engine.rng,"random",return_value=.9):
            engine.vary_shortage(parent,[parent],archive_assisted=True)
        self.assertEqual(engine.diagnostics["cooperation"]["ps_probes"],0)
        self.assertFalse(engine.ps_archive_guidance_enabled)

    def test_PM_generation_function_unchanged(self):
        engine,_=make_engine()
        self.assertIs(engine._generate_pm_child.__func__,WIPGraphDualPopulation._generate_pm_child)
        self.assertIs(engine._vary_makespan_specialized.__func__,WIPGraphDualPopulation._vary_makespan_specialized)
        self.assertTrue(engine.pm_archive_guidance_enabled)

    def test_PM_archive_cooperation_exact_between_variants(self):
        a,_=make_engine(core=False)
        b,_=make_engine(core=True)
        a.initialize();b.initialize()
        with patch.object(a.rng,"random",return_value=.9),patch.object(b.rng,"random",return_value=.9):
            sa,ca,oa=a._generate_pm_child(True)
            sb,cb,ob=b._generate_pm_child(True)
        self.assertEqual(a.genotype_key(sa),b.genotype_key(sb))
        self.assertEqual(a.genotype_key(ca),b.genotype_key(cb))
        self.assertEqual(oa,ob)
        self.assertEqual(a.rng.getstate(),b.rng.getstate())
        self.assertEqual(a.diagnostics["cooperation"],b.diagnostics["cooperation"])
        self.assertEqual(a.diagnostics["cooperation"]["pm_attempts"],1)

    def test_PS_cooperation_does_not_change_semantics(self):
        a,parent=make_engine()
        b,other=make_engine()
        with patch.object(a.rng,"random",return_value=.9),patch.object(b.rng,"random",return_value=.9):
            ca=a.vary_shortage(parent,[parent],False)
            cb=b.vary_shortage(other,[],True)
        self.assertEqual(a.genotype_key(ca),b.genotype_key(cb))
        self.assertEqual(a.rng.getstate(),b.rng.getstate())
        self.assertEqual(a._last_ps_guide_source,"self")

    def test_replacement_exact(self):
        a,parent=make_engine()
        ops,buffers,_,_=fixture()
        b=v.BasicPSReference(ops,buffers,N=3,N_A=12,FE_max=60,seed=0,T_coop=2)
        child=d.selection.exact_basic(a,parent,random.Random(5))
        d.offline_decode(a.decoder,child)
        pa,pb=[parent.copy()],[parent.copy()]
        self.assertEqual(a.similarity_replace(pa,child,"shortage"),b.similarity_replace(pb,child,"shortage"))
        self.assertEqual(population_state(pa),population_state(pb))

    def test_archive_update_exact(self):
        a,parent=make_engine(observer=True)
        b,_=make_engine(core=False)
        children=[]
        for i in range(8):
            child=d.selection.exact_basic(a,parent,random.Random(i))
            d.offline_decode(a.decoder,child)
            children.append(child)
        a.update_archive(children);b.update_archive(children)
        self.assertEqual(population_state(a.A),population_state(b.A))
        self.assertEqual(a.rng.getstate(),b.rng.getstate())

    def test_D_O_equivalence(self):
        test_restricted_equivalence()

    def test_frozen_240_state_H_and_mass_minimality(self):
        paths=sorted((d.ROOT/"cache/verified").glob("*/seed_*.json"))
        if len(paths)!=6:
            self.skipTest("Frozen DISVA inputs required; formal fidelity gate never skips reconstruction")
        count=0
        for path in paths:
            record=json.loads(path.read_text())
            for state in record["states"]:
                phi={tuple(op):value for op,value in state["phi"]}
                ind=d.r.prefix_individual(state["genome"]);ind.shortage=state["S_seed"]
                ordered=sorted(phi,key=lambda op:(-phi[op],op))
                diagnosis=ShortageDiagnosis([],phi,set(),[],0)
                engine=object.__new__(WIPGraphDualPopulation);engine.rho=.8
                high=engine.high_influence_set(ind,diagnosis)
                self.assertEqual(high,{tuple(op) for op in state["H"]})
                self.assertGreaterEqual(sum(phi[op] for op in ordered[:len(high)])/ind.shortage,.8)
                self.assertLess(sum(phi[op] for op in ordered[:len(high)-1])/ind.shortage,.8)
                self.assertAlmostEqual(sum(phi.values()),ind.shortage,places=8)
                count+=1
        self.assertEqual(count,240)

    def test_default_variant_disabled_observer_off(self):
        ops,buffers,_,_=fixture()
        engine=v.CorePSGateAlgorithm(ops,buffers,N=3,N_A=12,FE_max=60,seed=0)
        self.assertFalse(engine.core_ps_enabled)
        self.assertFalse(engine.gate_observer)
        self.assertEqual(engine.algorithm_variant,"basic_ps")

    def test_observer_isolation_full_small_trajectory(self):
        ops,buffers,_,_=fixture()
        for core in (False,True):
            a=v.CorePSGateAlgorithm(ops,buffers,N=3,N_A=12,FE_max=60,seed=2,T_coop=2,core_ps_enabled=core)
            b=v.CorePSGateAlgorithm(ops,buffers,N=3,N_A=12,FE_max=60,seed=2,T_coop=2,core_ps_enabled=core,gate_observer=True)
            a.run();b.run()
            self.assertEqual(d.local.formal_state_signature(a),d.local.formal_state_signature(b))
            self.assertEqual(a._first_source_by_genotype,b._first_source_by_genotype)
            self.assertEqual(len(b.gate_events),27)
            self.assertEqual(len(a.gate_events),0)

    def test_baseline_reference_exact(self):
        ops,buffers,_,_=fixture()
        a=v.BasicPSReference(ops,buffers,N=3,N_A=12,FE_max=60,seed=3,T_coop=2)
        b=v.CorePSGateAlgorithm(ops,buffers,N=3,N_A=12,FE_max=60,seed=3,T_coop=2)
        a.run();b.run()
        self.assertEqual(d.local.formal_state_signature(a),d.local.formal_state_signature(b))

    def test_instance_aggregation_not_seed_pseudoreplication(self):
        from experiments import run_core_ps_integration_gate as g
        rows=[{"instance":g.PRIMARY[0],"seed":i,"size_group":"S1","profile":"P1","buffer_level":"B1",
               "set":"primary_unseen","metric":"nhv","basic_ps":1,"core_ps":2+i,"benefit":1+i} for i in range(3)]
        means=g.instance_pairs(rows)
        self.assertEqual(len(means),1)
        self.assertEqual(means[0]["benefit"],2)
        self.assertEqual(means[0]["n_seeds"],3)


if __name__=="__main__":
    unittest.main()
