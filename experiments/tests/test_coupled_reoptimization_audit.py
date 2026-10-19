"""CRNA fidelity and search semantics; no formal sources modified."""

import copy
import random
import unittest
from unittest.mock import patch

from experiments.diagnostics import coupled_reoptimization_audit as c
from experiments.diagnostics import diagnosis_subproblem_viability as d
from tests.test_wip_graph_dual_population import fixture


def setup_case():
    ops,buffers,seq,ms=fixture()
    engine=c.CaptureCorePS(ops,buffers,N=3,N_A=12,FE_max=60,seed=7,T_coop=2,
                           core_ps_enabled=True,gate_observer=True,capture=True,checkpoints=(1,))
    parent=c.Individual(seq,ms,operation_order=tuple(engine.encoder.ms_index_order))
    engine.evaluate(parent)
    engine._core_child(parent)
    state=engine.crna_states[0]
    state.update(instance="fixture",seed=7)
    return engine,parent,state


class CRNATests(unittest.TestCase):
    def scripted(self,policy,values,cmax_deltas=None):
        engine,parent,state=setup_case()
        scores=iter(values)
        costs=iter(cmax_deltas or [0]*8)
        seen=[]
        def decode(ind):
            ind.shortage=parent.shortage+next(scores)
            ind.makespan=parent.makespan+next(costs)
            return ind
        def propose(ind,allowed,rng):
            seen.append(engine.genotype_key(ind))
            return d.restricted_basic_mutation(engine,ind,allowed,rng)
        result=c.episode(engine,state,policy,0,decode=decode,propose=propose)
        return engine,parent,state,seen,result

    def test_01_seed_objectives_equal(self):
        engine,parent,state=setup_case()
        rebuilt=c.reconstruct(engine,state)
        self.assertEqual((rebuilt.shortage,rebuilt.makespan),(parent.shortage,parent.makespan))

    def test_02_provenance_equal_and_tamper_detected(self):
        engine,parent,state=setup_case()
        self.assertEqual(c.reconstruct(engine,state).provenance,parent.provenance)
        state=copy.deepcopy(state);state["parent"].provenance.events.clear()
        with self.assertRaisesRegex(AssertionError,"provenance"):
            c.reconstruct(engine,state)

    def test_03_phi_reconstruction_and_tamper_detected(self):
        engine,parent,state=setup_case()
        c.reconstruct(engine,state)
        state=copy.deepcopy(state)
        state["diagnosis"].phi[next(iter(state["diagnosis"].phi))]+=1
        with self.assertRaisesRegex(AssertionError,"phi/H"):
            c.reconstruct(engine,state)

    def test_04_H_reconstruction(self):
        engine,parent,state=setup_case()
        state={**state,"H":frozenset()}
        with self.assertRaisesRegex(AssertionError,"phi/H"):
            c.reconstruct(engine,state)

    def test_05_H_minimality(self):
        engine,parent,state=setup_case()
        phi=state["diagnosis"].phi
        order=sorted(phi,key=lambda op:(-phi[op],op))
        n=len(state["H"])
        self.assertGreaterEqual(sum(phi[op] for op in order[:n])/parent.shortage,.8)
        self.assertLess(sum(phi[op] for op in order[:n-1])/parent.shortage,.8)

    def test_06_H_frozen(self):
        engine,parent,state=setup_case()
        before=copy.deepcopy(state["H"])
        with patch.object(engine,"diagnose_shortage",side_effect=AssertionError("rediagnose")),patch.object(engine,"high_influence_set",side_effect=AssertionError("new H")):
            c.episode(engine,state,"H-Sequential",0)
        self.assertEqual(state["H"],before)

    def test_07_H_independent_seed_parent(self):
        engine,parent,state,seen,_=self.scripted("H-Independent",[-1,-2,-3,-4,-5,-6,-7,-8])
        self.assertEqual(seen,[engine.genotype_key(parent)]*8)

    def test_08_global_independent_seed_parent(self):
        engine,parent,state,seen,_=self.scripted("Global-Independent",[-1,-2,-3,-4,-5,-6,-7,-8])
        self.assertEqual(seen,[engine.genotype_key(parent)]*8)

    def test_09_accepted_child_next_incumbent(self):
        engine,parent,state,seen,(summary,steps,final)=self.scripted("H-Sequential",[-1,-2,-3,-4,-5,-6,-7,-8])
        for a,b in zip(steps,steps[1:]):
            self.assertEqual(a["proposal_hash"],b["proposal_parent_hash"])
        self.assertEqual(summary["accepted_moves"],8)

    def test_10_rejection_keeps_incumbent(self):
        engine,parent,state,seen,(summary,steps,final)=self.scripted("H-Sequential",[-1,1,2,3,4,5,6,7])
        self.assertEqual(summary["accepted_moves"],1)
        self.assertTrue(all(s["incumbent_hash"]==steps[0]["proposal_hash"] for s in steps))

    def test_11_global_sequential_strict_semantics(self):
        engine,parent,state,seen,(summary,steps,final)=self.scripted("Global-Sequential",[-1,-1,-2,0,2,-3,-3,4])
        self.assertEqual(summary["accepted_moves"],3)
        self.assertEqual([s["accepted"] for s in steps],[True,False,True,False,False,True,False,False])

    def test_12_no_equal_shortage_acceptance(self):
        engine,parent,state,seen,(summary,steps,final)=self.scripted("H-Sequential",[0]*8,[-100]*8)
        self.assertEqual(summary["accepted_moves"],0)
        self.assertEqual(final["genome"],c.r.prefix_genome(parent))

    def test_13_non_H_MS_unchanged(self):
        engine,parent,state=setup_case()
        high=state["H"]
        for policy in c.POLICIES[:2]:
            summary,steps,final=c.episode(engine,state,policy,2)
            child=c.r.prefix_individual(final["genome"])
            self.assertTrue(all(child.ms_map[op]==parent.ms_map[op] for op in parent.ms_map if op not in high))

    def test_14_non_H_OS_projection_unchanged(self):
        engine,parent,state=setup_case()
        for policy in c.POLICIES[:2]:
            summary,steps,final=c.episode(engine,state,policy,3)
            child=c.r.prefix_individual(final["genome"])
            self.assertEqual([op for op in child.os_seq if op not in state["H"]],
                             [op for op in parent.os_seq if op not in state["H"]])

    def test_15_D_O_exact_formal_RNG(self):
        engine,parent,state=setup_case()
        for trial in range(100):
            a,b=random.Random(trial),random.Random(trial)
            formal=d.selection.exact_basic(engine,parent,a)
            restricted,meta=d.restricted_basic_mutation(engine,parent,set(parent.os_seq),b)
            self.assertEqual(engine.genotype_key(formal),engine.genotype_key(restricted))
            self.assertEqual(a.getstate(),b.getstate())

    def test_16_no_rediagnosis(self):
        engine,parent,state=setup_case()
        with patch.object(engine,"diagnose_shortage",side_effect=AssertionError("diagnosis forbidden")):
            for policy in c.POLICIES:
                c.episode(engine,state,policy,1)

    def test_17_population_archive_unchanged(self):
        engine,parent,state=setup_case()
        engine.P_M=[parent];engine.P_S=[parent];engine.A=[parent]
        before=c.local.formal_state_signature(engine)
        for policy in c.POLICIES:
            c.episode(engine,state,policy,0)
        self.assertEqual(c.local.formal_state_signature(engine),before)

    def test_18_formal_rng_unchanged(self):
        engine,parent,state=setup_case()
        before=engine.rng.getstate()
        for policy in c.POLICIES:
            c.episode(engine,state,policy,0)
        self.assertEqual(engine.rng.getstate(),before)

    def test_19_one_proposal_one_decode(self):
        engine,parent,state=setup_case()
        original=engine.decoder.decode
        with patch.object(engine.decoder,"decode",wraps=original) as decode:
            summary,steps,final=c.episode(engine,state,"H-Sequential",0)
        self.assertEqual(decode.call_count,len(steps))
        self.assertEqual(summary["budget_used"],len(steps))

    def test_20_budget8(self):
        engine,parent,state=setup_case()
        for policy in c.POLICIES:
            summary,steps,final=c.episode(engine,state,policy,0)
            self.assertEqual(summary["budget_used"],8)

    def test_21_exhaustion_no_global_fallback(self):
        engine,parent,state=setup_case();state={**state,"H":frozenset()}
        with patch.object(engine,"basic_mutation",side_effect=AssertionError("global fallback")):
            summary,steps,final=c.episode(engine,state,"H-Sequential",0)
        self.assertEqual(summary["budget_used"],0)
        self.assertEqual(summary["exhausted_step"],1)

    def test_22_makespan_not_acceptance(self):
        engine,parent,state,seen,(summary,steps,final)=self.scripted("H-Sequential",[-1]*8,[1000]*8)
        self.assertEqual(summary["accepted_moves"],1)
        self.assertEqual(summary["Cmax_delta"],1000)

    def test_23_archive_never_enters_search(self):
        engine,parent,state=setup_case()
        a=c.episode(engine,state,"H-Sequential",0)
        state={**state,"archive":[{"shortage":-100,"Cmax":-100}]}
        b=c.episode(engine,state,"H-Sequential",0)
        self.assertEqual(a[1],b[1]);self.assertEqual(a[2],b[2])

    def test_24_identical_base_trial_seed_all_policies(self):
        engine,parent,state=setup_case()
        a=c.episode(engine,state,"H-Independent",0)[1][0]
        b=c.episode(engine,state,"H-Sequential",0)[1][0]
        self.assertEqual(a["proposal_genome"],b["proposal_genome"])
        a=c.episode(engine,state,"Global-Independent",0)[1][0]
        b=c.episode(engine,state,"Global-Sequential",0)[1][0]
        self.assertEqual(a["proposal_genome"],b["proposal_genome"])

    def test_25_capture_default_off(self):
        ops,buffers,seq,ms=fixture()
        engine=c.CaptureCorePS(ops,buffers,N=3,N_A=12,FE_max=60,seed=7,core_ps_enabled=True)
        self.assertFalse(engine.crna_capture)

    def test_26_observer_off_on_complete_state(self):
        ops,buffers,seq,ms=fixture()
        engines=[]
        for enabled in (False,True):
            engine=c.CaptureCorePS(ops,buffers,N=3,N_A=12,FE_max=60,seed=7,T_coop=2,
                                   core_ps_enabled=True,gate_observer=True,capture=enabled,checkpoints=(10,20,30))
            engine.run();engines.append(engine)
        self.assertEqual(c.final_signature(engines[0]),c.final_signature(engines[1]))
        self.assertEqual(engines[0].gate_events,engines[1].gate_events)
        self.assertEqual(len(engines[1].crna_states),3)

    def test_27_distances_match_frozen_definition(self):
        engine,parent,state=setup_case()
        for trial in range(12):
            child=d.selection.exact_basic(engine,parent,random.Random(trial))
            result=c.distances(parent,child,state["H"])
            self.assertEqual(result["d_total"],engine.structural_distance(parent,child))

    def test_28_independent_earliest_tie(self):
        engine,parent,state,seen,(summary,steps,final)=self.scripted("H-Independent",[-1]*8,[-1,-8,-8,-8,-8,-8,-8,-8])
        self.assertEqual(final["genome"],steps[0]["proposal_genome"])
        self.assertEqual(final["Cmax"],parent.makespan-1)

    def test_29_global_observed_choice_exact(self):
        engine,parent,state=setup_case()
        for trial in range(100):
            a,b=random.Random(trial),random.Random(trial)
            formal=d.selection.exact_basic(engine,parent,a)
            observed,meta=c.global_proposal(engine,parent,b)
            self.assertEqual(engine.genotype_key(formal),engine.genotype_key(observed))
            self.assertEqual(a.getstate(),b.getstate())
            d.assert_restricted(parent,observed,set(parent.os_seq),meta)

    def test_30_reconstruction_restores_all_counters(self):
        engine,parent,state=setup_case()
        before=c.local.formal_state_signature(engine)
        c.reconstruct(engine,state)
        self.assertEqual(c.local.formal_state_signature(engine),before)

    def test_31_first_specialized_capture_no_outcome_filter(self):
        ops,buffers,seq,ms=fixture()
        engine=c.CaptureCorePS(ops,buffers,N=3,N_A=12,FE_max=60,seed=7,T_coop=2,
                               core_ps_enabled=True,gate_observer=True,capture=True,checkpoints=(10,20))
        engine.run()
        specialized=[e for e in engine.gate_events if e["branch"]=="specialized"]
        for state in engine.crna_states:
            first=next(e for e in specialized if e["fe_index"]>state["checkpoint"])
            self.assertEqual(state["child_FE"],first["fe_index"])


if __name__=="__main__":
    unittest.main()
