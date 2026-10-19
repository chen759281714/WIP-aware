import copy
import hashlib
import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from src.algorithms.wip_graph_dual_population import HLNSDualPopulation
from src.algorithms.wip_graph_dual_population import restricted_atomic_fallback
from tests.hlns_test_support import TEST_ALGORITHMS, make_hlns_algorithm, make
from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from tests.performance_trajectory_regression import population_state


SMOKE_RUNS=[]


def smoke(engine,**kwargs):
    if engine.FE_max>60:
        raise AssertionError("Correctness smoke must never become a benchmark run")
    before=engine.n_evaluations
    result=engine.run(**kwargs)
    SMOKE_RUNS.append({"variant":engine.algorithm_variant,"FE_max":engine.FE_max,
                       "actual_FE":engine.n_evaluations-before,"instance":"unit_fixture"})
    return result


class HLNSIntegrationTests(unittest.TestCase):
    def test_joint_no_inplace_mutation(self):
        e,p=make();e.evaluate(p);before=copy.deepcopy(p)
        child=e.joint_repair(p,set(p.os_seq),random.Random(3))
        self.assertEqual(p,before);self.assertIsNot(child,p)
        self.assertIsNot(child.os_seq,p.os_seq);self.assertIsNot(child.ms_map,p.ms_map)

    def test_fresh_evaluation_fields(self):
        e,p=make();e.evaluate(p);child=e.joint_repair(p,set(p.os_seq),random.Random(3))
        for attr in ("makespan","shortage","schedule","buffer_trace","stats","provenance"):
            self.assertIsNone(getattr(child,attr))

    def test_multiple_OS_and_all_MS_can_change(self):
        class ZeroRNG:
            def random(self):return 0.0
            def choice(self,options):return options[0]
        e,p=make();child=e.joint_repair(p,set(p.os_seq),ZeroRNG())
        self.assertEqual(sum(child.ms_map[op]!=p.ms_map[op] for op in p.ms_map),len(p.ms_map))
        before={op:i for i,op in enumerate(p.os_seq)};after={op:i for i,op in enumerate(child.os_seq)}
        changed=sum((before[a]<before[b])!=(after[a]<after[b]) for i,a in enumerate(p.os_seq) for b in p.os_seq[i+1:] if a[0]!=b[0])
        self.assertGreaterEqual(changed,2)

    def test_no_decode_in_repair_and_real_decoder_accepts(self):
        e,p=make()
        with patch.object(e.decoder,"decode",side_effect=AssertionError("intermediate decode")):
            child=e.joint_repair(p,set(p.os_seq),random.Random(5))
        e.evaluate(child);self.assertIsNotNone(child.provenance)

    def test_one_PS_generation_one_evaluation(self):
        e,p=make();e.evaluate(p);before=e.n_evaluations
        original=e.decoder.decode
        with patch.object(e.decoder,"decode",wraps=original) as decoded,patch.object(e.rng,"random",return_value=.9):
            child=e.vary_shortage(p)
            self.assertEqual(e.n_evaluations,before);self.assertEqual(decoded.call_count,0)
            e.evaluate(child)
            self.assertEqual(decoded.call_count,1);self.assertEqual(e.n_evaluations,before+1)

    def test_global_branch_kept(self):
        e,p=make()
        with patch.object(e.rng,"random",return_value=0) as draw,patch.object(e,"joint_repair",side_effect=AssertionError("HLNS")), \
             patch.object(e,"diagnose_shortage",side_effect=AssertionError("diagnosis")):
            child=e.vary_shortage(p)
        self.assertEqual(draw.call_count,1);self.assertEqual(e.hlns_statistics["hlns_calls"],0)
        self.assertEqual(e._last_ps_move_kind,"basic_mutation")

    def test_same_genotype_single_restricted_fallback(self):
        e,p=make();e.evaluate(p)
        same=Individual(p.os_seq[:],p.ms_map.copy(),operation_order=p.operation_order)
        with patch.object(e.rng,"random",return_value=.9),patch.object(e,"joint_repair",return_value=same), \
             patch("src.algorithms.wip_graph_dual_population.restricted_atomic_fallback",wraps=restricted_atomic_fallback) as fallback:
            child=e.vary_shortage(p)
        self.assertEqual(fallback.call_count,1);self.assertNotEqual(e.genotype_key(child),e.genotype_key(p))
        self.assertEqual(e.hlns_statistics["hlns_normal_children"],0)

    def test_full_exhausted_global_final_fallback(self):
        ops={"A":[{"machines":{"m":1},"buffer_in":None,"buffer_out":None}]}
        e=make_hlns_algorithm(ops,{},N=1,N_A=2,FE_max=2,seed=0)
        p=Individual([("A",0)],{("A",0):"m"},shortage=1,operation_order=(("A",0),))
        with patch.object(e.rng,"random",return_value=.9),patch.object(e,"select_destroy_set",return_value=set(p.os_seq)):
            child=e.vary_shortage(p)
        self.assertEqual(e.genotype_key(child),e.genotype_key(p))
        self.assertEqual(e.hlns_statistics["hlns_global_fallbacks"],1)
        self.assertEqual(e.hlns_statistics["hlns_unchanged_children"],1)

    @unittest.skip("Retired internal OS-repair ablation")
    def test_MS_only_exhausted_never_reenables_OS(self):
        ops={j:[{"machines":{"m":1},"buffer_in":None,"buffer_out":None}] for j in ("A","B")}
        e=make_hlns_algorithm(ops,{},"HLNS_WO_OS_REPAIR",N=1,N_A=2,FE_max=2,seed=0)
        seq=[("A",0),("B",0)];p=Individual(seq,{op:"m" for op in seq},shortage=1,operation_order=tuple(seq))
        with patch.object(e.rng,"random",return_value=.9),patch.object(e,"select_destroy_set",return_value=set(seq)), \
             patch.object(e,"basic_mutation",side_effect=AssertionError("global/OS fallback")):
            child=e.vary_shortage(p)
        self.assertEqual(child.os_seq,seq);self.assertEqual(child.ms_map,p.ms_map)
        self.assertEqual(e.hlns_statistics["hlns_global_fallbacks"],0)

    @unittest.skip("Retired internal MS-repair ablation")
    def test_OS_only_exhausted_never_reenables_MS(self):
        ops={"A":[{"machines":{"m":1,"n":2},"buffer_in":None,"buffer_out":None} for _ in range(2)]}
        e=make_hlns_algorithm(ops,{},"HLNS_WO_MS_REPAIR",N=1,N_A=2,FE_max=2,seed=0)
        seq=[("A",0),("A",1)];p=Individual(seq,{op:"m" for op in seq},shortage=1,operation_order=tuple(seq))
        with patch.object(e.rng,"random",return_value=.9),patch.object(e,"select_destroy_set",return_value=set(seq)), \
             patch.object(e,"basic_mutation",side_effect=AssertionError("global/MS fallback")):
            child=e.vary_shortage(p)
        self.assertEqual(child.os_seq,seq);self.assertEqual(child.ms_map,p.ms_map)

    def test_no_parent_acceptance_gate_and_stats_distinction(self):
        e,p=make();p.shortage=0
        child=e.basic_mutation(p);e._finish_hlns_child(p,child);e.evaluate(child)
        nearest=child.copy();nearest.shortage=child.shortage+1
        population=[nearest]
        self.assertTrue(e.similarity_replace(population,child,"shortage"))
        self.assertIs(population[0],child)
        self.assertEqual(e.hlns_statistics["hlns_parent_shortage_improvement_count"],0)
        self.assertEqual(e.hlns_statistics["hlns_population_replacement_count"],1)

    def test_rejected_PS_child_still_enters_archive(self):
        e,p=make();p.shortage=5;p.makespan=10
        child=p.copy();child.ms_map=child.ms_map.copy();op=p.os_seq[0]
        child.ms_map[op]=next(m for m in e.operations[op[0]][op[1]]["machines"] if m!=p.ms_map[op])
        child.shortage=6;child.makespan=5
        nearest=child.copy();nearest.shortage=4;nearest.makespan=100
        e.A=[p];population=[nearest]
        self.assertFalse(e.similarity_replace(population,child,"shortage"))
        e.update_archive([child]);self.assertIn(child,e.A)

    def test_PM_seed_pipeline_inherited(self):
        self.assertIs(HLNSDualPopulation._generate_pm_child,WIPGraphDualPopulation._generate_pm_child)
        self.assertIs(HLNSDualPopulation.run,WIPGraphDualPopulation.run)
        self.assertIs(HLNSDualPopulation.update_archive,WIPGraphDualPopulation.update_archive)
        self.assertIs(HLNSDualPopulation.structural_distance,WIPGraphDualPopulation.structural_distance)

    def test_PM_MGS_and_AC_match_historical(self):
        e,p=make();e.evaluate(p);e.P_M=[p];e.A=[p]
        old=WIPGraphDualPopulation(e.operations,e.buffers,variant="HLNS_FULL",N=3,N_A=12,FE_max=18,seed=0)
        old.evaluate(p.copy());old.P_M=[p.copy()];old.A=[p.copy()]
        a,ca,sa=e._generate_pm_child(True);b,cb,sb=old._generate_pm_child(True)
        self.assertEqual(e.genotype_key(a),old.genotype_key(b));self.assertEqual(e.genotype_key(ca),old.genotype_key(cb))
        self.assertEqual(sa,sb);self.assertEqual(e.rng.getstate(),old.rng.getstate())

    def test_no_PS_guides_or_SIS_direction(self):
        e,p=make();e.evaluate(p)
        for name in ("choose_guide", "choose_archive_guide_for_shortage", "_build_relink_units"):
            self.assertFalse(hasattr(e, name))
        with patch.object(e.rng,"random",return_value=.9), \
             patch.object(e,"archive_guide",side_effect=AssertionError("PS archive")):
            e.vary_shortage(p)

    def test_all_variants_reproducible_and_observer_neutral(self):
        for cls in TEST_ALGORITHMS:
            variant=cls.algorithm_variant.fget(None)
            with self.subTest(variant=variant):
                engines=[make(variant,observe=observe,seed=3)[0] for observe in (True,True,False)]
                for e in engines:
                    smoke(e)
                    self.assertEqual(e.n_evaluations,18);self.assertEqual((len(e.P_M),len(e.P_S)),(3,3))
                    self.assertLessEqual(len(e.A),12);self.assertTrue(e.A)
                    self.assertNotIn("relinking",e.diagnostics)
                    self.assertTrue(all(not e.dominates(x,y) for x in e.A for y in e.A if x is not y))
                    self.assertEqual(sum(v["generated"] for v in e.diagnostics["ps_moves"].values()),6)
                state=lambda x:(population_state(x.P_M),population_state(x.P_S),population_state(x.A),x.rng.getstate())
                self.assertEqual(state(engines[0]),state(engines[1]));self.assertEqual(state(engines[0]),state(engines[2]))

    def test_archive_batch_cadence(self):
        e,_=make(fe=24)
        with patch.object(e,"update_archive",wraps=e.update_archive) as updated, \
             patch.object(e.decoder,"decode",wraps=e.decoder.decode) as decoded:
            smoke(e)
        self.assertEqual(updated.call_count,4);self.assertEqual(decoded.call_count,24)

    def test_odd_FE_stop_no_overshoot(self):
        e,_=make(fe=13);smoke(e);self.assertEqual(e.n_evaluations,13)
        with self.assertRaises(RuntimeError):e.evaluate(e.random_individual())

    @unittest.skip("Historical BasicPSReference subclasses the replaced Full; not an H-LNS reference")
    def test_basic_PS_new_framework_matches_reference(self):
        from experiments.core_ps_variants import BasicPSReference
        e,_=make("HLNS_WO_HLNS",seed=1)
        old=BasicPSReference(e.operations,e.buffers,N=3,N_A=12,FE_max=18,seed=1)
        smoke(e);smoke(old)
        self.assertEqual((population_state(e.P_M),population_state(e.P_S),population_state(e.A)),
                         (population_state(old.P_M),population_state(old.P_S),population_state(old.A)))
        self.assertEqual(e.rng.getstate(),old.rng.getstate())

    @unittest.skip("Retired internal parent-bias ablation")
    def test_bias_flag_passed_without_q_one_hack(self):
        e,p=make("HLNS_WO_PARENT_BIAS");high=set(p.os_seq)
        from src.algorithms import wip_graph_dual_population as mod
        with patch.object(mod,"biased_topological_os_repair",wraps=mod.biased_topological_os_repair) as os_repair, \
             patch.object(mod,"biased_machine_repair",wraps=mod.biased_machine_repair) as ms_repair:
            e.joint_repair(p,high,random.Random(2))
        self.assertFalse(os_repair.call_args.kwargs["parent_bias"]);self.assertFalse(ms_repair.call_args.kwargs["parent_bias"])
        self.assertLess(os_repair.call_args.args[2],1)

    def test_no_speed_evidence_in_joint_repair(self):
        e,p=make();other,_=make()
        for job,ops in other.operations.items():
            for op in ops:
                op["machines"]={m:100000-i for i,m in enumerate(op["machines"])}
        # Changing processing values must not affect candidate generation.
        a=e.joint_repair(p,set(p.os_seq),random.Random(2));b=other.joint_repair(p,set(p.os_seq),random.Random(2))
        self.assertEqual(e.genotype_key(a),other.genotype_key(b))

    def test_fixed_skeleton_and_MS(self):
        e,p=make();high=set(p.os_seq[2:4])
        for i in range(20):
            child=e.joint_repair(p,high,random.Random(i))
            self.assertEqual([op for op in child.os_seq if op not in high],[op for op in p.os_seq if op not in high])
            self.assertTrue(all(child.ms_map[op]==p.ms_map[op] for op in p.ms_map if op not in high))

    @unittest.skip("Historical Full source hash intentionally changed by the authorized H-LNS replacement")
    def test_history_hashes_before_after(self):
        root=Path(__file__).resolve().parents[1]
        manifest=json.loads((root/"docs/hlns_historical_sha256.json").read_text())
        self.assertGreater(len(manifest),100)
        for relative,expected in manifest.items():
            with self.subTest(file=relative):
                self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(),expected)


if __name__=="__main__":unittest.main()
