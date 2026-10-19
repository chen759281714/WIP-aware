import random
import unittest
from unittest.mock import patch

from tests.hlns_test_support import TEST_ALGORITHMS, make_hlns_algorithm, make
from src.algorithms.wip_graph_dual_population import HLNSDiagnosisError
from src.algorithms.wip_graph_dual_population import Individual
from tests.hlns_test_support import fixture


class HLNSVariantTests(unittest.TestCase):
    def test_all_four(self):
        self.assertEqual(len(TEST_ALGORITHMS),4)
        for cls in TEST_ALGORITHMS:
            name=cls.algorithm_variant.fget(None)
            engine,parent=make(name)
            self.assertEqual(engine.algorithm_variant,name)
            self.assertEqual((engine.rho,engine.p_mut,engine.T_coop),(.8,.1,10))
            self.assertFalse(engine.ps_archive_guidance_enabled)

    def test_unknown_raises(self):
        for name in ("Full","unknown",None,4):
            with self.assertRaises(ValueError):make(name)

    def test_identity_read_only_and_contradictions(self):
        e,_=make()
        with self.assertRaises(AttributeError):e.algorithm_variant="Full"
        ops,buffers,_,_=fixture()
        for cls in TEST_ALGORITHMS:
            with self.assertRaises(ValueError):cls(ops,buffers,variant="unknown")

    def test_H_matches_formal_and_D_is_all_H(self):
        e,p=make();e.evaluate(p)
        d=e.diagnose_shortage(p);expected=e.high_influence_set(p,d)
        self.assertEqual(e.select_destroy_set(p),expected)

    @unittest.skip("Retired internal diagnosis ablation, not one of the three paper variants")
    def test_random_D_size_and_uniform_population(self):
        e,p=make("HLNS_WO_DIAGNOSIS");e.evaluate(p)
        high=e.high_influence_set(p,e.diagnose_shortage(p))
        with patch.object(e.rng,"sample",wraps=e.rng.sample) as sample:
            d=e.select_destroy_set(p)
        self.assertEqual(len(d),len(high));self.assertEqual(sample.call_args.args,(e.encoder.ms_index_order,len(high)))
        self.assertGreater(e.diagnostics["shortage_diagnosis"]["calls"],0)

    def test_wo_mgs_retains_archive_seed(self):
        e,p=make("HLNS_WO_MGS");e.evaluate(p);e.A=[p]
        with patch.object(e.rng,"random",return_value=.9),patch.object(e,"archive_guide",return_value=p) as guide, \
             patch.object(e,"makespan_actions",side_effect=AssertionError("MGS")), \
             patch.object(e,"tournament",side_effect=AssertionError("lost archive seed")):
            seed,child,source=e._generate_pm_child(True)
        self.assertIs(seed,p);self.assertEqual(source,"P_M_archive");self.assertEqual(guide.call_count,1)
        self.assertEqual(e.diagnostics["cooperation"]["pm_usable"],1)

    def test_wo_ac_disables_seed_not_archive(self):
        e,p=make("HLNS_WO_AC");e.evaluate(p);e.P_M=[p];e.A=[p]
        with patch.object(e.rng,"random",return_value=.9),patch.object(e,"archive_guide",side_effect=AssertionError("AC")), \
             patch.object(e,"_vary_makespan_specialized",return_value=p):
            _,_,source=e._generate_pm_child(True)
        self.assertEqual(source,"P_M_self");e.update_archive([p]);self.assertTrue(e.A)

    def test_wo_hlns_draws_branch_but_no_diagnosis(self):
        e,p=make("HLNS_WO_HLNS")
        expected=random.Random();expected.setstate(e.rng.getstate());expected.random()
        ref,_=make("HLNS_WO_HLNS");ref.rng=expected;child_ref=ref.basic_mutation(p)
        with patch.object(e,"diagnose_shortage",side_effect=AssertionError("diagnosis")):
            child=e.vary_shortage(p)
        self.assertEqual(e.genotype_key(child),ref.genotype_key(child_ref));self.assertEqual(e.rng.getstate(),expected.getstate())
        self.assertEqual(e.hlns_statistics["hlns_calls"],0)

    @unittest.skip("Retired internal OS-repair ablation")
    def test_MS_only_specialized_and_same_fallback(self):
        e,p=make("HLNS_WO_OS_REPAIR");e.evaluate(p)
        with patch.object(e.rng,"random",return_value=.9),patch.object(e,"joint_repair",return_value=Individual(p.os_seq[:],p.ms_map.copy(),operation_order=p.operation_order)):
            child=e.vary_shortage(p)
        self.assertEqual(child.os_seq,p.os_seq);self.assertNotEqual(child.ms_map,p.ms_map)
        self.assertEqual(e.hlns_statistics["hlns_same_genotype_fallbacks"],1)

    @unittest.skip("Retired internal MS-repair ablation")
    def test_OS_only_specialized_and_same_fallback(self):
        e,p=make("HLNS_WO_MS_REPAIR");e.evaluate(p)
        with patch.object(e.rng,"random",return_value=.9),patch.object(e,"joint_repair",return_value=Individual(p.os_seq[:],p.ms_map.copy(),operation_order=p.operation_order)):
            child=e.vary_shortage(p)
        self.assertEqual(child.ms_map,p.ms_map);self.assertNotEqual(child.os_seq,p.os_seq)

    @unittest.skip("Retired internal single-component repair ablations")
    def test_missing_H_single_component_stays_disabled(self):
        for name in ("HLNS_WO_MS_REPAIR","HLNS_WO_OS_REPAIR"):
            e,p=make(name);p.shortage=0
            with patch.object(e.rng,"random",return_value=.9):child=e.vary_shortage(p)
            if name=="HLNS_WO_MS_REPAIR":self.assertEqual(child.ms_map,p.ms_map)
            else:self.assertEqual(child.os_seq,p.os_seq)

    def test_invalid_phi_recorded_and_raised(self):
        e,p=make();e.evaluate(p);d=e.diagnose_shortage(p);d.phi[next(iter(d.phi))]+=1
        with patch.object(e,"diagnose_shortage",return_value=d):
            with self.assertRaises(HLNSDiagnosisError):e.select_destroy_set(p)
        self.assertEqual(e.hlns_statistics["hlns_diagnosis_errors"],1)

    def test_unknown_diagnosis_error_not_swallowed(self):
        e,p=make();e.evaluate(p)
        with patch.object(e,"diagnose_shortage",side_effect=KeyError("bug")):
            with self.assertRaises(KeyError):e.select_destroy_set(p)
        self.assertEqual(e.hlns_statistics["diagnosis_error_reasons"],{"KeyError":1})

    def test_zero_shortage_no_diagnosis(self):
        e,p=make();p.shortage=0
        with patch.object(e.rng,"random",return_value=.9),patch.object(e,"diagnose_shortage",side_effect=AssertionError("zero diagnosis")):
            e.vary_shortage(p)
        self.assertEqual(e.hlns_statistics["hlns_global_fallbacks"],1)

    def test_invalid_encoding_exposed_not_retried(self):
        e,p=make();high=set(p.os_seq)
        with patch("src.algorithms.wip_graph_dual_population.biased_topological_os_repair",return_value=[p.os_seq[0]]*len(p.os_seq)) as repair:
            with self.assertRaises(ValueError):e.joint_repair(p,high)
        self.assertEqual(repair.call_count,1)

    def test_unavailable_diagnosis_explicit_fallback(self):
        e,p=make();p.shortage=2
        with patch.object(e.rng,"random",return_value=.9):child=e.vary_shortage(p)
        self.assertEqual(e.hlns_statistics["fallback_reasons"],{"unavailable_diagnosis":1})
        self.assertEqual(e.hlns_statistics["hlns_global_fallbacks"],1)

    def test_empty_H_explicit_fallback(self):
        e,p=make();e.evaluate(p)
        with patch.object(e,"high_influence_set",return_value=set()):
            self.assertEqual(e.select_destroy_set(p),set())
        self.assertEqual(e.hlns_statistics["fallback_reasons"],{"empty_H":1})

    def test_nonminimal_H_is_error(self):
        e,p=make();e.evaluate(p)
        with patch.object(e,"high_influence_set",return_value=set(p.os_seq)):
            with self.assertRaises(HLNSDiagnosisError):e.select_destroy_set(p)

    def test_invalid_anchor_is_error(self):
        e,p=make();e.evaluate(p);d=e.diagnose_shortage(p);d.unanchored_intervals=1
        with patch.object(e,"diagnose_shortage",return_value=d):
            with self.assertRaises(HLNSDiagnosisError):e.select_destroy_set(p)

    @unittest.skip("Retired internal OS-repair ablation")
    def test_no_OS_repair_normal_path(self):
        e,p=make("HLNS_WO_OS_REPAIR")
        for seed in range(20):self.assertEqual(e.joint_repair(p,set(p.os_seq),random.Random(seed)).os_seq,p.os_seq)

    @unittest.skip("Retired internal MS-repair ablation")
    def test_no_MS_repair_normal_path(self):
        e,p=make("HLNS_WO_MS_REPAIR")
        for seed in range(20):self.assertEqual(e.joint_repair(p,set(p.os_seq),random.Random(seed)).ms_map,p.ms_map)

    def test_statistics_copy_is_read_only_to_engine(self):
        e,p=make();stats=e.hlns_statistics;stats["fallback_reasons"]["external"]=8
        self.assertNotIn("external",e.hlns_statistics["fallback_reasons"])
