"""Frozen branch-source cooperation and factual shortage anchor checks."""

import inspect
import unittest
import warnings
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from src.algorithms.wip_graph_dual_population import (
    Individual, ShortageDiagnosis, WIPGraphDualPopulation,
)
from src.algorithms.wip_graph_dual_population_wo_mgs import WIPGraphDualPopulationWithoutMGS
from src.algorithms.wip_graph_dual_population_wo_sis import WIPGraphDualPopulationWithoutSIS
from src.algorithms.wip_graph_dual_population_wo_ac import WIPGraphDualPopulationWithoutAC
from src.solution.decoder import BufferStateEvent, DecodeProvenance, ProvenanceEvent
from tests.test_wip_graph_dual_population import fixture


class CooperationTests(unittest.TestCase):
    def setUp(self):
        operations, buffers, os_seq, ms = fixture()
        self.search = WIPGraphDualPopulation(operations, buffers, N=2, N_A=4,
                                             FE_max=8, T_coop=1, seed=7)
        self.pm = Individual(os_seq[:], ms.copy(), makespan=10, shortage=2)
        self.ps = Individual(os_seq[:], ms.copy(), makespan=11, shortage=1)
        self.archive = Individual(os_seq[:], ms.copy(), makespan=9, shortage=3)
        self.search.P_M, self.search.P_S = [self.pm], [self.ps]

    def check_pm(self, cooperation, mutation, archive_available=True):
        self.search.p_mut = float(mutation)
        marker = self.search.random_individual()
        with patch.object(self.search, "tournament", return_value=self.pm) as tournament, \
             patch.object(self.search, "archive_guide",
                          return_value=self.archive if archive_available else None) as archive, \
             patch.object(self.search, "basic_mutation", return_value=marker) as basic, \
             patch.object(self.search, "_vary_makespan_specialized", return_value=marker) as mgs:
            seed, child, source = self.search._generate_pm_child(cooperation)
        attempted = cooperation and not mutation
        expected = self.archive if attempted and archive_available else self.pm
        self.assertIs(seed, expected)
        self.assertIs(child, marker)
        self.assertEqual(source, "P_M_archive" if expected is self.archive else "P_M_self")
        self.assertEqual(archive.call_count, int(attempted))
        self.assertEqual(tournament.call_count, int(expected is self.pm))
        if mutation:
            basic.assert_called_once_with(self.pm)
            mgs.assert_not_called()
        else:
            mgs.assert_called_once_with(expected)
            basic.assert_not_called()
        counts = self.search.diagnostics["cooperation"]
        self.assertEqual(counts["pm_attempts"], int(attempted))
        self.assertEqual(counts["pm_fallback_self"], int(attempted and not archive_available))

    def check_ps(self, cooperation, mutation, archive_available=True):
        self.search.p_mut = float(mutation)
        diagnosis = ShortageDiagnosis([], {("J0", 0): 1.0}, set(), [], 0)
        marker = self.search.random_individual()
        with patch.object(self.search, "diagnose_shortage", return_value=diagnosis) as diagnose, \
             patch.object(self.search, "choose_archive_guide_for_shortage",
                          return_value=self.archive if archive_available else None) as archive, \
             patch.object(self.search, "choose_guide", return_value=self.pm) as self_guide, \
             patch.object(self.search, "_build_relink_units", return_value=[]), \
             patch.object(self.search, "basic_mutation", return_value=marker) as basic:
            self.search.vary_shortage(self.ps, self.search.P_S, archive_assisted=cooperation)
        attempted = cooperation and not mutation
        self.assertEqual(archive.call_count, int(attempted))
        self.assertEqual(diagnose.call_count, int(not mutation))
        self.assertEqual(self_guide.call_count,
                         int(not mutation and not (cooperation and archive_available)))
        basic.assert_called_once_with(self.ps)
        counts = self.search.diagnostics["cooperation"]
        self.assertEqual(counts["ps_attempts"], int(attempted))
        self.assertEqual(counts["ps_probes"], int(attempted))
        self.assertEqual(counts["ps_fallback_self"], int(attempted and not archive_available))

    def test_coop_01_normal_pm_mutation(self):
        self.check_pm(False, True)

    def test_coop_02_normal_pm_specialized(self):
        self.check_pm(False, False)

    def test_coop_03_cooperation_pm_mutation(self):
        self.check_pm(True, True)

    def test_coop_04_cooperation_pm_specialized(self):
        self.check_pm(True, False)

    def test_coop_05_unavailable_pm_archive(self):
        self.check_pm(True, False, False)

    def test_coop_06_normal_ps_mutation(self):
        self.check_ps(False, True)

    def test_coop_07_normal_ps_specialized(self):
        self.check_ps(False, False)

    def test_coop_08_cooperation_ps_mutation(self):
        self.check_ps(True, True)

    def test_coop_09_cooperation_ps_specialized(self):
        self.check_ps(True, False)

    def test_coop_10_invalid_ps_archive(self):
        self.check_ps(True, False, False)

    def test_coop_11_retired_parameter_absent(self):
        retired = "gamma" + "_A"
        self.assertNotIn(retired, inspect.signature(WIPGraphDualPopulation).parameters)
        self.assertFalse(hasattr(self.search, retired))
        root = Path(__file__).resolve().parents[1]
        for directory in ("src", "tests", "experiments"):
            for path in (root / directory).rglob("*.py"):
                if "results" not in path.parts:
                    self.assertNotIn(retired.lower(), path.read_text().lower(), str(path))

    def run_controlled_round(self, cls, p_mut):
        operations, buffers, _, _ = fixture()
        search = cls(operations, buffers, N=2, N_A=4, FE_max=8,
                     T_coop=1, p_mut=p_mut, seed=4)
        search.initialize()
        for seed in search.P_S:
            seed.shortage = 1.0
        diagnosis = ShortageDiagnosis([], {("J0", 0): 1.0}, set(), [], 0)

        def specialized(seed):
            search._last_pm_move_kind = "machine_reassign"
            return search.random_individual()

        original_basic = search.basic_mutation

        def basic(seed):
            self.assertTrue(any(seed is z for z in search.P_M + search.P_S))
            return original_basic(seed)

        with ExitStack() as stack:
            mgs = stack.enter_context(patch.object(search, "_vary_makespan_specialized",
                                                   side_effect=specialized))
            archive_pm = stack.enter_context(patch.object(search, "archive_guide",
                                                          return_value=search.A[0]))
            diagnose = stack.enter_context(patch.object(search, "diagnose_shortage",
                                                        return_value=diagnosis))
            archive_ps = stack.enter_context(patch.object(search, "choose_archive_guide_for_shortage",
                                                          return_value=None))
            stack.enter_context(patch.object(search, "choose_guide", return_value=None))
            stack.enter_context(patch.object(search, "similarity_replace", return_value=False))
            basic_mock = stack.enter_context(patch.object(search, "basic_mutation", side_effect=basic))
            search.run()
        self.assertEqual(search.n_evaluations, 8)
        self.assertTrue(search.A)
        return search, mgs.call_count, diagnose.call_count, archive_pm.call_count, archive_ps.call_count, basic_mock.call_count

    def test_full_cooperation_all_specialized(self):
        search, mgs, sis, pm, ps, _ = self.run_controlled_round(WIPGraphDualPopulation, 0)
        self.assertEqual((mgs, sis, pm, ps), (2, 2, 2, 2))
        self.assertEqual(search.diagnostics["cooperation"]["pm_attempts"], 2)
        self.assertEqual(search.diagnostics["cooperation"]["ps_attempts"], 2)

    def test_full_cooperation_all_mutation(self):
        search, mgs, sis, pm, ps, basic = self.run_controlled_round(WIPGraphDualPopulation, 1)
        self.assertEqual((mgs, sis, pm, ps, basic), (0, 0, 0, 0, 4))
        self.assertEqual(search.diagnostics["cooperation"]["pm_attempts"], 0)
        self.assertEqual(search.diagnostics["cooperation"]["ps_attempts"], 0)

    def test_variant_cooperation_matrix(self):
        matrix = ((WIPGraphDualPopulation, (2, 2, 2, 2)),
                  (WIPGraphDualPopulationWithoutMGS, (0, 2, 0, 2)),
                  (WIPGraphDualPopulationWithoutSIS, (2, 0, 2, 0)),
                  (WIPGraphDualPopulationWithoutAC, (2, 2, 0, 0)))
        for cls, expected in matrix:
            with self.subTest(variant=cls.algorithm_variant):
                self.assertEqual(self.run_controlled_round(cls, 0)[1:5], expected)

    def test_unsuccessful_archive_mgs_basic_fallback_uses_pm(self):
        self.search.p_mut = 0
        with patch.object(self.search, "archive_guide", return_value=self.archive), \
             patch.object(self.search, "_vary_makespan_specialized", return_value=None), \
             patch.object(self.search, "tournament", return_value=self.pm), \
             patch.object(self.search, "basic_mutation", return_value=self.pm) as basic:
            seed, _, source = self.search._generate_pm_child(True)
        basic.assert_called_once_with(self.pm)
        self.assertIs(seed, self.pm)
        self.assertEqual(source, "P_M_self")


class FactualAnchorTests(unittest.TestCase):
    def setUp(self):
        operations, buffers, self.os_seq, self.ms = fixture()
        self.search = WIPGraphDualPopulation(operations, buffers, N=1, N_A=2,
                                             FE_max=2, p_mut=0, seed=0)

    def individual(self, events, start=0, end=10, shortage=10.0):
        causes = {ev.cause_event_id: ProvenanceEvent(ev.cause_event_id, ev.time,
                  "release" if ev.action == "put" else "start", "J0", 0, "M0")
                  for ev in events if ev.cause_event_id is not None}
        return Individual(self.os_seq[:], self.ms.copy(), shortage=shortage,
                          stats={"shortage": {"per_buffer_active_start": {"B": start},
                                              "per_buffer_active_end": {"B": end},
                                              "per_buffer_low_wip": {"B": 1}}},
                          provenance=DecodeProvenance(events=causes, buffer_events={"B": events}))

    def test_anchor_01_deficit_opening_take(self):
        ind = self.individual([BufferStateEvent(1, 0, "B", "init", None, 1, 1),
                               BufferStateEvent(3, 2, "B", "take", "J0", 1, 0, 2)], shortage=8)
        diagnosis = self.search.diagnose_shortage(ind)
        self.assertEqual(diagnosis.influence_events, {2})
        self.assertEqual(sum(diagnosis.phi.values()), 8)
        self.assertEqual(diagnosis.unanchored_intervals, 0)

    def test_anchor_02_deficit_relief_put(self):
        ind = self.individual([BufferStateEvent(1, 0, "B", "init", None, 0, 0),
                               BufferStateEvent(3, 5, "B", "put", "J0", 0, 1, 2)], shortage=5)
        diagnosis = self.search.diagnose_shortage(ind)
        self.assertEqual(diagnosis.influence_events, {2})
        self.assertEqual(sum(diagnosis.phi.values()), 5)

    def test_anchor_03_initial_low_right_endpoint_relief(self):
        ind = self.individual([BufferStateEvent(1, 0, "B", "init", None, 0, 0),
                               BufferStateEvent(3, 10, "B", "put", "J0", 0, 1, 2)])
        diagnosis = self.search.diagnose_shortage(ind)
        self.assertEqual(diagnosis.influence_events, {2})
        self.assertEqual(sum(diagnosis.phi.values()), 10)
        self.assertEqual(diagnosis.unanchored_intervals, 0)

    def assert_unanchored(self, ind):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            diagnosis = self.search.diagnose_shortage(ind)
        self.assertTrue(captured)
        self.assertEqual(diagnosis.unanchored_intervals, 1)
        self.assertEqual(diagnosis.phi, {})
        self.assertEqual(diagnosis.influence_events, set())
        self.assertEqual(diagnosis.influence_dependencies, [])
        self.assertEqual(self.search.diagnostics["shortage_diagnosis"]["incomplete_phi_calls"], 1)
        with patch.object(self.search, "diagnose_shortage", return_value=diagnosis), \
             patch.object(self.search, "high_influence_set", side_effect=AssertionError("invalid H")), \
             patch.object(self.search, "_build_relink_units", side_effect=AssertionError("invalid SIS")), \
             patch.object(self.search, "basic_mutation", return_value=ind) as basic:
            self.search.vary_shortage(ind, [ind], archive_assisted=True)
        basic.assert_called_once_with(ind)
        self.assertEqual(self.search._last_ps_move_kind, "fallback_basic_mutation_phi_invalid")

    def test_anchor_04_no_factual_anchor_invalid(self):
        self.assert_unanchored(self.individual([
            BufferStateEvent(1, 0, "B", "init", None, 0, 0)]))

    def test_anchor_05_history_cannot_fabricate_anchor(self):
        events = [BufferStateEvent(1, 0, "B", "init", None, 0, 0)]
        events.extend(BufferStateEvent(2 * i + 1, i, "B", "take", "J0", 0, 0, 2 * i)
                      for i in range(1, 6))
        self.assert_unanchored(self.individual(events, start=6, end=10, shortage=4))


if __name__ == "__main__":
    unittest.main()
