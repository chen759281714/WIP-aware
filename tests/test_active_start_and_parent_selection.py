"""Initial WIP reconstruction, unrestricted PS seeds and actual variant hooks."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from src.algorithms.wip_graph_dual_population_wo_mgs import WIPGraphDualPopulationWithoutMGS
from src.algorithms.wip_graph_dual_population_wo_sis import WIPGraphDualPopulationWithoutSIS
from src.algorithms.wip_graph_dual_population_wo_ac import WIPGraphDualPopulationWithoutAC
from src.solution.decoder import BufferStateEvent, DecodeProvenance, ProvenanceEvent
from tests.test_wip_graph_dual_population import fixture


VARIANTS = (WIPGraphDualPopulation, WIPGraphDualPopulationWithoutMGS,
            WIPGraphDualPopulationWithoutSIS, WIPGraphDualPopulationWithoutAC)


def run_counted_case(cls, operations, buffers, *, seed=1, fe_max=30,
                     p_mut=0, n=3, t_coop=1):
    """Spy on original hooks without replacing operators or decoding decisions."""
    search = cls(operations, buffers, N=n, N_A=12, FE_max=fe_max,
                 T_coop=t_coop, p_mut=p_mut, seed=seed)
    search.initialize()
    counts = {"basic_pm": 0, "basic_ps": 0, "zero_ps": 0}
    original_basic = search.basic_mutation
    original_ps = search.vary_shortage
    original_pm = search._generate_pm_child
    side = [None]

    def basic(parent):
        if side[0] == "M" and any(parent is z for z in search.P_M):
            counts["basic_pm"] += 1
        elif side[0] == "S" and any(parent is z for z in search.P_S):
            counts["basic_ps"] += 1
        else:
            raise AssertionError("BasicMutation parent is not specialist-seeded")
        return original_basic(parent)

    def ps(parent, pool, archive_assisted=False):
        side[0] = "S"
        assert any(parent is z for z in search.P_S)
        counts["zero_ps"] += int(parent.shortage == 0)
        return original_ps(parent, pool, archive_assisted=archive_assisted)

    def pm(cooperation):
        side[0] = "M"
        return original_pm(cooperation)

    trace = []

    def snapshot(algo):
        trace.append((algo.n_evaluations,
                      tuple(tuple((algo.genotype_key(z), z.makespan, z.shortage)
                                  for z in pop) for pop in (algo.P_M, algo.P_S, algo.A))))

    with ExitStack() as stack:
        spies = {}
        for label, method in (("mgs", "_vary_makespan_specialized"),
                              ("sis", "diagnose_shortage"),
                              ("high", "high_influence_set"),
                              ("units", "_build_relink_units"),
                              ("pm_archive", "archive_guide"),
                              ("ps_archive", "choose_archive_guide_for_shortage"),
                              ("archive_updates", "update_archive")):
            spies[label] = stack.enter_context(patch.object(search, method,
                                                            wraps=getattr(search, method)))
        stack.enter_context(patch.object(search, "basic_mutation", side_effect=basic))
        stack.enter_context(patch.object(search, "vary_shortage", side_effect=ps))
        stack.enter_context(patch.object(search, "_generate_pm_child", side_effect=pm))
        search.run(diagnostic_callback=snapshot)
        counts.update({label: spy.call_count for label, spy in spies.items()})
    assert search.n_evaluations == fe_max
    assert len(search.P_M) == len(search.P_S) == n
    assert 0 < len(search.A) <= search.N_A
    assert counts["archive_updates"] > 0
    if cls is WIPGraphDualPopulationWithoutMGS:
        assert counts["mgs"] == counts["pm_archive"] == 0
    if cls is WIPGraphDualPopulationWithoutSIS:
        assert counts["sis"] == counts["high"] == counts["units"] == counts["ps_archive"] == 0
    if cls is WIPGraphDualPopulationWithoutAC:
        assert counts["pm_archive"] == counts["ps_archive"] == 0
    if p_mut == 1:
        assert counts["mgs"] == counts["sis"] == counts["pm_archive"] == counts["ps_archive"] == 0
    return search, counts, trace


class ActiveStartTests(unittest.TestCase):
    def setUp(self):
        operations, buffers, _, _ = fixture()
        self.search = WIPGraphDualPopulation(operations, buffers, N=1, N_A=2, FE_max=2)

    def diagnose(self, events, start, end, area):
        causes = {ev.cause_event_id: ProvenanceEvent(
            ev.cause_event_id, ev.time, "release" if ev.action == "put" else "start",
            "J0", 0, "M0") for ev in events if ev.cause_event_id is not None}
        ind = Individual([], {}, shortage=area,
                         stats={"shortage": {"per_buffer_active_start": {"B": start},
                                             "per_buffer_active_end": {"B": end},
                                             "per_buffer_low_wip": {"B": 1}}},
                         provenance=DecodeProvenance(events=causes, buffer_events={"B": events}))
        diagnosis = self.search.diagnose_shortage(ind)
        self.assertEqual(diagnosis.unanchored_intervals, 0)
        self.assertAlmostEqual(sum(diagnosis.phi.values()), area, delta=1e-9)
        return diagnosis

    def test_start_01_first_put_cannot_supply_the_past(self):
        diagnosis = self.diagnose([
            BufferStateEvent(3, 5, "B", "put", "J0", 0, 1, 2)], 0, 10, 5)
        self.assertEqual(diagnosis.intervals[0][1:4], (0, 5, 5))
        self.assertEqual(diagnosis.influence_events, {2})

    def test_start_02_replay_historical_events(self):
        diagnosis = self.diagnose([
            BufferStateEvent(1, 0, "B", "init", None, 0, 0),
            BufferStateEvent(3, 1, "B", "put", "J0", 0, 1, 2),
            BufferStateEvent(5, 3, "B", "take", "J0", 1, 0, 4),
            BufferStateEvent(7, 8, "B", "put", "J0", 0, 1, 6)], 4, 10, 4)
        self.assertEqual(diagnosis.intervals[0][1:4], (4, 8, 4))
        self.assertEqual(diagnosis.influence_events, {6})

    def test_start_03_same_time_uses_event_id_not_storage_order(self):
        diagnosis = self.diagnose([
            BufferStateEvent(1, 0, "B", "init", None, 0, 0),
            BufferStateEvent(5, 2, "B", "take", "J0", 1, 0, 4),
            BufferStateEvent(3, 2, "B", "put", "J0", 0, 1, 2),
            BufferStateEvent(7, 6, "B", "put", "J0", 0, 1, 6)], 2, 10, 4)
        self.assertEqual(diagnosis.intervals[0][1:4], (2, 6, 4))
        self.assertEqual(diagnosis.influence_events, {4, 6})

    def test_start_04_initial_low_interval_relief_and_phi(self):
        diagnosis = self.diagnose([
            BufferStateEvent(1, 0, "B", "init", None, 0, 0),
            BufferStateEvent(3, 7, "B", "put", "J0", 0, 1, 2)], 2, 7, 5)
        self.assertEqual(diagnosis.intervals[0][1:4], (2, 7, 5))
        self.assertEqual(diagnosis.influence_events, {2})
        self.assertEqual(diagnosis.phi, {("J0", 0): 5})

    def test_configured_initial_content_is_not_forced_to_zero(self):
        self.search.buffers["B"]["init_content"] = ["J0", "J0"]
        diagnosis = self.diagnose([
            BufferStateEvent(3, 5, "B", "take", "J0", 1, 0, 2)], 0, 10, 5)
        self.assertEqual(diagnosis.intervals[0][1:4], (5, 10, 5))


class ShortageParentTests(unittest.TestCase):
    def setUp(self):
        operations, buffers, os_seq, ms = fixture()
        self.search = WIPGraphDualPopulation(operations, buffers, N=2, N_A=4,
                                             FE_max=8, p_mut=0, seed=7)
        self.zero = Individual(os_seq[:], ms.copy(), makespan=10, shortage=0)
        self.positive = Individual(os_seq[:], ms.copy(), makespan=9, shortage=1)

    def test_ps_seed_01_zero_wins_primary_objective(self):
        population = [self.positive, self.zero]
        with patch.object(self.search.rng, "sample", return_value=population):
            self.assertIs(self.search.tournament(population, "shortage"), self.zero)

    def test_ps_seed_02_zero_tie_uses_structural_novelty(self):
        other = self.zero.copy()
        population = [self.zero, other, self.positive]

        def distance(a, b):
            if a is self.zero and b is other:
                return 1.0
            return 0.9 if a is self.zero else 0.2

        with patch.object(self.search.rng, "sample", return_value=[other, self.zero]), \
             patch.object(self.search, "structural_distance", side_effect=distance):
            self.assertIs(self.search.tournament(population, "shortage"), self.zero)

    def test_ps_seed_03_zero_falls_back_without_sis_guide(self):
        for cooperation in (False, True):
            with self.subTest(cooperation=cooperation), \
                 patch.object(self.search, "diagnose_shortage", side_effect=AssertionError("SIS")), \
                 patch.object(self.search, "choose_guide", side_effect=AssertionError("self guide")), \
                 patch.object(self.search, "choose_archive_guide_for_shortage",
                              side_effect=AssertionError("archive guide")), \
                 patch.object(self.search, "basic_mutation", return_value=self.zero) as basic:
                self.search.vary_shortage(self.zero, [self.zero, self.positive], cooperation)
                basic.assert_called_once_with(self.zero)
                self.assertEqual(self.search._last_ps_move_kind,
                                 "fallback_basic_mutation_zero_shortage")

    def test_run_passes_entire_ps_population_in_both_round_types(self):
        for t_coop in (1, 10):
            operations, buffers, _, _ = fixture()
            search = WIPGraphDualPopulation(operations, buffers, N=2, N_A=4,
                                             FE_max=8, p_mut=0, T_coop=t_coop, seed=7)
            search.initialize()
            search.P_S[0].shortage = 0
            search.P_S[1].shortage = 1
            zero = search.P_S[0]
            with patch.object(search, "tournament", wraps=search.tournament) as tournament, \
                 patch.object(search.rng, "sample", side_effect=lambda pool, n: list(pool)[:n]), \
                 patch.object(search, "similarity_replace", return_value=False), \
                 patch.object(search, "vary_shortage", wraps=search.vary_shortage) as variation:
                search.run()
            ps_calls = [call for call in tournament.call_args_list if call.args[1] == "shortage"]
            self.assertEqual(len(ps_calls), 2)
            self.assertTrue(all(call.args[0] is search.P_S for call in ps_calls))
            self.assertTrue(all(call.args[0] is zero for call in variation.call_args_list))


class ActualVariantCallTests(unittest.TestCase):
    def test_specialized_cooperation_matrix_and_fe_structure(self):
        operations, buffers, _, _ = fixture()
        expected = ((True, True, True, True), (False, True, False, True),
                    (True, False, True, False), (True, True, False, False))
        for cls, row in zip(VARIANTS, expected):
            with self.subTest(variant=cls.algorithm_variant):
                _, counts, _ = run_counted_case(cls, operations, buffers)
                actual = tuple(counts[key] > 0 for key in ("mgs", "sis", "pm_archive", "ps_archive"))
                self.assertEqual(actual, row)

    def test_all_mutation_matrix_and_specialist_parents(self):
        operations, buffers, _, _ = fixture()
        for cls in VARIANTS:
            with self.subTest(variant=cls.algorithm_variant):
                _, counts, _ = run_counted_case(cls, operations, buffers, p_mut=1)
                self.assertEqual(counts["basic_pm"], 12)
                self.assertEqual(counts["basic_ps"], 12)


if __name__ == "__main__":
    unittest.main()
