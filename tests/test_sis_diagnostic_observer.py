"""Read-only observer contracts, actual tie winner and diagnostic summaries."""

import unittest

from experiments import run_pre_ablation as runner
from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation
from tests.test_wip_graph_dual_population import fixture


class SISObserverTests(unittest.TestCase):
    def setUp(self):
        ops, buffers, os_seq, ms = fixture()
        self.search = WIPGraphDualPopulation(ops, buffers, N=3, N_A=12, FE_max=30,
                                             p_mut=0, T_coop=1, seed=1)

    def test_actual_nearest_tie_and_detached_callback(self):
        self.search.initialize()
        seed = self.search.P_S[0]
        first, second, child = seed.copy(), seed.copy(), seed.copy()
        first.shortage = seed.shortage + 10
        second.shortage = seed.shortage + 20
        child.shortage = seed.shortage
        relink = {"high_count": 1, "initial_unit_count": 1, "K": 1,
                  "applied_unit_count": 1, "applied_os_units": 1, "applied_ms_units": 0,
                  "total_os_discrepancy_reduction": 1}
        self.search._last_ps_move_kind = "shortage_relink"
        self.search._sis_diagnostic_context = (seed, second, relink)
        observations = []

        def collect(row):
            observations.append(row.copy())
            row["S_nearest"] = -999  # Detached values cannot alter the replacement decision.

        self.search._sis_diagnostic_callback = collect
        population = [first, second]
        self.assertTrue(self.search.similarity_replace(population, child, "shortage"))
        self.assertIs(population[0], child)
        self.assertIs(population[1], second)
        self.assertEqual(observations[0]["S_nearest"], first.shortage)
        self.assertFalse(observations[0]["nearest_is_guide_if_in_PS"])
        self.assertEqual(observations[0]["class_ABCD"], "C")

    def test_hook_records_all_relinks_without_extra_fe(self):
        observations = []
        self.search.run(sis_diagnostic_callback=observations.append)
        self.assertEqual(self.search.n_evaluations, 30)
        self.assertEqual(len(observations), self.search.diagnostics["ps_moves"]["shortage_relink"]["generated"])
        self.assertTrue(observations)
        for row in observations:
            runner.validate_sis_row(row)
        self.assertIsNone(self.search._sis_diagnostic_callback)
        self.assertIsNone(self.search._sis_diagnostic_context)

    def test_observer_does_not_change_small_fixture_trajectory(self):
        ops, buffers, _, _ = fixture()
        def run(enabled):
            search = WIPGraphDualPopulation(ops, buffers, N=3, N_A=12, FE_max=30,
                                             p_mut=0, T_coop=1, seed=1)
            trace, observations = [], []
            def snapshot(a):
                trace.append((a.n_evaluations, a.rng.getstate(),
                              [[(a.genotype_key(z), z.makespan, z.shortage) for z in p]
                               for p in (a.P_M, a.P_S, a.A)]))
            search.run(diagnostic_callback=snapshot,
                       sis_diagnostic_callback=observations.append if enabled else None)
            return trace, search.diagnostics, observations
        off, on = run(False), run(True)
        self.assertEqual(off[:2], on[:2])
        self.assertTrue(on[2])

    def test_abcd_and_mismatch_denominators(self):
        base = {"delta_shortage_seed": 1, "delta_makespan_seed": 0, "K": 2,
                "applied_unit_count": 2, "H_size": 4, "U0_size": 4,
                "d_child_seed": .1, "d_child_guide": .2, "d_seed_guide": .3,
                "guide_shortage_gap": 3, "total_os_discrepancy_reduction": 2,
                "guide_distance_reduction": .1,
                "guide_source": "self", "applied_os_units": 2, "applied_ms_units": 0}
        rows = [{**base, "class_ABCD": label, "accepted": label in "AC",
                 "improves_seed_shortage": label in "AB"} for label in "ABCD"]
        summary = runner.sis_summary(rows)
        self.assertEqual(summary["acceptance_rate"], .5)
        self.assertEqual(summary["improves_seed_rate"], .5)
        self.assertEqual(summary["improved_seed_but_rejected_fraction_all"], .25)
        self.assertEqual(summary["improved_seed_but_rejected_fraction_rejected"], .5)
        self.assertEqual(summary["improved_seed_but_blocked_by_nearest"], 1)
        self.assertEqual(summary["B_acceptance_rate"], 0)

    def test_spearman_ties_and_constant(self):
        self.assertEqual(runner.sis_spearman([1, 1, 2], [2, 2, 1]), -1)
        self.assertIsNone(runner.sis_spearman([1, 1], [1, 2]))

    def test_separate_diagnostic_output_and_parameter_defaults(self):
        settings = runner.sis_settings()
        self.assertNotEqual(settings.output_root, runner.current_settings().output_root)
        self.assertEqual(settings.fe_max, 5000)
        self.assertEqual(settings.n, 100)
        self.assertEqual(settings.n_a, 200)
        self.assertEqual(len(runner.SIS_DIAGNOSTIC_INSTANCES) * len(runner.SIS_DIAGNOSTIC_SEEDS), 6)
        self.assertEqual(runner.sis_k_bucket({"applied_unit_count": 4}), "3-4")
        self.assertEqual(runner.sis_composition({"applied_os_units": 1, "applied_ms_units": 1}), "mixed")


if __name__ == "__main__":
    unittest.main()
