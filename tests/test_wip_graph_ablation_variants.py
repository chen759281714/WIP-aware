"""Isolation and accounting checks for the three dual-population ablations."""

import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import run_compare_experiments as comparison_runner
from experiments.run_compare_experiments import ALGORITHMS
from experiments.validate_ablation_variants import run_smoke_case
from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation
from src.algorithms.wip_graph_dual_population_wo_ac import WIPGraphDualPopulationWithoutAC
from src.algorithms.wip_graph_dual_population_wo_mgs import WIPGraphDualPopulationWithoutMGS
from src.algorithms.wip_graph_dual_population_wo_sis import WIPGraphDualPopulationWithoutSIS
from tests.test_wip_graph_dual_population import fixture


CLASSES = (WIPGraphDualPopulation, WIPGraphDualPopulationWithoutMGS,
           WIPGraphDualPopulationWithoutSIS, WIPGraphDualPopulationWithoutAC)


class AblationVariantTests(unittest.TestCase):
    def setUp(self):
        self.operations, self.buffers, _, _ = fixture()

    def make(self, cls):
        return cls(self.operations, self.buffers, N=3, N_A=4, FE_max=60,
                   T_coop=2, seed=1)

    def test_full_read_only_metadata_does_not_change_trajectory(self):
        baseline = self.make(WIPGraphDualPopulation)
        without_metadata = self.make(WIPGraphDualPopulation)
        for key in ("algorithm_variant", "pm_specialized_kind", "ps_specialized_kind",
                    "pm_archive_guidance_enabled", "ps_archive_guidance_enabled"):
            without_metadata.diagnostics.pop(key)

        def snapshot(search):
            return (search.n_evaluations,
                    tuple(tuple((search.genotype_key(ind), ind.makespan, ind.shortage)
                                for ind in population)
                          for population in (search.P_M, search.P_S, search.A)))

        first, second = [], []
        baseline.run(diagnostic_callback=lambda search: first.append(snapshot(search)))
        without_metadata.run(diagnostic_callback=lambda search: second.append(snapshot(search)))
        self.assertEqual(first, second)
        self.assertEqual(baseline.rng.getstate(), without_metadata.rng.getstate())

    def test_without_mgs_only_replaces_pm_search(self):
        search = self.make(WIPGraphDualPopulationWithoutMGS)
        with patch.object(search, "makespan_actions", side_effect=AssertionError("MGS called")), \
             patch.object(search, "archive_guide", wraps=search.archive_guide) as archive_seed, \
             patch.object(search, "choose_archive_guide_for_shortage",
                          wraps=search.choose_archive_guide_for_shortage) as ps_guide:
            search.run()
        self.assertEqual(archive_seed.call_count, 0)
        self.assertGreater(ps_guide.call_count, 0)
        self.assertEqual(len(search.diagnostics["pm_action_sets"]), 0)
        self.assertEqual(search.diagnostics["cooperation"]["pm_usable"], 0)
        self.assertNotIn("P_M_archive", search.diagnostics["source_generated"])
        self.assertGreater(search.diagnostics["shortage_diagnosis"]["calls"], 0)
        self.assertGreater(search.diagnostics["cooperation"]["ps_probes"], 0)
        self.assertEqual(search.diagnostics["pm_moves"]["basic_mutation"]["generated"], 27)

    def test_without_sis_only_replaces_ps_search(self):
        search = self.make(WIPGraphDualPopulationWithoutSIS)
        with patch.object(search, "diagnose_shortage", side_effect=AssertionError("SIS called")), \
             patch.object(search, "_build_relink_units", side_effect=AssertionError("SIS called")), \
             patch.object(search, "choose_archive_guide_for_shortage",
                          side_effect=AssertionError("archive guide called")), \
             patch.object(search, "makespan_actions", wraps=search.makespan_actions) as mgs, \
             patch.object(search, "archive_guide", side_effect=lambda: search.A[0]) as pm_guide:
            search.run()
        self.assertGreater(mgs.call_count, 0)
        self.assertGreater(pm_guide.call_count, 0)
        self.assertGreater(search.diagnostics["cooperation"]["pm_usable"], 0)
        self.assertEqual(search.diagnostics["shortage_diagnosis"]["calls"], 0)
        self.assertEqual(search.diagnostics["cooperation"]["ps_probes"], 0)
        self.assertEqual(search.diagnostics["ps_moves"]["basic_mutation"]["generated"], 27)

    def test_without_ac_keeps_search_and_archive_but_no_guidance(self):
        search = self.make(WIPGraphDualPopulationWithoutAC)
        with patch.object(search, "choose_archive_guide_for_shortage",
                          side_effect=AssertionError("archive guide called")), \
             patch.object(search, "update_archive", wraps=search.update_archive) as update:
            search.run()
        self.assertGreater(update.call_count, 1)
        self.assertTrue(search.A)
        self.assertGreater(len(search.diagnostics["pm_action_sets"]), 0)
        self.assertGreater(search.diagnostics["shortage_diagnosis"]["calls"], 0)
        self.assertEqual(search.diagnostics["cooperation"]["pm_usable"], 0)
        self.assertEqual(search.diagnostics["cooperation"]["ps_probes"], 0)
        self.assertNotIn("P_M_archive", search.diagnostics["source_generated"])
        self.assertNotIn("P_S_archive", search.diagnostics["source_generated"])

    def test_shared_config_fe_structure_and_variant_flags(self):
        expected = {
            "full": ("MGS", "SIS", True, True),
            "wo_mgs": ("BasicMutation", "SIS", False, True),
            "wo_sis": ("MGS", "BasicMutation", True, False),
            "wo_ac": ("MGS", "SIS", False, False),
        }
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                search = self.make(cls)
                search.run()
                self.assertEqual(search.n_evaluations, search.FE_max)
                self.assertEqual((len(search.P_M), len(search.P_S)), (search.N, search.N))
                self.assertLessEqual(len(search.A), search.N_A)
                self.assertTrue(search.A)
                diag = search.diagnostics
                self.assertEqual(
                    (diag["pm_specialized_kind"], diag["ps_specialized_kind"],
                     diag["pm_archive_guidance_enabled"], diag["ps_archive_guidance_enabled"]),
                    expected[diag["algorithm_variant"]])
                self.assertEqual((search.rho, search.eta, search.p_mut, search.T_coop),
                                 (0.8, 0.3, 0.1, 2))

    def test_registry_names_share_runner_contract(self):
        names = ("WIPGraphDualPopulation", "WIPGraphDualPopulationWithoutMGS",
                 "WIPGraphDualPopulationWithoutSIS", "WIPGraphDualPopulationWithoutAC")
        instance = {"operations": self.operations, "buffers": self.buffers}
        for name, cls in zip(names, CLASSES):
            with self.subTest(name=name):
                self.assertIs(ALGORITHMS[name], cls)
                result = run_smoke_case(instance, name, seed=1, N=3, N_A=4, FE_max=13)
                self.assertEqual(result["FE"], 13)
                self.assertGreater(result["archive_size"], 0)

    def test_comparison_runner_accepts_all_variant_identifiers(self):
        path = Path(__file__).resolve().parents[1] / (
            "data/final_benchmark/instances/WIPHFSP_S1_P1_B1.json")
        with patch.object(comparison_runner, "POP_SIZE", 6), \
             patch.object(comparison_runner, "MAX_EVALUATIONS", 13):
            for name in ALGORITHMS:
                with self.subTest(name=name):
                    result = comparison_runner.run_once(str(path), 1, name)
                    self.assertEqual(result["algorithm"], name)
                    self.assertEqual(result["representative_result"]["n_evaluations"], 13)
                    self.assertTrue(result["pareto_front"])


if __name__ == "__main__":
    unittest.main()
