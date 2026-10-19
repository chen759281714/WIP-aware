"""Offline path fidelity, deterministic stratification, and overrun definitions."""

import json
import unittest
from unittest.mock import patch

from experiments import run_pre_ablation as runner
from tests.test_ps_relinking import diagnosis, individual, make_search
from tests.test_wip_graph_dual_population import fixture
from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation
from src.solution.decoder import StageBufferWIPScheduler


class SISPrefixReplayTests(unittest.TestCase):
    def path_fixture(self):
        search = make_search("ABCDE", stages=2, eta=1, rho=1, p_mut=0)
        x = individual([(j, k) for j in "ABCDE" for k in range(2)])
        y = individual([(j, k) for k in range(2) for j in "EDCBA"], shortage=9, machine="M1")
        phi = {key: 1 for key in x.os_seq}
        with patch.object(search, "diagnose_shortage", return_value=diagnosis(phi)), \
             patch.object(search, "_machine_shortage_evidence", return_value=set(phi)):
            child = search.vary_shortage(x, [y])
        return search, x, y, child, json.loads(json.dumps(search.diagnostics["relinking"][-1]))

    def test_exact_dynamic_unit_path_and_full_genotype(self):
        search, x, y, child, relink = self.path_fixture()
        state, fe = search.rng.getstate(), search.n_evaluations
        path = runner.rebuild_prefix_path(search, x, y, relink, runner.SIS_PREFIX_STEPS)
        self.assertEqual(path[-1]["genome"], runner.prefix_genome(child))
        self.assertEqual(path[0]["genome"], runner.prefix_genome(x))
        self.assertEqual(state, search.rng.getstate())
        self.assertEqual(fe, search.n_evaluations)
        for point in path:
            self.assertEqual(point["applied_os_units_so_far"] + point["applied_ms_units_so_far"], point["step"])
            search.encoder.validate_os(runner.prefix_individual(point["genome"]).os_seq)

    def test_full_smaller_than_checkpoint_no_duplicate_or_excess(self):
        search, x, y, child, relink = self.path_fixture()
        path = runner.rebuild_prefix_path(search, x, y, relink, [0, 1, 2, 4, 8, 16, 32, 64, "full"])
        steps = [r["step"] for r in path]
        self.assertEqual(len(steps), len(set(steps)))
        self.assertLessEqual(max(steps), relink["applied_unit_count"])
        self.assertNotIn(64, steps)

    def test_action_mismatch_fails_not_approximate_replay(self):
        search, x, y, child, relink = self.path_fixture()
        next(item for item in relink["applied_units"] if item["kind"] == "OS")["target_index"] += 1
        with self.assertRaisesRegex(AssertionError, "concrete action"):
            runner.rebuild_prefix_path(search, x, y, relink, [0, 1, "full"])

    def test_initial_unit_order_mismatch_fails(self):
        search, x, y, child, relink = self.path_fixture()
        relink["initial_units"].reverse()
        with self.assertRaisesRegex(AssertionError, "U0/order"):
            runner.rebuild_prefix_path(search, x, y, relink, [0, "full"])

    def test_prefix_is_actual_formal_first_action(self):
        search, x, y, child, relink = self.path_fixture()
        path = runner.rebuild_prefix_path(search, x, y, relink, [0, 1, "full"])
        action = relink["applied_units"][0]
        expected = x.copy()
        if action["kind"] == "OS":
            expected.os_seq = search._insert(expected.os_seq, tuple(action["key"]), action["target_index"])
        else:
            expected.ms_map[tuple(action["key"])] = action["machine"]
        self.assertEqual(path[1]["genome"], runner.prefix_genome(expected))

    def test_offline_decode_matches_evaluate_without_fe_or_rng(self):
        ops, buffers, _, _ = fixture()
        engine = WIPGraphDualPopulation(ops, buffers, N=3, FE_max=30, seed=1)
        ind = engine.random_individual()
        reference = engine.evaluate(ind.copy())
        state, fe = engine.rng.getstate(), engine.n_evaluations
        decoded = runner.prefix_offline_decode(StageBufferWIPScheduler(ops, buffers), ind)
        self.assertEqual((decoded["shortage"], decoded["makespan"]), (reference.shortage, reference.makespan))
        self.assertEqual(engine.rng.getstate(), state)
        self.assertEqual(engine.n_evaluations, fe)
        self.assertFalse(engine.P_M or engine.P_S or engine.A)

    def test_capture_subclass_preserves_formal_trajectory_and_genotype(self):
        ops, buffers, _, _ = fixture()
        arguments = dict(N=3, N_A=12, FE_max=30, p_mut=0, T_coop=1, seed=1)
        baseline = WIPGraphDualPopulation(ops, buffers, **arguments)
        expected = []
        baseline.run(sis_diagnostic_callback=expected.append)
        capture = runner.SISPrefixCaptureAlgorithm(ops, buffers, **arguments)
        capture.prefix_expected = {r["fe_index"]: r for r in expected}
        capture.prefix_selected = capture.prefix_expected.copy()
        capture.prefix_seen, capture.prefix_captures = set(), []
        capture.run(sis_diagnostic_callback=lambda row: None)
        self.assertTrue(capture.prefix_captures)
        self.assertEqual(baseline.rng.getstate(), capture.rng.getstate())
        self.assertEqual(baseline.diagnostics, capture.diagnostics)
        for left, right in zip((baseline.P_M, baseline.P_S, baseline.A), (capture.P_M, capture.P_S, capture.A)):
            self.assertEqual([(baseline.genotype_key(z), z.shortage, z.makespan) for z in left],
                             [(capture.genotype_key(z), z.shortage, z.makespan) for z in right])
        for event in capture.prefix_captures:
            path = runner.rebuild_prefix_path(capture, runner.prefix_individual(event["seed"]),
                                             runner.prefix_individual(event["guide"]), event["relink"], runner.SIS_PREFIX_STEPS)
            self.assertEqual(path[-1]["genome"], event["formal_full"])

    def test_balanced_deterministic_sampling_and_quartile_severity(self):
        rows = []
        for name in ["P2", "P3"]:
            for seed in range(3):
                for source in ["self", "archive"]:
                    for i in range(100):
                        rows.append(dict(instance=name, seed=seed, guide_source=source,
                                         fe_index=i + (100 if source == "self" else 0),
                                         delta_shortage_seed=1 if i < 20 else -i))
        records = [{"children": rows}]
        a, strata = runner.sample_prefix_events(records, 150, 4)
        self.assertEqual((a, strata), runner.sample_prefix_events(records, 150, 4))
        self.assertEqual(len(a), 150)
        self.assertEqual(len({(r["instance"], r["seed"], r["fe_index"]) for r in a}), 150)
        self.assertEqual(len(strata), 36)
        self.assertTrue(all(r["sample_count"] >= 4 for r in strata))
        self.assertTrue(all(r["delta_shortage_seed"] > 0 for r in a if r["final_group"] == "improved"))
        self.assertTrue(all(r["delta_shortage_seed"] <= -80 for r in a if r["final_group"] == "severe_worse"))

    def test_impossible_sampling_target_fails(self):
        with self.assertRaises(ValueError):
            runner.sample_prefix_events([], 150, 4)

    def event(self, shortages):
        rows = []
        for step, shortage in enumerate(shortages):
            rows.append(dict(instance="P2", seed=0, original_fe_index=20,
                             guide_source="self", final_group="mild_worse",
                             final_accepted=shortages[-1] < shortages[0], prefix_step=step,
                             full_K=len(shortages) - 1, is_full=step == len(shortages) - 1,
                             shortage=shortage, delta_shortage_from_seed=shortages[0] - shortage))
        return runner.prefix_event_summary(rows)

    def test_or1_is_subset_of_or2(self):
        e = self.event([100, 90, 120])
        self.assertTrue(e["OR1"] and e["OR2"])
        self.assertFalse(e["OR3"])
        self.assertEqual((e["best_prefix_step"], e["overrun_loss"]), (1, 30))

    def test_prefix_zero_or2_does_not_imply_effective_early_unit(self):
        e = self.event([100, 110, 120])
        self.assertTrue(e["OR2"])
        self.assertFalse(e["OR1"] or e["early_prefix_improved"])
        self.assertEqual(e["best_prefix_step"], 0)

    def test_best_full_and_tie_rule(self):
        self.assertTrue(self.event([100, 90, 80])["OR3"])
        tied = self.event([100, 80, 80])
        self.assertTrue(tied["OR3"])
        self.assertFalse(tied["OR2"])
        self.assertEqual(tied["best_prefix_step"], 2)

    def test_accepted_final_can_still_have_overrun_without_or1(self):
        e = self.event([100, 80, 90])
        self.assertFalse(e["OR1"])
        self.assertTrue(e["OR2"] and e["final_accepted"])


if __name__ == "__main__":
    unittest.main()
