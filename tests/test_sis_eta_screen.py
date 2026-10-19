"""Controlled eta tasks, independent effectiveness, pooled metrics, and resume guards."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import run_pre_ablation as r
from tests.test_wip_graph_dual_population import fixture


class SISEtaScreenTests(unittest.TestCase):
    def settings(self, root):
        return r.Settings(root / "instances", root / "results", 30, 3, 12, .1, .8, .3, 1)

    def row(self, *, child=9, nearest=8, accepted=False, label="B"):
        delta = 10 - child
        return {"S_seed": 10, "S_child": child, "S_nearest": nearest,
                "accepted": accepted, "class_ABCD": label, "H_size": 10, "U0_size": 20,
                "K": 1, "applied_unit_count": 1, "applied_os_units": 1, "applied_ms_units": 0,
                "delta_shortage_seed": delta, "buffer_delta": {"b": delta}}

    def test_seed_improvement_is_not_acceptance(self):
        c = r.EtaOperatorCollector(.05)
        c(self.row())
        c(self.row())
        summary = c.summary()
        self.assertEqual(summary["sis_improves_seed_count"], 2)
        self.assertEqual(summary["sis_accepted"], 0)
        self.assertEqual(summary["sis_improves_seed_rate"], 1)
        self.assertEqual(summary["sis_acceptance_rate"], 0)

    def test_acceptance_does_not_imply_seed_improvement(self):
        c = r.EtaOperatorCollector(.05)
        c(self.row(child=11, nearest=12, accepted=True, label="C"))
        self.assertEqual(c.summary()["sis_accepted"], 1)
        self.assertEqual(c.summary()["sis_improves_seed_count"], 0)

    def test_wrong_eta_formula_fails(self):
        with self.assertRaisesRegex(AssertionError, "Eta/K"):
            r.EtaOperatorCollector(.3)(self.row())

    def test_empty_collector_and_histogram(self):
        self.assertIsNone(r.EtaOperatorCollector(.05).summary()["sis_improves_seed_rate"])
        c = r.EtaOperatorCollector(.05)
        c(self.row())
        self.assertEqual(c.summary()["K_histogram"], {"1": 1})

    def test_default_configuration_30_tasks_no_default_eta_edit(self):
        self.assertEqual(len(r.SIS_ETA_SCREEN_INSTANCES) * len(r.SIS_ETA_SCREEN_VALUES) * len(r.SIS_ETA_SCREEN_SEEDS), 30)
        self.assertEqual(r.SIS_ETA_SCREEN_VALUES, [.02, .05, .10, .20, .30])
        self.assertEqual(r.ETA, .30)
        s = r.eta_screen_settings()
        self.assertEqual((s.n, s.n_a, s.rho, s.p_mu, s.t_coop, s.fe_max), (100, 200, .8, .1, 10, 5000))
        changed = r.eta_task_settings(s, .05)
        self.assertEqual(changed.eta, .05)
        self.assertEqual({k: v for k, v in s.algorithm_parameters(0).items() if k != "eta"},
                         {k: v for k, v in changed.algorithm_parameters(0).items() if k != "eta"})

    def test_paths_are_separate_by_eta(self):
        s = r.eta_screen_settings()
        self.assertNotEqual(r.eta_raw_path(r.EtaTask("P2", .05, 0), s), r.eta_raw_path(r.EtaTask("P2", .30, 0), s))
        self.assertIn("eta_0.10", str(r.eta_raw_path(r.EtaTask("P2", .10, 0), s)))

    def test_real_run_raw_parameters_fe_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            s = self.settings(Path(directory))
            ops, buffers, _, _ = fixture()
            r.write_json_atomic(s.instance_dir / "Tiny.json", {"operations": ops, "buffers": buffers})
            task = r.EtaTask("Tiny", .05, 0)
            result = r.run_eta_task(task, s, "test")
            raw = r.load_eta_raw(task, s, "test")
            self.assertEqual(raw["actual_FE"], 30)
            self.assertEqual(raw["algorithm_parameters"]["eta"], .05)
            self.assertTrue(raw["final_archive"])
            self.assertEqual(result["sis_generated"], raw["diagnostics"]["ps_moves"]["shortage_relink"]["generated"])
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                r.load_eta_raw(task, s, "changed")

    def synthetic(self, task, s, points):
        c = r.EtaOperatorCollector(.05)
        c(self.row())
        return {"instance": task.instance, "variant": "full", "eta": task.eta, "seed": task.seed,
                "FE_max": 30, "actual_FE": 30, "runtime_seconds": 1, "fingerprint": "test",
                "algorithm_parameters": r.eta_task_settings(s, task.eta).algorithm_parameters(task.seed),
                "final_archive_size": len(points), "final_archive": [{"makespan": x, "shortage": y} for x, y in points],
                "best_makespan": min(x for x, y in points), "best_shortage": min(y for x, y in points),
                "sis_operator": c.summary(), "diagnostics": {"ps_moves": {"shortage_relink": {"generated": 1, "accepted": 0}}}}

    def test_pooled_bounds_reference_and_paired_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            s = self.settings(Path(directory))
            for eta, points in ((.05, [(0, 4), (2, 2), (4, 0)]), (.30, [(1, 3), (3, 1)])):
                task = r.EtaTask("Tiny", eta, 0)
                r.write_json_atomic(r.eta_raw_path(task, s), self.synthetic(task, s, points))
            with patch.object(r, "plot_eta_figures"):
                summary = r.analyze_eta_screen(s, ["Tiny"], [.05, .30], [0], "test")
            self.assertEqual(summary["normalization"]["Tiny"]["bounds"], (0, 4, 0, 4))
            self.assertEqual(summary["normalization"]["Tiny"]["reference_size"], 5)
            small, base = summary["run_metrics"]
            self.assertAlmostEqual(small["nhv"], .46 / 1.21)
            self.assertAlmostEqual(base["nhv"], .4725 / 1.21)
            paired = summary["paired_comparisons"][0]
            self.assertLess(paired["delta_nhv"], 0)
            self.assertGreater(paired["delta_igd"], 0)
            self.assertEqual(paired["delta_best_makespan"], 1)
            self.assertEqual(paired["delta_best_shortage"], 1)
            wtl = {row["metric"]: row for row in summary["wtl"] if row["scope"] == "overall"}
            self.assertEqual(wtl["nhv"]["losses"], 1)
            self.assertEqual(wtl["igd"]["wins"], 1)

    def test_incomplete_eta_pool_never_analyzed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                r.analyze_eta_screen(self.settings(Path(directory)), ["Tiny"], [.05, .30], [0], "test")


if __name__ == "__main__":
    unittest.main()
