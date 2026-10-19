"""Matched atomic mechanics, natural-state sampling, isolation and pairing."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import run_pre_ablation as r
from experiments import sis_local_viability as v
from src.algorithms.wip_graph_dual_population import Individual, ShortageDiagnosis, WIPGraphDualPopulation
from tests.test_wip_graph_dual_population import fixture


class LocalViabilityTests(unittest.TestCase):
    def setUp(self):
        operations, buffers, os_seq, ms = fixture()
        self.engine = WIPGraphDualPopulation(operations, buffers, N=3, N_A=12, FE_max=30,
                                             seed=1, p_mut=0, T_coop=1)
        self.ind = Individual(os_seq, ms, operation_order=tuple(self.engine.encoder.ms_index_order))
        self.state = {"instance": "Tiny", "seed": 1, "fe_index": 9, "eligible_index": 0,
                      "genome": r.prefix_genome(self.ind), "H": [list(op) for op in os_seq],
                      "phi": [[list(op), 1] for op in os_seq], "S_seed": 10, "Cmax_seed": 10,
                      "buffer_shortage": {"B": 10}}

    def test_diagnostic_plan_repeatable_identity_sensitive_no_formal_rng(self):
        before = self.engine.rng.getstate()
        plan = v.make_trial_plan("Tiny", 0, 9, 0)
        self.assertEqual(plan, v.make_trial_plan("Tiny", 0, 9, 0))
        self.assertNotEqual(plan, v.make_trial_plan("Tiny", 0, 9, 1))
        self.assertNotEqual(plan, v.make_trial_plan("Tiny", 1, 9, 0))
        self.assertTrue(all(0 <= t < 1 for t in vars(plan).values()))
        self.assertEqual(before, self.engine.rng.getstate())

    def test_even_sampling_covers_temporal_range_no_rng(self):
        picks = v.even_indices(2400, 40)
        self.assertEqual(picks, list(range(30, 2400, 60)))
        self.assertEqual(len(set(picks)), 40)
        with self.assertRaises(ValueError):
            v.even_indices(39, 40)

    def test_all_os_candidates_legal_global_nonnoop(self):
        candidates = v.atomic_candidates(self.engine, self.ind)["OS"]
        cross_stage = False
        for key, gaps in candidates.items():
            for gap in gaps:
                child = self.engine._insert(self.ind.os_seq, key, gap)
                self.engine.encoder.validate_os(child)
                self.assertNotEqual(child, self.ind.os_seq)
                self.assertEqual(child.index(key), gap)
                old = self.ind.os_seq.index(key)
                crossed = self.ind.os_seq[min(old, gap):max(old, gap) + 1]
                cross_stage |= any(op[0] != key[0] and op[1] != key[1] for op in crossed)
        self.assertTrue(cross_stage)

    def test_os_candidates_exact_basic_mutation_mechanics(self):
        moves = v.atomic_candidates(self.engine, self.ind)["OS"]
        for target_token in (0, .3, .7, .999):
            for action_token in (0, .4, .999):
                tokens = iter((0, target_token, action_token))
                with patch.object(self.engine.rng, "choice", side_effect=lambda options: v.quantile_choice(options, next(tokens))):
                    formal = self.engine.basic_mutation(self.ind)
                key = v.quantile_choice(list(moves), target_token)
                gap = v.quantile_choice(moves[key], action_token)
                self.assertEqual(formal.os_seq, self.engine._insert(self.ind.os_seq, key, gap))
                self.assertEqual(formal.ms_map, self.ind.ms_map)

    def test_ms_candidates_exact_basic_mutation_no_speed_bias(self):
        moves = v.atomic_candidates(self.engine, self.ind)["MS"]
        for target_token in (0, .3, .7, .999):
            tokens = iter((.9, target_token, .7))
            with patch.object(self.engine.rng, "choice", side_effect=lambda options: v.quantile_choice(options, next(tokens))):
                formal = self.engine.basic_mutation(self.ind)
            key = v.quantile_choice(list(moves), target_token)
            machine = v.quantile_choice(moves[key], .7)
            self.assertEqual(formal.ms_map[key], machine)
            self.assertEqual(formal.os_seq, self.ind.os_seq)
            self.assertEqual(sum(formal.ms_map[op] != self.ind.ms_map[op] for op in self.ind.ms_map), 1)

    def test_available_type_distribution_not_always_50_50(self):
        for job in self.engine.operations:
            for op in self.engine.operations[job]:
                op["machines"] = {next(iter(op["machines"])): 1}
        for token in (0, .999):
            common, children = v.matched_atomic_actions(self.engine, self.ind, self.state, v.TrialPlan(token, .1, .2, .3))
            self.assertEqual(common["move_type"], "OS")
            self.assertEqual(len(children), 2)

    def test_weighted_sampling_zero_phi_excluded(self):
        options = [("A", 0), ("B", 0), ("C", 0)]
        phi = {options[0]: 0, options[1]: 1, options[2]: 3}
        self.assertEqual(v.weighted_target(options, phi, 0), options[1])
        self.assertEqual(v.weighted_target(options, phi, .2499), options[1])
        self.assertEqual(v.weighted_target(options, phi, .25), options[2])
        with self.assertRaises(ValueError):
            v.weighted_target(options, {}, .5)

    def test_shared_quantile_under_different_candidate_counts(self):
        self.assertEqual(v.quantile_choice(list(range(3)), .6), 1)
        self.assertEqual(v.quantile_choice(list(range(7)), .6), 4)

    def test_type_fallback_jointly_switches_policies(self):
        candidates = {"OS": {("J0", 0): [1]}, "MS": {("J1", 0): ["M4"]}}
        self.state["H"] = [["J1", 0]]
        with patch.object(v, "atomic_candidates", return_value=candidates):
            common, children = v.matched_atomic_actions(self.engine, self.ind, self.state, v.TrialPlan(0, .2, .3, .4))
        self.assertEqual(common["requested_move_type"], "OS")
        self.assertTrue(common["type_switched"])
        self.assertEqual([metadata["move_type"] for child, metadata in children], ["MS", "MS"])

    def test_no_common_action_never_one_policy_random_fallback(self):
        self.state["phi"] = [[op, 0] for op in self.state["H"]]
        common, children = v.matched_atomic_actions(self.engine, self.ind, self.state, v.TrialPlan(0, .2, .3, .4))
        self.assertEqual(common["status"], "no-common-action")
        self.assertEqual(children, [])

    def test_action_copies_genes_and_uses_no_formal_rng(self):
        before = copy.deepcopy(self.ind)
        rng = self.engine.rng.getstate()
        for type_token in (0, .99):
            common, children = v.matched_atomic_actions(self.engine, self.ind, self.state, v.TrialPlan(type_token, .1, .5, .7))
            for child, metadata in children:
                self.assertIsNone(child.stats)
                self.assertIsNot(child.os_seq, self.ind.os_seq)
                self.assertIsNot(child.ms_map, self.ind.ms_map)
        self.assertEqual(before, self.ind)
        self.assertEqual(rng, self.engine.rng.getstate())

    def test_capture_includes_no_guide_states_without_outcome_condition(self):
        operations, buffers, _, _ = fixture()
        engine = v.LocalStateCaptureAlgorithm(operations, buffers, N=3, N_A=12, FE_max=30,
                                             seed=1, p_mut=0, T_coop=1)
        engine.local_instance, engine.local_seed = "Tiny", 1
        states = []
        engine.local_capture_sink = states.append
        with patch.object(engine, "choose_guide", return_value=None), patch.object(engine, "choose_archive_guide_for_shortage", return_value=None):
            engine.run()
        self.assertTrue(states)
        self.assertEqual(engine.diagnostics["ps_moves"].get("shortage_relink", {}).get("generated", 0), 0)
        for state in states:
            self.assertGreater(state["S_seed"], 0)
            self.assertTrue(state["H"])
            self.assertEqual(state["diagnosis"]["unanchored_intervals"], 0)
            self.assertAlmostEqual(sum(value for op, value in state["phi"]), state["S_seed"])

    def test_invalid_diagnosis_not_captured(self):
        self.engine.local_instance, self.engine.local_seed = "Tiny", 1
        self.ind.shortage, self.ind.makespan = 10, 10
        self.ind.stats = {"shortage": {"per_buffer_shortage_area": {"B": 10}}}
        key = self.ind.os_seq[0]
        self.assertIsNone(v.capture_state(self.engine, self.ind, ShortageDiagnosis([], {key: 9}, set(), [], 0), {key}))
        self.assertIsNone(v.capture_state(self.engine, self.ind, ShortageDiagnosis([], {key: 10}, set(), [], 1), {key}))
        self.assertIsNone(v.capture_state(self.engine, self.ind, ShortageDiagnosis([], {key: 10}, set(), [], 0), set()))

    def test_offline_one_decode_per_child_fe_rng_pop_archive_unchanged(self):
        self.engine.initialize()
        decoded = r.prefix_offline_decode(self.engine.decoder, self.ind)
        self.state.update(S_seed=decoded["shortage"], Cmax_seed=decoded["makespan"], buffer_shortage=decoded["buffer_shortage"])
        before = v.formal_state_signature(self.engine)
        with patch.object(r, "prefix_offline_decode", wraps=r.prefix_offline_decode) as decode:
            rows, skipped = v.offline_trials(self.engine, [self.state])
        self.assertEqual(len(rows), 8)
        self.assertEqual(decode.call_count, 8)
        self.assertFalse(skipped)
        self.assertEqual(before, v.formal_state_signature(self.engine))

    def test_signed_buffer_deltas_and_relationships(self):
        state = {**self.state, "buffer_shortage": {"B": 6, "C": 4}}
        metadata = {"policy": "RandomLocal", "move_type": "MS", "absolute_displacement": None}
        row = v.trial_result(state, {"makespan": 11, "shortage": 8, "buffer_shortage": {"B": 3, "C": 5}},
                             metadata, 0, v.TrialPlan(0, 0, 0, 0))
        self.assertEqual(row["delta_S"], 2)
        self.assertEqual(row["total_positive_buffer_delta"], 3)
        self.assertEqual(row["total_negative_buffer_delta"], -1)
        self.assertEqual(row["changed_buffer_count"], 2)
        self.assertEqual(row["seed_child_relation"], "shortage_improves_makespan_worsens")
        with self.assertRaises(AssertionError):
            v.trial_result(state, {"makespan": 11, "shortage": 7, "buffer_shortage": {"B": 3, "C": 5}}, metadata, 0, v.TrialPlan(0, 0, 0, 0))

    def synthetic_rows(self):
        metadata = {"move_type": "MS", "absolute_displacement": None}
        rows = []
        for policy, ds_values in (("RandomLocal", [1, 1, 1, 1]), ("InfluenceLocal", [5, 0, 0, 0])):
            for trial, delta in enumerate(ds_values):
                rows.append(v.trial_result(self.state, {"shortage": 10 - delta, "makespan": 10,
                                                       "buffer_shortage": {"B": 10 - delta}},
                                           {**metadata, "policy": policy}, trial, v.TrialPlan(0, 0, 0, 0)))
        return rows

    def test_state_level_primary_unit_not_eight_independent_children(self):
        states, pairs = v.paired_summaries([self.state], self.synthetic_rows())
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["n_matched_trials"], 4)
        self.assertEqual(states[0]["paired_delta"], .25)
        primary = next(row for row in pairs if row["unit"] == "state" and row["scope"] == "overall")
        secondary = next(row for row in pairs if row["unit"] == "trial" and row["scope"] == "overall")
        self.assertEqual((primary["wins"], primary["n_pairs"]), (1, 1))
        self.assertEqual((secondary["wins"], secondary["losses"], secondary["n_pairs"]), (1, 3, 4))

    def test_policy_changed_and_conditional_magnitude(self):
        rows = self.synthetic_rows()[4:]
        summary = v.policy_summary(rows)
        self.assertEqual(summary["improve_rate"], .25)
        self.assertEqual(summary["shortage_changed_rate"], .25)
        self.assertEqual(summary["mean_abs_delta_S_conditional_changed"], 5)
        self.assertEqual(summary["mean_delta_S"], 1.25)

    def test_broken_pairs_rejected(self):
        with self.assertRaises(AssertionError):
            v.paired_summaries([self.state], self.synthetic_rows()[:-1])

    def test_capture_on_off_fixture_trajectory_exact(self):
        ops, buffers, _, _ = fixture()
        def run(cls, enabled):
            search = cls(ops, buffers, N=3, N_A=12, FE_max=30, seed=1, p_mut=0, T_coop=1)
            states = []
            if enabled:
                search.local_instance, search.local_seed = "Tiny", 1
                search.local_capture_sink = states.append
            trace = []
            def snapshot(engine):
                trace.append(v.formal_state_signature(engine))
            search.run(diagnostic_callback=snapshot)
            return trace, v.formal_state_signature(search), states
        off = run(WIPGraphDualPopulation, False)
        disabled = run(v.LocalStateCaptureAlgorithm, False)
        enabled = run(v.LocalStateCaptureAlgorithm, True)
        self.assertEqual(off[:2], disabled[:2])
        self.assertEqual(off[:2], enabled[:2])
        self.assertTrue(enabled[2])

    def test_formal_parameters_and_new_mode_only(self):
        s = v.local_settings()
        self.assertEqual(s.algorithm_parameters(0), {"N": 100, "N_A": 200, "p_mut": .1,
                                                    "rho": .8, "eta": .3, "T_coop": 10, "FE_max": 5000, "seed": 0})
        self.assertEqual(r.MODE, "run")
        self.assertEqual(r.ETA, .3)
        self.assertEqual(v.STATES_PER_TRAJECTORY * len(v.INSTANCES) * len(v.SEEDS), 240)
        self.assertNotEqual(v.OUTPUT_ROOT, r.OUTPUT_ROOT)


if __name__ == "__main__":
    unittest.main()
