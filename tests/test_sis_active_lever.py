"""Exact decision observation, factual competition, causal tracing and ECT controls."""

import copy
import sys
import unittest
from unittest.mock import patch

from experiments import run_pre_ablation as r
from experiments import sis_local_viability as local
from experiments import sis_role_direction as role
from experiments import sis_active_lever as v
from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from src.solution.decoder import StageBufferWIPScheduler
from tests.test_wip_graph_dual_population import fixture


class ActiveLeverTests(unittest.TestCase):
    def setUp(self):
        ops, buffers, os_seq, ms = fixture()
        for job in ops:
            ms[(job, 0)] = "M4"
        self.engine = WIPGraphDualPopulation(ops, buffers, N=3, N_A=12, FE_max=30, seed=1)
        self.ind = Individual(os_seq, ms, operation_order=tuple(self.engine.encoder.ms_index_order))

    def seed(self):
        snapshots = []
        with v.observe_start_decisions(snapshots.append):
            decoded = role.decode_events(self.engine.decoder, self.ind)
        x = self.ind.copy()
        x.makespan, x.shortage = decoded["makespan"], decoded["shortage"]
        x.provenance, x.stats = decoded["provenance"], decoded["stats"]
        x.schedule, x.buffer_trace = decoded["schedule"], decoded["trace"]
        verifier = WIPGraphDualPopulation(self.engine.operations, self.engine.buffers, N=3, N_A=12, FE_max=30, seed=0)
        diagnosis = verifier.diagnose_shortage(x)
        verifier.local_instance, verifier.local_seed = "Tiny", 1
        state = local.capture_state(verifier, x, diagnosis, verifier.high_influence_set(x, diagnosis))
        return state, decoded, snapshots

    def test_observer_reads_real_candidates_and_individual_same_time_choices(self):
        state, decoded, snapshots = self.seed()
        nonempty = [s for s in snapshots if s["chosen_op"]]
        self.assertEqual(len(nonempty), len(self.ind.os_seq))
        self.assertEqual(nonempty[0]["chosen_op"], ("J0", 0, "M4"))
        self.assertEqual(v.competitors(nonempty[0], ("J0", 0), "M4"), [("J1", 0), ("J2", 0)])
        self.assertEqual(len({s["decision_seq"] for s in snapshots}), len(snapshots))
        for s in nonempty:
            self.assertEqual(tuple(s["chosen_op"]), min(s["feasible_ops_before_choice"], key=lambda c: c[3])[:3])
            self.assertTrue(all(all(c[4:]) for c in s["feasible_ops_before_choice"]))

    def test_observer_off_on_decode_identical_and_detached(self):
        off = role.decode_events(self.engine.decoder, self.ind)
        snapshots = []
        def collect(s):
            snapshots.append(s)
            s["chosen_op"] = ("Fake", -1, "Fake")
        with v.observe_start_decisions(collect):
            on = role.decode_events(self.engine.decoder, self.ind)
        self.assertEqual(off, on)
        self.assertIsNone(sys.gettrace())

    def test_observer_restored_even_if_callback_fails(self):
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with v.observe_start_decisions(lambda s: (_ for _ in ()).throw(RuntimeError("probe"))):
                self.engine.decoder.decode(self.ind.os_seq, self.ind.ms_map)
        self.assertIsNone(sys.gettrace())

    def test_factual_priority_requires_delayed_op_in_winner_snapshot(self):
        state, decoded, snapshots = self.seed()
        context = v.active_context(self.engine, decoded, snapshots)
        self.assertEqual(context[3][("J1", 0)]["factual_priority_competitors"], [["J0", 0]])
        fake = copy.deepcopy(snapshots)
        first = next(s for s in fake if s["chosen_op"] == ("J0", 0, "M4"))
        first["feasible_ops_before_choice"] = tuple(c for c in first["feasible_ops_before_choice"] if c[:2] != ("J1", 0))
        context = v.active_context(self.engine, decoded, fake)
        self.assertEqual(context[3][("J1", 0)]["factual_priority_competitors"], [])

    def test_immediate_priority_relation_changes_one_operation_preserves_route(self):
        for side, delayed, other in (("supply", ("J1", 0), ("J0", 0)), ("demand", ("J0", 0), ("J1", 0))):
            candidate = v.priority_move(self.engine, self.ind, delayed, other, side)
            self.assertIsNotNone(candidate)
            child = v.construct_child(self.engine, self.ind, side + "_AMP", candidate)
            moved = tuple(candidate["moved_op"])
            self.assertEqual([op for op in child.os_seq if op != moved], [op for op in self.ind.os_seq if op != moved])
            self.assertEqual(child.ms_map, self.ind.ms_map)
            self.engine.encoder.validate_os(child.os_seq)

    def test_blocked_release_traces_unblocking_not_accelerating_completed_anchor(self):
        state, decoded, snapshots = self.seed()
        blocked = [rec for rec in decoded["schedule"] if rec["release"] > rec["end"]]
        if not blocked:
            self.skipTest("Fixture has no blocked release")
        rec = blocked[0]
        op = (rec["job"], rec["op"])
        anchor = {"op": list(op), "event_id": decoded["provenance"].op_events[op]["release"], "event_time": rec["release"]}
        interval = {"anchors": {"supply": [anchor], "demand": []}}
        inspected = []
        with patch.object(v, "ect_options", side_effect=lambda e, i, d, key: inspected.append(key)):
            trace = v.trace_first(self.engine, self.ind, decoded, v.active_context(self.engine, decoded, snapshots), interval, "AMR")
        self.assertNotIn(op, inspected)
        self.assertTrue(trace["examined"][0]["blocked_release"])

    def test_demand_does_not_trace_predecessors(self):
        state, decoded, snapshots = self.seed()
        op = ("J2", 1)
        anchor = {"op": list(op), "event_id": decoded["provenance"].op_events[op]["start"],
                  "event_time": role.event_times(decoded, op)["start"]}
        context = v.active_context(self.engine, decoded, snapshots)
        with patch.object(v, "competitors", return_value=[]):
            trace = v.trace_first(self.engine, self.ind, decoded, context,
                                  {"anchors": {"demand": [anchor]}}, "demand_AMP")
        self.assertEqual(trace["status"], "no_action")
        self.assertEqual(len(trace["examined"]), 1)

    def test_earliest_slot_includes_blocking_and_full_duration(self):
        schedule = [{"job": "A", "op": 0, "machine": "M", "start": 0, "end": 2, "release": 8},
                    {"job": "B", "op": 0, "machine": "M", "start": 12, "end": 14, "release": 14}]
        self.assertEqual(v.earliest_slot(schedule, ("X", 0), "M", 3, 3), 8)
        self.assertEqual(v.earliest_slot(schedule, ("X", 0), "M", 3, 5), 14)

    def test_current_ect_excludes_own_occupancy(self):
        schedule = [{"job": "A", "op": 0, "machine": "M", "start": 0, "end": 10, "release": 10}]
        self.assertEqual(v.earliest_slot(schedule, ("A", 0), "M", 0, 5), 0)

    def test_readiness_uses_actual_predecessor_and_exact_job_supply(self):
        state, decoded, snapshots = self.seed()
        ready, prerequisites = v.readiness(decoded, self.engine.operations, self.engine.buffers, ("J1", 1))
        self.assertEqual(ready, role.event_times(decoded, ("J1", 0))["release"])
        self.assertTrue(any(p[0] == "wip" for p in prerequisites))
        self.assertEqual(v.readiness(decoded, self.engine.operations, self.engine.buffers, ("J0", 0))[0], 0)

    def test_ect_is_strict_and_not_just_fastest(self):
        state, decoded, snapshots = self.seed()
        op = ("J1", 0)
        with patch.object(v, "earliest_slot", side_effect=lambda schedule, key, machine, ready, duration: 100 if machine == "M1" else 0):
            self.assertIsNone(v.ect_options(self.engine, self.ind, decoded, op))
        candidate = v.ect_options(self.engine, self.ind, decoded, ("J0", 0))
        self.assertGreater(candidate["predicted_completion_gain"], 0)

    def test_control_same_moved_op_exact_displacement_and_private_rng(self):
        state, decoded, snapshots = self.seed()
        candidate = v.priority_move(self.engine, self.ind, ("J1", 0), ("J0", 0), "supply")
        before = self.engine.rng.getstate()
        rng = v.diagnostic_rng(state, {"interval_id": 1}, v.action_key("supply_AMP", candidate))
        control = v.amp_control(self.engine, self.ind, candidate, rng)
        self.assertEqual(control["moved_op"], candidate["moved_op"])
        self.assertNotEqual(control["gap"], candidate["gap"])
        self.assertTrue(control["exact_displacement_match"])
        self.assertEqual(self.engine.rng.getstate(), before)

    def test_interval_coverage_prefers_different_buffers(self):
        intervals = [{"buffer_id": bid, "interval_area": area} for bid, area in (("B1", 10), ("B1", 9), ("B2", 8), ("B3", 7))]
        self.assertEqual([i["buffer_id"] for i in v.selected_intervals(intervals)], ["B1", "B2", "B3"])

    def test_no_action_does_not_fallback_to_random(self):
        state, decoded, snapshots = self.seed()
        trace = v.trace_first(self.engine, self.ind, decoded, v.active_context(self.engine, decoded, snapshots),
                              {"anchors": {"supply": []}}, "AMR")
        self.assertEqual(trace["status"], "no_action")
        self.assertNotIn("candidate", trace)

    def test_action_key_deduplicates_identical_os_action_across_roles(self):
        candidate = {"moved_op": ["J1", 0], "gap": 0}
        self.assertEqual(v.action_key("supply_AMP", candidate), v.action_key("demand_AMP", candidate))

    def test_offline_isolation_and_identical_genotype_decode_cache(self):
        state, decoded, snapshots = self.seed()
        self.engine.initialize()
        before = local.formal_state_signature(self.engine)
        verifier = WIPGraphDualPopulation(self.engine.operations, self.engine.buffers, N=3, N_A=12, FE_max=6, seed=0)
        with patch.object(role, "decode_events", wraps=role.decode_events) as decode:
            rows, starts, trace = v.offline_state(self.engine, verifier, self.engine.decoder, state)
        self.assertEqual(decode.call_count, 1 + trace["child_decodes"])
        self.assertEqual(local.formal_state_signature(self.engine), before)
        self.assertGreaterEqual(len(rows), trace["child_decodes"])

    def test_state_aggregation_not_all_actions_independent(self):
        base = {"instance": "X", "seed": 0, "fe_index": 1, "family": "AMR",
                "delta_S_buffer": 0, "collateral_delta": 0, "local_realized": False, "anchor_realized": False}
        rows = []
        for action_id, delta in enumerate((3, -1, -1)):
            rows.extend([{**base, "action_id": action_id, "policy": "ECT_AMR", "delta_S": delta},
                         {**base, "action_id": action_id, "policy": "RandomMS", "delta_S": 0}])
        pairs, summaries = v.comparisons(rows)
        primary = next(s for s in summaries if s["scope"] == "overall" and s["unit"] == "state")
        self.assertEqual(primary["states"], 1)
        self.assertEqual(primary["wins"], 1)
        self.assertEqual(primary["action_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
