"""Factual roles, matched short-range direction, isolated seed replay and effects."""

import copy
import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments import run_pre_ablation as r
from experiments import sis_local_viability as local
from experiments import sis_role_direction as v
from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from src.solution.decoder import BufferStateEvent, DecodeProvenance, StageBufferWIPScheduler
from tests.test_wip_graph_dual_population import fixture


class RoleDirectionTests(unittest.TestCase):
    def setUp(self):
        ops, buffers, os_seq, ms = fixture()
        self.engine = WIPGraphDualPopulation(ops, buffers, N=3, N_A=12, FE_max=30, seed=1)
        self.ind = Individual(os_seq, ms, operation_order=tuple(self.engine.encoder.ms_index_order))

    def original_state(self):
        decoded = v.decode_events(StageBufferWIPScheduler(self.engine.operations, self.engine.buffers), self.ind)
        x = self.ind.copy()
        x.makespan, x.shortage = decoded["makespan"], decoded["shortage"]
        x.stats, x.provenance = decoded["stats"], decoded["provenance"]
        x.schedule, x.buffer_trace = decoded["schedule"], decoded["trace"]
        verifier = WIPGraphDualPopulation(self.engine.operations, self.engine.buffers, N=3, N_A=12, FE_max=30, seed=1)
        d = verifier.diagnose_shortage(x)
        verifier.local_instance, verifier.local_seed = "Tiny", 1
        state = local.capture_state(verifier, x, d, verifier.high_influence_set(x, d))
        self.assertIsNotNone(state)
        return state, decoded

    def interval(self, area, anchors):
        return {"buffer_id": "B", "interval_id": int(area), "interval_area": area,
                "t_start": 0, "t_end": 10, "anchors": {"supply": anchors, "demand": anchors}}

    def anchor(self, op, time=1, event_id=1):
        return {"op": list(op), "event_time": time, "event_id": event_id,
                "deficit_pre": 1, "deficit_post": 0}

    def test_factual_roles_exact_formal_time_boundaries(self):
        prov = DecodeProvenance()
        ops = {"J": [{"buffer_out": "B"}, {"buffer_in": "B"}]}
        def event(t, role, cause=True):
            supply = role == "supply"
            kind, idx = ("release", 0) if supply else ("start", 1)
            node = prov.operation_event(t, kind, "J", idx, "M") if cause else None
            ev = BufferStateEvent(prov.next_id(), t, "B", "put" if supply else "take", "J",
                                  0 if supply else 1, 1 if supply else 0, node)
            prov.buffer_events.setdefault("B", []).append(ev)
            return node
        left_supply = event(2, "supply")
        left_demand = event(2, "demand")
        right_demand = event(6, "demand")
        right_supply = event(6, "supply")
        event(7, "supply")
        event(4, "supply", cause=False)
        state = {"diagnosis": {"intervals": [["B", 2, 6, 4, []]]}}
        decoded = {"provenance": prov, "stats": {"shortage": {"per_buffer_low_wip": {"B": 1}}}}
        intervals = v.factual_intervals(state, decoded, ops)
        self.assertEqual([a["event_id"] for a in intervals[0]["anchors"]["supply"]], [right_supply])
        self.assertEqual([a["event_id"] for a in intervals[0]["anchors"]["demand"]], [left_demand])

    def test_no_future_or_nearest_fabricated_anchor(self):
        state, decoded = self.original_state()
        empty = copy.deepcopy(state)
        empty["diagnosis"]["intervals"] = [["B", -5, -1, 4, []]]
        intervals = v.factual_intervals(empty, decoded, self.engine.operations)
        self.assertEqual(intervals[0]["anchors"], {"supply": [], "demand": []})

    def test_roles_have_deficit_change_and_correct_operation_events(self):
        state, decoded = self.original_state()
        intervals = v.factual_intervals(state, decoded, self.engine.operations)
        for interval in intervals:
            for role in ("supply", "demand"):
                for anchor in interval["anchors"][role]:
                    before, after = anchor["deficit_pre"], anchor["deficit_post"]
                    self.assertTrue(after < before if role == "supply" else after > before)
                    event = decoded["provenance"].events[anchor["event_id"]]
                    self.assertEqual(event.kind, "release" if role == "supply" else "start")

    def test_bounded_symmetric_os_minimum_displacement(self):
        op = ("J1", 0)
        options = v.directional_options(self.engine, self.ind, op, "supply_os")
        self.assertEqual(options["preferred"]["gap"], self.ind.os_seq.index(op) - 1)
        self.assertEqual(options["counter"]["gap"], self.ind.os_seq.index(op) + 1)
        for option in options.values():
            child = v.make_directional_child(self.engine, self.ind, op, option)
            self.engine.encoder.validate_os(child.os_seq)
            self.assertEqual(abs(child.os_seq.index(op) - self.ind.os_seq.index(op)), 1)
            self.assertEqual(child.ms_map, self.ind.ms_map)

    def test_demand_preferred_is_later_not_earlier(self):
        op = ("J1", 1)
        options = v.directional_options(self.engine, self.ind, op, "demand_os")
        self.assertEqual(options["preferred"]["action"], "later")
        self.assertEqual(options["counter"]["action"], "earlier")

    def test_one_sided_os_never_forced_into_pair(self):
        options = v.directional_options(self.engine, self.ind, ("J0", 0), "supply_os")
        self.assertEqual(set(options), {"counter"})
        intervals = [self.interval(100, [self.anchor(("J0", 0))])]
        selection = v.select_interval_action(self.engine, self.ind, intervals, "supply_os")
        self.assertEqual(selection["status"], "one_sided")

    def test_top_ineligible_interval_skipped_for_strict_pair(self):
        intervals = [self.interval(100, [self.anchor(("J0", 0))]),
                     self.interval(80, [self.anchor(("J1", 0))])]
        selection = v.select_interval_action(self.engine, self.ind, intervals, "supply_os")
        self.assertEqual(selection["status"], "strict_pair")
        self.assertEqual(selection["interval"]["interval_area"], 80)

    def test_deterministic_target_tie_not_phi_or_H(self):
        a = self.anchor(("J2", 0), time=2, event_id=7)
        b = self.anchor(("J1", 0), time=1, event_id=8)
        intervals = [self.interval(100, [a, b])]
        selection = v.select_interval_action(self.engine, self.ind, intervals, "supply_os")
        self.assertEqual(selection["anchor"]["op"], ["J1", 0])

    def test_ms_strict_fast_slow_and_machine_id_ties(self):
        op = ("J1", 0)
        self.engine.operations[op[0]][op[1]]["machines"] = {"Current": 10, "F2": 2, "F1": 2,
                                                             "S2": 20, "S1": 20, "Equal": 10}
        self.ind.ms_map[op] = "Current"
        options = v.directional_options(self.engine, self.ind, op, "supply_ms")
        self.assertEqual(options["preferred"]["machine"], "F1")
        self.assertEqual(options["counter"]["machine"], "S1")

    def test_equal_processing_time_not_directional_alternative(self):
        op = ("J1", 0)
        self.engine.operations[op[0]][op[1]]["machines"] = {"M1": 2, "Other": 2}
        self.assertEqual(v.directional_options(self.engine, self.ind, op, "supply_ms"), {})

    def test_faster_only_is_one_sided(self):
        op = ("J1", 0)
        self.ind.ms_map[op] = "M4"
        options = v.directional_options(self.engine, self.ind, op, "supply_ms")
        self.assertEqual(set(options), {"preferred"})

    def test_ms_child_changes_only_one_assignment_and_copies(self):
        old = copy.deepcopy(self.ind)
        child = v.make_directional_child(self.engine, self.ind, ("J1", 0), {"machine": "M4", "action": "slow"})
        self.assertEqual(child.os_seq, old.os_seq)
        self.assertEqual(sum(child.ms_map[op] != old.ms_map[op] for op in old.ms_map), 1)
        self.assertEqual(self.ind, old)
        self.assertIsNone(child.stats)

    def test_seed_fidelity_objectives_diagnosis_phi_H_and_dependencies(self):
        state, decoded = self.original_state()
        verifier = WIPGraphDualPopulation(self.engine.operations, self.engine.buffers, N=3, N_A=12, FE_max=6, seed=0)
        verified = v.verify_seed(state, decoded, verifier)
        self.assertEqual(verified.shortage, state["S_seed"])
        bad = copy.deepcopy(state)
        bad["phi"][0][1] += 1
        with self.assertRaises(AssertionError):
            v.verify_seed(bad, decoded, verifier)
        bad = copy.deepcopy(state)
        bad["S_seed"] += 1
        with self.assertRaises(AssertionError):
            v.verify_seed(bad, decoded, verifier)

    def test_one_seed_decode_one_per_constructed_child_and_no_formal_changes(self):
        state, decoded = self.original_state()
        self.engine.initialize()
        self.assertTrue(self.engine.P_M and self.engine.P_S and self.engine.A)
        verifier = WIPGraphDualPopulation(self.engine.operations, self.engine.buffers, N=3, N_A=12, FE_max=6, seed=0)
        before = local.formal_state_signature(self.engine)
        decoder = StageBufferWIPScheduler(self.engine.operations, self.engine.buffers)
        with patch.object(v, "decode_events", wraps=v.decode_events) as decode:
            rows, selected = v.run_state(self.engine, verifier, decoder, state)
        self.assertEqual(decode.call_count, 1 + len(rows))
        self.assertLessEqual(len(rows), 6)
        self.assertEqual(local.formal_state_signature(self.engine), before)
        self.assertEqual(verifier.n_evaluations, 0)
        self.assertEqual(len(selected["selections"]), 3)

    def basic_pair_rows(self, family="supply_os"):
        common = {"instance": "Tiny", "seed": 0, "fe_index": 12, "action_family": family,
                  "pair_status": "strict_pair", "target_op": ["J1", 0], "buffer_id": "B", "interval_id": 0,
                  "blocking_group": "unblocked", "old_position": 4, "displacement": 1,
                  "delta_S_target_buffer": 2, "delta_C": 0, "collateral_delta": -3,
                  "timing_class": "realized"}
        return [{**common, "direction": "preferred", "new_position": 3, "delta_S_total": -1},
                {**common, "direction": "counter", "new_position": 5, "delta_S_total": -4,
                 "delta_S_target_buffer": -1}]

    def test_positive_pair_gain_does_not_imply_seed_improvement(self):
        pairs = v.strict_pairs(self.basic_pair_rows())
        self.assertEqual(pairs[0]["pair_gain"], 3)
        self.assertEqual(pairs[0]["target_buffer_pair_gain"], 3)
        summary = v.pair_summary(pairs, ("action_family",))[0]
        self.assertEqual(summary["total_wins"], 1)
        self.assertEqual(v.effects(self.basic_pair_rows())["improve"], 0)

    def test_one_sided_excluded_from_primary_pairs(self):
        rows = self.basic_pair_rows()
        rows[0]["pair_status"] = rows[1]["pair_status"] = "one_sided"
        self.assertEqual(v.strict_pairs(rows), [])

    def test_invalid_asymmetric_radius_and_incomplete_pairs_rejected(self):
        for field, value in (("displacement", 2), ("new_position", 2)):
            rows = self.basic_pair_rows()
            rows[0][field] = value
            with self.assertRaises(AssertionError):
                v.strict_pairs(rows)
        with self.assertRaises(AssertionError):
            v.strict_pairs(self.basic_pair_rows()[:1])

    def test_target_buffer_collateral_is_not_total_effect(self):
        rows = self.basic_pair_rows()
        summary = v.effects(rows)
        self.assertEqual(summary["target_improved_total_worsened"], 1)
        self.assertEqual(summary["mean_delta_S_total"], -2.5)
        self.assertEqual(summary["mean_delta_S_target_buffer"], .5)

    def test_timing_realized_unchanged_opposite_for_preferred_and_counter(self):
        state, decoded = self.original_state()
        for family in v.FAMILIES:
            demand = family == "demand_os"
            op = ("J1", 1 if demand else 0)
            selection = {"interval": self.interval(1, []), "anchor": self.anchor(op),
                         "family": family, "role": "demand" if demand else "supply", "status": "strict_pair"}
            for direction in ("preferred", "counter"):
                sign = (1 if demand else -1) * (1 if direction == "preferred" else -1)
                for alignment, expected in ((1, "realized"), (0, "unchanged"), (-1, "opposite")):
                    child_decoded = copy.deepcopy(decoded)
                    kind = "start" if demand else "release"
                    event_id = child_decoded["provenance"].op_events[op][kind]
                    old = child_decoded["provenance"].events[event_id]
                    child_decoded["provenance"].events[event_id] = dataclasses.replace(old, time=old.time + sign * alignment)
                    row = v.action_row(self.engine, state, decoded, child_decoded, selection, direction,
                                       {"action": "test"}, self.ind.copy())
                    self.assertEqual(row["timing_class"], expected)
                    self.assertEqual(row["expected_timing_realized"], expected == "realized")

    def test_empty_subgroups_report_no_samples_not_zero_effect(self):
        rows = v.group_effects([], ("action_family", "direction"), [("supply_ms", "preferred")])
        self.assertEqual(rows[0]["n"], 0)
        self.assertIsNone(rows[0]["improve_rate"])
        pairs = v.pair_summary([], ("action_family",), [("supply_ms",)])
        self.assertEqual(pairs[0]["total_n_pairs"], 0)
        self.assertIsNone(pairs[0]["total_mean_paired_delta"])

    def test_no_algorithm_or_parameter_change_and_bounded_scope(self):
        self.assertEqual(v.ROLE_OS_RADIUS, 4)
        self.assertEqual(r.MODE, "run")
        self.assertEqual(r.ETA, .3)
        self.assertEqual(v.FAMILIES, ("supply_os", "demand_os", "supply_ms"))
        self.assertNotEqual(v.OUTPUT_ROOT, local.OUTPUT_ROOT)


if __name__ == "__main__":
    unittest.main()
