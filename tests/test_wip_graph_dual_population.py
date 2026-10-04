"""Small deterministic acceptance checks for provenance and dual search."""

import math
import os
import sys
import unittest
import warnings
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.algorithms.wip_graph_dual_population import (Individual, Move,
                                                       ShortageDiagnosis, WIPGraphDualPopulation)
from src.solution.decoder import (BufferStateEvent, DecodeProvenance, ProvenanceEvent,
                                  StageBufferWIPScheduler)
from src.solution.encoder import Encoder


def fixture():
    operations = {
        "J0": [
            {"machines": {"M0": 1, "M4": 2}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M2": 10, "M5": 11}, "buffer_in": "B", "buffer_out": None},
        ],
        "J1": [
            {"machines": {"M1": 2, "M4": 3}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M2": 1, "M5": 2}, "buffer_in": "B", "buffer_out": None},
        ],
        "J2": [
            {"machines": {"M3": 3, "M4": 4}, "buffer_in": None, "buffer_out": "B"},
            {"machines": {"M2": 1, "M5": 2}, "buffer_in": "B", "buffer_out": None},
        ],
    }
    buffers = {"B": {"capacity": 1, "low_wip": 1}}
    os_seq = [("J0", 0), ("J1", 0), ("J2", 0),
              ("J0", 1), ("J1", 1), ("J2", 1)]
    ms = Encoder(operations).build_ms_map(["M0", "M2", "M1", "M2", "M3", "M2"])
    return operations, buffers, os_seq, ms


class DecoderProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.ops, self.buffers, self.os, self.ms = fixture()
        self.decoder = StageBufferWIPScheduler(self.ops, self.buffers)

    def test_default_compatible_and_event_edges(self):
        old = self.decoder.decode(self.os, self.ms)
        new = self.decoder.decode(self.os, self.ms, return_provenance=True)
        self.assertEqual(old, new[:3])
        prov = new[3]
        self.assertIsInstance(prov, DecodeProvenance)
        ids = sorted(list(prov.events) + [e.event_id for values in prov.buffer_events.values() for e in values])
        self.assertEqual(ids, list(range(1, len(ids)+1)))
        edges = {(e.src_event_id, e.dst_event_id, e.kind) for e in prov.active_dependencies}
        for key, events in prov.op_events.items():
            self.assertEqual(set(events), {"start", "complete", "release"})
            self.assertIn((events["start"], events["complete"], "processing"), edges)
            self.assertIn((events["complete"], events["release"], "completion_release"), edges)
            self.assertEqual(prov.events[events["release"]].time,
                             next(rec["release"] for rec in new[1] if (rec["job"], rec["op"]) == key))

    def test_blocking_unblocking_and_exact_job_source(self):
        _, schedule, _, prov = self.decoder.decode(self.os, self.ms, True)
        rec = next(x for x in schedule if (x["job"], x["op"]) == ("J2", 0))
        self.assertGreater(rec["release"], rec["end"])
        blocked_release = prov.op_events[("J2", 0)]["release"]
        unblock = [e for e in prov.active_dependencies if e.kind == "unblocking"
                   and e.dst_event_id == blocked_release]
        self.assertEqual(len(unblock), 1)
        self.assertEqual(unblock[0].buffer_id, "B")
        self.assertEqual(prov.events[unblock[0].src_event_id].kind, "start")
        self.assertEqual(prov.events[unblock[0].src_event_id].time,
                         prov.events[blocked_release].time)
        self.assertLess(unblock[0].src_event_id, blocked_release)
        for event in prov.buffer_events["B"]:
            self.assertEqual(event.level_after - event.level_before,
                             1 if event.action == "put" else -1 if event.action == "take" else 0)
            if event.action == "put":
                self.assertEqual(event.cause_event_id,
                                 prov.op_events[(event.job, 0)]["release"])
            if event.action == "take":
                self.assertEqual(event.cause_event_id,
                                 prov.op_events[(event.job, 1)]["start"])
        wip = [e for e in prov.active_dependencies if e.kind == "wip"]
        self.assertTrue(wip)
        for edge in wip:
            self.assertEqual(prov.events[edge.src_event_id].job,
                             prov.events[edge.dst_event_id].job)
            self.assertEqual(prov.events[edge.src_event_id].time,
                             prov.events[edge.dst_event_id].time)

    def test_machine_edges_only_if_release_is_simultaneous(self):
        ops = {
            "A": [{"machines": {"M": 1}, "buffer_in": None, "buffer_out": None}],
            "B": [{"machines": {"M": 3}, "buffer_in": None, "buffer_out": None}],
        }
        dec = StageBufferWIPScheduler(ops, {})
        _, _, _, prov = dec.decode([("A", 0), ("B", 0)],
                                   {("A", 0): "M", ("B", 0): "M"}, True)
        edges = [e for e in prov.active_dependencies if e.kind == "machine"]
        self.assertEqual(len(edges), 1)
        self.assertEqual(prov.events[edges[0].src_event_id].time,
                         prov.events[edges[0].dst_event_id].time)
        # A's machine is idle after A finishes while B waits for its own WIP.
        ops = {
            "A": [{"machines": {"M": 1}, "buffer_in": None, "buffer_out": None}],
            "B": [{"machines": {"X": 3}, "buffer_in": None, "buffer_out": "Q"},
                  {"machines": {"M": 1}, "buffer_in": "Q", "buffer_out": None}],
        }
        dec = StageBufferWIPScheduler(ops, {"Q": {"capacity": 1}})
        _, _, _, prov = dec.decode([("A", 0), ("B", 0), ("B", 1)],
                                   {("A", 0): "M", ("B", 0): "X", ("B", 1): "M"}, True)
        self.assertFalse([e for e in prov.active_dependencies if e.kind == "machine"])

    def test_low_wip_ceil_default(self):
        d = StageBufferWIPScheduler(self.ops, {"B": {"capacity": 4}})
        m, s, t = d.decode(self.os, self.ms)
        self.assertEqual(d.analyze(s, t, m)["shortage"]["per_buffer_low_wip"]["B"], 2)


class SearchTests(unittest.TestCase):
    def setUp(self):
        ops, buffers, _, _ = fixture()
        self.search = WIPGraphDualPopulation(ops, buffers, N=3, N_A=4,
                                              FE_max=13, T_coop=1, gamma_A=0.5, seed=3)

    def test_shortage_mass_h_and_unanchored_diagnostic(self):
        self.search.initialize()
        for ind in self.search.P_S:
            diagnosis = self.search.diagnose_shortage(ind)
            self.assertAlmostEqual(sum(x[3] for x in diagnosis.intervals), ind.shortage)
            self.assertEqual(diagnosis.unanchored_intervals, 0)
            self.assertAlmostEqual(sum(diagnosis.phi.values()), ind.shortage)
            if ind.shortage:
                ordered = sorted(diagnosis.phi, key=lambda k: (-diagnosis.phi[k], k))
                cumulative = 0
                cut = 0
                while cumulative / ind.shortage < self.search.rho:
                    cumulative += diagnosis.phi[ordered[cut]]
                    cut += 1
                self.assertEqual(self.search.high_influence_set(ind, diagnosis), set(ordered[:cut]))
        artificial = Individual([], {}, shortage=1.0,
                                stats={"shortage": {"per_buffer_active_start": {"B": 0},
                                                    "per_buffer_active_end": {"B": 1},
                                                    "per_buffer_low_wip": {"B": 1}}},
                                provenance=DecodeProvenance(buffer_events={
                                    "B": [BufferStateEvent(1, 0, "B", "init", None, 0, 0)]}))
        with warnings.catch_warnings(record=True) as captured:
            diagnosis = self.search.diagnose_shortage(artificial)
        self.assertEqual(diagnosis.unanchored_intervals, 1)
        self.assertTrue(captured)

    def test_distance_and_precedence(self):
        self.search.initialize()
        x = self.search.P_M[0]
        self.assertEqual(self.search.structural_distance(x, x), 0)
        y = Individual(list(reversed(x.os_seq)), x.ms_map.copy())
        # Stage-wise distance is defined over permutations, independently of full OS positions.
        a = Individual([("J0", 0), ("J0", 1), ("J1", 0), ("J1", 1), ("J2", 0), ("J2", 1)], x.ms_map.copy())
        b = Individual([("J0", 0), ("J1", 0), ("J2", 0), ("J0", 1), ("J1", 1), ("J2", 1)], x.ms_map.copy())
        self.assertEqual(self.search.structural_distance(a, b), 0)
        d = self.search.structural_distance(x, y)
        self.assertGreaterEqual(d, 0)
        self.assertLessEqual(d, 1)
        self.assertEqual(d, self.search.structural_distance(y, x))
        for _ in range(30):
            child = self.search.basic_mutation(x)
            self.search.encoder.validate_os(child.os_seq)

    def test_relinking_is_restricted_and_not_evaluated_midway(self):
        self.search.initialize()
        x = max(self.search.P_S, key=lambda ind: ind.shortage)
        diagnosis = self.search.diagnose_shortage(x)
        high = self.search.high_influence_set(x, diagnosis)
        guide = min(self.search.P_S, key=lambda ind: ind.shortage)
        actions = self.search.relinking_actions(x, guide, high, diagnosis)
        self.assertTrue(all(a.key in high or (a.other in high if a.other else False)
                            for a in actions))
        before = self.search.n_evaluations
        child = self.search.vary_shortage(x, [guide])
        self.assertEqual(self.search.n_evaluations, before)
        self.search.encoder.validate_os(child.os_seq)
        self.search.evaluate(child)
        self.assertEqual(self.search.n_evaluations, before + 1)

    def test_similarity_replacement_and_archive(self):
        self.search.initialize()
        incumbent = self.search.P_M[0]
        better = Individual(incumbent.os_seq[:], incumbent.ms_map.copy(),
                            incumbent.makespan - 1, incumbent.shortage + 99)
        self.assertTrue(self.search.similarity_replace([incumbent], better, "makespan"))
        worse_shortage = Individual(incumbent.os_seq[:], incumbent.ms_map.copy(),
                                    incumbent.makespan - 1, incumbent.shortage + 1)
        self.assertFalse(self.search.similarity_replace([incumbent], worse_shortage, "shortage"))
        base = self.search.P_M[0]
        genotypes = [self.search.random_individual() for _ in range(20)]
        unique = list({self.search.genotype_key(z): z for z in genotypes}.values())
        self.assertGreaterEqual(len(unique), 4)
        for i, z in enumerate(unique):
            z.makespan, z.shortage = 10 + i, 20 - i
        unique[1].makespan, unique[1].shortage = unique[0].makespan, unique[0].shortage
        self.search.A = []
        self.search.update_archive(unique[:4] + [unique[0]])
        self.assertEqual(len(self.search.A), 4)
        self.assertEqual(len({self.search.genotype_key(z) for z in self.search.A}), 4)
        self.assertTrue(all(not self.search.dominates(a, b)
                            for a in self.search.A for b in self.search.A if a is not b))
        self.search.update_archive(unique[4:])
        self.assertLessEqual(len(self.search.A), self.search.N_A)
        self.assertIn(min(z.makespan for z in unique), [z.makespan for z in self.search.A])
        self.assertIn(min(z.shortage for z in unique), [z.shortage for z in self.search.A])

    def test_fe_budget_including_cooperation_and_final_partial_round(self):
        archive = self.search.run()
        self.assertEqual(self.search.n_evaluations, self.search.FE_max)
        self.assertTrue(archive)
        self.assertEqual(len(self.search.P_M), self.search.N)
        self.assertEqual(len(self.search.P_S), self.search.N)
        self.assertTrue(all(not self.search.dominates(a, b)
                            for a in archive for b in archive if a is not b))

    def test_shortage_interval_right_boundary_anchor_direction(self):
        def synthetic(action, before, after, event_kind):
            provenance = DecodeProvenance(
                events={2: ProvenanceEvent(2, 10, event_kind, "J0", 0, "M0")},
                buffer_events={"B": [
                    BufferStateEvent(1, 0, "B", "init", None, before, before),
                    BufferStateEvent(3, 10, "B", action, "J0", before, after, 2),
                ]})
            stats = {"shortage": {"per_buffer_active_start": {"B": 0},
                                  "per_buffer_active_end": {"B": 10},
                                  "per_buffer_low_wip": {"B": 2 if action == "take" else 1}}}
            return Individual([], {}, shortage=10.0, stats=stats, provenance=provenance)

        with warnings.catch_warnings(record=True):
            terminal_take = self.search.diagnose_shortage(synthetic("take", 1, 0, "start"))
        self.assertEqual(terminal_take.intervals[0][2], 10)
        self.assertEqual(terminal_take.intervals[0][3], 10.0)
        self.assertNotIn(2, terminal_take.influence_events)
        self.assertEqual(terminal_take.unanchored_intervals, 1)

        closing_put = self.search.diagnose_shortage(synthetic("put", 0, 1, "release"))
        self.assertEqual(closing_put.intervals[0][3], 10.0)
        self.assertIn(2, closing_put.influence_events)
        self.assertEqual(closing_put.unanchored_intervals, 0)

    def test_incomplete_phi_falls_back_before_h_and_relinking(self):
        self.search.p_mut = 0.0
        provenance = DecodeProvenance(buffer_events={
            "B": [BufferStateEvent(1, 0, "B", "init", None, 0, 0)]})
        x = Individual([], {}, shortage=1.0,
                       stats={"shortage": {"per_buffer_active_start": {"B": 0},
                                           "per_buffer_active_end": {"B": 1},
                                           "per_buffer_low_wip": {"B": 1}}},
                       provenance=provenance)
        marker = Individual([], {})
        with warnings.catch_warnings(record=True) as caught, \
             patch.object(self.search, "basic_mutation", return_value=marker) as basic, \
             patch.object(self.search, "high_influence_set", side_effect=AssertionError("H_S called")):
            result = self.search.vary_shortage(x, [])
        self.assertIs(result, marker)
        basic.assert_called_once_with(x)
        self.assertTrue(caught)
        self.assertGreater(self.search.unanchored_shortage_intervals, 0)

        complete_but_wrong_mass = ShortageDiagnosis([], {("J0", 0): 0.5}, set(), [])
        with warnings.catch_warnings(record=True) as caught, \
             patch.object(self.search, "diagnose_shortage", return_value=complete_but_wrong_mass), \
             patch.object(self.search, "basic_mutation", return_value=marker), \
             patch.object(self.search, "high_influence_set", side_effect=AssertionError("H_S called")):
            self.assertIs(self.search.vary_shortage(x, []), marker)
        self.assertTrue(caught)
        self.assertEqual(self.search.phi_mass_mismatch_count, 1)

    def test_machine_reassign_deduplicated_and_priority_span_recorded(self):
        ops = {"A": [{"machines": {"M": 1, "X": 2}, "buffer_in": None, "buffer_out": None}]}
        a = WIPGraphDualPopulation(ops, {}, N=1, N_A=2, FE_max=2, seed=1)
        p = DecodeProvenance()
        s = p.operation_event(0, "start", "A", 0, "M")
        c = p.operation_event(1, "complete", "A", 0, "M")
        r = p.operation_event(1, "release", "A", 0, "M")
        p.edge(s, c, "processing")
        p.edge(c, r, "completion_release")
        ind = Individual([("A", 0)], {("A", 0): "M"}, makespan=1,
                         provenance=p, operation_order=(("A", 0),))
        actions = [move for move in a.makespan_actions(ind) if move.kind == "machine_reassign"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].key, ("A", 0))
        self.assertEqual(actions[0].depth, 0)

        pair_ops = {name: [{"machines": {"M": 1}, "buffer_in": None, "buffer_out": None}]
                    for name in ("A", "B")}
        b = WIPGraphDualPopulation(pair_ops, {}, N=1, N_A=2, FE_max=2, seed=1, p_mut=0.0)
        seed = b.evaluate(Individual([("A", 0), ("B", 0)],
                                     {("A", 0): "M", ("B", 0): "M"},
                                     operation_order=(("A", 0), ("B", 0))))
        moved = b.vary_makespan(seed)
        self.assertNotEqual(seed.os_seq, moved.os_seq)
        self.assertEqual(b.machine_priority_move_spans, [1])

    def test_makespan_action_selection_uses_all_depths(self):
        self.search.p_mut = 0.0
        _, _, os_seq, ms_map = fixture()
        seed = Individual(os_seq, ms_map)
        actions = [Move("machine_priority", ("J1", 0), ("J0", 0), depth=7),
                   Move("machine_reassign", ("J0", 0), depth=0)]
        kinds = []
        with patch.object(self.search, "makespan_actions", return_value=actions):
            for random_seed in range(60):
                self.search.rng.seed(random_seed)
                self.search.vary_makespan(seed)
                kinds.append(self.search._last_pm_move_kind)
        self.assertEqual(set(kinds), {action.kind for action in actions})
        self.assertEqual(len(self.search.diagnostics["pm_action_sets"]), 60)
        for record in self.search.diagnostics["pm_action_sets"]:
            self.assertEqual(record, {"machine_priority": 1,
                                      "machine_reassign": 1, "total_actions": 2})
        for action in actions:
            self.assertEqual(set(self.search.diagnostics["pm_selected_depths"][action.kind]),
                             {action.depth})

    def test_makespan_fallback_and_mutation_probability(self):
        seed = Individual([], {})
        marker = Individual([], {})
        self.search.p_mut = 0.0
        with patch.object(self.search, "makespan_actions", return_value=[]), \
             patch.object(self.search, "basic_mutation", return_value=marker) as basic:
            self.assertIs(self.search.vary_makespan(seed), marker)
            basic.assert_called_once_with(seed)
        self.assertEqual(self.search._last_pm_move_kind, "fallback_basic_mutation")
        self.assertEqual(self.search.diagnostics["pm_action_sets"], [])
        self.search.p_mut = 1.0
        with patch.object(self.search, "makespan_actions", side_effect=AssertionError("called")), \
             patch.object(self.search, "basic_mutation", return_value=marker) as basic:
            self.assertIs(self.search.vary_makespan(seed), marker)
            basic.assert_called_once_with(seed)
        self.assertEqual(self.search._last_pm_move_kind, "basic_mutation")

    def test_makespan_variation_does_not_consume_fe(self):
        self.search.initialize()
        before = self.search.n_evaluations
        child = self.search.vary_makespan(self.search.P_M[0])
        self.assertEqual(self.search.n_evaluations, before)
        self.search.evaluate(child)
        self.assertEqual(self.search.n_evaluations, before + 1)

    def test_unblocking_propagates_to_machine_action_without_direct_move(self):
        ops = {name: [{"machines": {"M": 1, "X": 2} if name == "V" else {"M": 1},
                       "buffer_in": None, "buffer_out": None}]
               for name in ("W", "V", "U")}
        search = WIPGraphDualPopulation(ops, {}, N=1, N_A=2, FE_max=2, seed=1)
        prov = DecodeProvenance()
        release_w = prov.operation_event(10, "release", "W", 0, "M")
        start_v = prov.operation_event(10, "start", "V", 0, "M")
        release_u = prov.operation_event(10, "release", "U", 0, "M")
        prov.edge(release_w, start_v, "machine", machine="M")
        prov.edge(start_v, release_u, "unblocking", buffer_id="B")
        ind = Individual([("W", 0), ("V", 0), ("U", 0)],
                         {("W", 0): "M", ("V", 0): "M", ("U", 0): "M"},
                         makespan=10, provenance=prov)
        _, edges, _ = search._backtrace(prov, {release_u})
        self.assertTrue(any(edge.kind == "unblocking" for edge in edges))
        actions = search.makespan_actions(ind)
        self.assertTrue(all(move.kind in {"machine_priority", "machine_reassign"}
                            for move in actions))
        self.assertNotIn("unblocking_priority", {move.kind for move in actions})
        self.assertNotIn("unblocking_priority", search.diagnostics["pm_moves"])
        self.assertEqual({move.kind for move in actions},
                         {"machine_priority", "machine_reassign"})
        machine = next(move for move in actions if move.kind == "machine_priority")
        self.assertEqual((machine.key, machine.other), (("V", 0), ("W", 0)))
        self.assertTrue(search._pm_action_tags[id(machine)]["unblocking"])
        self.assertEqual(search.diagnostics["pm_propagation"]["unblocking_with_machine_pred"], 1)
        self.assertEqual(search.diagnostics["pm_propagation"]["unblocking_chain_reaches_actionable"], 1)

    def test_unblocking_wip_chain_reaches_upstream_action_and_deduplicates(self):
        ops = {name: [{"machines": {"M": 1, "X": 2} if name == "P" else {"M": 1},
                       "buffer_in": None, "buffer_out": None}]
               for name in ("W", "P", "V", "U1", "U2")}
        search = WIPGraphDualPopulation(ops, {}, N=1, N_A=2, FE_max=2, seed=1)
        prov = DecodeProvenance()
        release_w = prov.operation_event(10, "release", "W", 0, "M")
        start_p = prov.operation_event(10, "start", "P", 0, "M")
        complete_p = prov.operation_event(10, "complete", "P", 0, "M")
        release_p = prov.operation_event(10, "release", "P", 0, "M")
        start_v = prov.operation_event(10, "start", "V", 0, "M")
        release_u1 = prov.operation_event(10, "release", "U1", 0, "M")
        release_u2 = prov.operation_event(10, "release", "U2", 0, "M")
        prov.edge(release_w, start_p, "machine", machine="M")
        prov.edge(release_w, start_p, "machine", machine="M")
        prov.edge(start_p, complete_p, "processing")
        prov.edge(complete_p, release_p, "completion_release")
        prov.edge(release_p, start_v, "wip", buffer_id="B")
        prov.edge(start_v, release_u1, "unblocking", buffer_id="B")
        prov.edge(start_v, release_u2, "unblocking", buffer_id="B")
        keys = [(name, 0) for name in ("W", "P", "V", "U1", "U2")]
        ind = Individual(keys, {key: "M" for key in keys}, makespan=10, provenance=prov)
        actions = search.makespan_actions(ind)
        machine = [move for move in actions if move.kind == "machine_priority"]
        reassignment = [move for move in actions if move.kind == "machine_reassign"]
        self.assertEqual(len(machine), 1)
        self.assertEqual(len(reassignment), 1)
        self.assertEqual(machine[0].key, ("P", 0))
        self.assertEqual(reassignment[0].key, ("P", 0))
        self.assertTrue(all(search._pm_action_tags[id(move)]["unblocking"] for move in actions))
        self.assertTrue(all(search._pm_action_tags[id(move)]["wip"] for move in actions))
        propagation = search.diagnostics["pm_propagation"]
        self.assertEqual(propagation["unblocking_edges_total"], 2)
        self.assertEqual(propagation["unblocking_with_wip_pred"], 2)
        self.assertEqual(propagation["unblocking_chain_reaches_actionable"], 2)

    def test_archive_shortage_guide_filters_validity_before_sparsity(self):
        x = Individual([], {}, shortage=9.0)
        make = lambda m, s: Individual([], {}, makespan=m, shortage=s)
        extreme_m = make(10, 10)
        invalid_sparse = make(20, 8)
        valid = make(30, 5)
        extreme_s = make(40, 2)
        self.search.A = [extreme_m, invalid_sparse, valid, extreme_s]
        diagnosis = ShortageDiagnosis([], {("J0", 0): 9.0}, set(), [])
        high = {("J0", 0)}
        fake_crowd = {id(extreme_m): math.inf, id(invalid_sparse): 100.0,
                      id(valid): 1.0, id(extreme_s): math.inf}
        with patch.object(self.search, "_crowding", return_value=fake_crowd), \
             patch.object(self.search, "relinking_actions",
                          side_effect=lambda _, y, _h, _d: [Move("os", ("J0", 0))] if y is valid else []):
            self.assertIs(self.search.choose_archive_guide_for_shortage(x, diagnosis, high), valid)

        second_valid = make(35, 4)
        self.search.A.insert(-1, second_valid)
        fake_crowd[id(second_valid)] = 2.0
        with patch.object(self.search, "_crowding", return_value=fake_crowd), \
             patch.object(self.search, "relinking_actions",
                          side_effect=lambda _, y, _h, _d: [Move("os", ("J0", 0))]
                          if y is valid or y is second_valid else []), \
             patch.object(self.search.rng, "sample", return_value=[valid, second_valid]):
            self.assertIs(self.search.choose_archive_guide_for_shortage(x, diagnosis, high),
                          second_valid)

        with patch.object(self.search, "relinking_actions", return_value=[]):
            self.assertIsNone(self.search.choose_archive_guide_for_shortage(x, diagnosis, high))

        self.search.p_mut = 0.0
        self.search.P_S = [x]
        marker = Individual([], {})
        with patch.object(self.search, "diagnose_shortage", return_value=diagnosis), \
             patch.object(self.search, "choose_archive_guide_for_shortage", return_value=None), \
             patch.object(self.search, "choose_guide", return_value=None) as self_guide, \
             patch.object(self.search, "basic_mutation", return_value=marker):
            self.assertIs(self.search.vary_shortage(x, self.search.P_S, archive_assisted=True), marker)
        self.assertIs(self_guide.call_args.args[1], self.search.P_S)

    def test_archive_sparse_binary_tournament_is_not_fixed_argmax(self):
        make = lambda m, s: Individual([], {}, makespan=m, shortage=s)
        first, high, medium, low, last = (make(10, 10), make(20, 8), make(30, 6),
                                           make(40, 4), make(50, 2))
        self.search.A = [first, high, medium, low, last]
        crowd = {id(first): math.inf, id(high): 5.0, id(medium): 3.0,
                 id(low): 1.0, id(last): math.inf}
        with patch.object(self.search, "_crowding", return_value=crowd), \
             patch.object(self.search.rng, "sample", side_effect=[[high, medium], [medium, low]]):
            self.assertIs(self.search.archive_guide(), high)
            self.assertIs(self.search.archive_guide(), medium)

    def test_tournament_primary_tie_uses_novelty_not_secondary(self):
        _, _, os_seq, ms = fixture()
        a = Individual(os_seq[:], ms.copy(), makespan=10, shortage=1.0)
        b_os = [("J0", 0), ("J2", 0), ("J1", 0),
                ("J0", 1), ("J1", 1), ("J2", 1)]
        b = Individual(b_os, ms.copy(), makespan=10, shortage=2.0)
        c_os = [("J2", 0), ("J1", 0), ("J0", 0),
                ("J0", 1), ("J1", 1), ("J2", 1)]
        c = Individual(c_os, ms.copy(), makespan=10, shortage=9.0)
        self.assertGreater(min(self.search.structural_distance(c, z) for z in (a, b)),
                           min(self.search.structural_distance(a, z) for z in (b, c)))
        with patch.object(self.search.rng, "sample", return_value=[a, c]):
            self.assertIs(self.search.tournament([a, b, c], "makespan"), c)
        a.shortage = b.shortage = c.shortage = 5.0
        a.makespan, c.makespan = 1, 99
        with patch.object(self.search.rng, "sample", return_value=[a, c]):
            self.assertIs(self.search.tournament([a, b, c], "shortage"), c)

    def test_initialization_diversity_window_and_ms_export_order(self):
        self.search.initialize()
        self.assertEqual(len(self.search.P_M), self.search.N)
        self.assertEqual(len(self.search.P_S), self.search.N)
        first = self.search.P_M[0]
        scrambled = dict(reversed(list(first.ms_map.items())))
        ordered = Individual(first.os_seq[:], scrambled,
                             operation_order=tuple(self.search.encoder.ms_index_order))
        self.assertEqual(ordered.MS,
                         [first.ms_map[key] for key in self.search.encoder.ms_index_order])
        self.assertEqual(first.MS, ordered.MS)
        self.assertEqual(first.copy().MS, first.MS)
        with self.assertRaises(ValueError):
            Individual(first.os_seq[:], scrambled).MS

        pool = [Individual(first.os_seq[:], first.ms_map.copy(), makespan=m, shortage=s)
                for m, s in [(10, 50), (11, 40), (12, 1), (12, 2), (12, 99)]]
        def distance(z, selected):
            return 1.0 if z.shortage == 99 else 0.1
        with patch.object(self.search, "structural_distance", side_effect=distance):
            chosen = self.search._diverse_initial_population(pool, "makespan")
        self.assertEqual([z.makespan for z in chosen], [10, 11, 12])
        self.assertEqual(chosen[-1].shortage, 99)

        shortage_pool = [Individual(first.os_seq[:], first.ms_map.copy(), makespan=m, shortage=s)
                         for m, s in [(50, 10), (40, 11), (1, 12), (2, 12), (99, 12)]]
        with patch.object(self.search, "structural_distance",
                          side_effect=lambda z, _: 1.0 if z.makespan == 99 else 0.1):
            selected = self.search._diverse_initial_population(shortage_pool, "shortage")
        self.assertEqual([z.shortage for z in selected], [10, 11, 12])
        self.assertEqual(selected[-1].makespan, 99)

    def test_opening_context_respects_same_time_event_id(self):
        p = DecodeProvenance(
            events={1: ProvenanceEvent(1, 0, "release", "J0", 0, "M0"),
                    4: ProvenanceEvent(4, 2, "start", "J1", 0, "M1")},
            buffer_events={"B": [
                BufferStateEvent(2, 0, "B", "put", "J0", 0, 1, 1),
                BufferStateEvent(3, 2, "B", "take", "J0", 1, 0, None),
                BufferStateEvent(5, 2, "B", "init", None, 0, 0, 4),
            ]})
        x = Individual([], {}, shortage=8.0,
                       stats={"shortage": {"per_buffer_active_start": {"B": 2},
                                           "per_buffer_active_end": {"B": 10},
                                           "per_buffer_low_wip": {"B": 1}}},
                       provenance=p)
        diagnosis = self.search.diagnose_shortage(x)
        self.assertEqual(diagnosis.unanchored_intervals, 0)
        self.assertIn(1, diagnosis.influence_events)
        self.assertNotIn(4, diagnosis.influence_events)

    def test_active_start_positive_context_excludes_same_time_future(self):
        p = DecodeProvenance(
            events={4: ProvenanceEvent(4, 2, "start", "J1", 0, "M1")},
            buffer_events={"B": [
                BufferStateEvent(1, 0, "B", "init", None, 0, 0),
                BufferStateEvent(5, 2, "B", "init", None, 0, 0, 4),
            ]})
        x = Individual([], {}, shortage=8.0,
                       stats={"shortage": {"per_buffer_active_start": {"B": 2},
                                           "per_buffer_active_end": {"B": 10},
                                           "per_buffer_low_wip": {"B": 1}}},
                       provenance=p)
        with warnings.catch_warnings(record=True) as caught:
            diagnosis = self.search.diagnose_shortage(x)
        self.assertEqual(diagnosis.unanchored_intervals, 1)
        self.assertNotIn(4, diagnosis.influence_events)
        self.assertTrue(caught)

    def test_compare_runner_export_contract(self):
        import experiments.run_compare_experiments as runner
        ops, buffers, _, _ = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tiny.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"spec": {"name": "tiny"}, "operations": ops,
                           "buffers": buffers}, handle)
            with patch.object(runner, "POP_SIZE", 4), patch.object(runner, "MAX_EVALUATIONS", 5):
                result = runner.run_once(path, 1, "WIPGraphDualPopulation")
        self.assertEqual(result["representative_result"]["n_evaluations"], 5)
        self.assertTrue(result["pareto_front"])
        for sol in result["pareto_front"]:
            self.assertEqual(len(sol["OS"]), len(sol["MS"]))
            self.assertEqual(set(sol["OS"]), set(Encoder(ops).ms_index_order))

        from experiments.analyze_pareto_solution import evaluate_solution
        restored = json.loads(json.dumps(result["pareto_front"][0]))
        decoded = evaluate_solution(ops, buffers, restored)
        self.assertEqual(decoded["makespan"], restored["makespan"])
        self.assertEqual(decoded["stats"]["shortage"]["total_shortage_area"],
                         restored["shortage"])

    def test_behavior_diagnostics_script_and_callback(self):
        from experiments.validate_wip_graph_dual_population_behavior import run_case
        ops, buffers, _, _ = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tiny.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"spec": {"name": "tiny"}, "operations": ops,
                           "buffers": buffers}, handle)
            result = run_case(Path(path), seed=2, N=3, N_A=4, FE_max=13)
        self.assertEqual(result["fe"], 13)
        self.assertEqual(result["diversity_checkpoints"]["initial"]["fe"], 6)
        self.assertEqual(result["diversity_checkpoints"]["100%"]["fe"], 13)
        generated = sum(v["generated"] for group in ("pm_moves", "ps_moves")
                        for v in result[group].values())
        self.assertEqual(generated, 7)
        self.assertEqual(sum(result["source_generated"].values()), 7)
        self.assertLessEqual(result["final_archive"]["size"], 4)


if __name__ == "__main__":
    unittest.main()
