"""Objective derivatives, binding Jacobian rules and read-only grid diagnostics."""

import copy
import unittest
from unittest.mock import patch

from experiments.diagnostics import shortage_sensitivity as v
from experiments import sis_local_viability as local
from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from tests.test_wip_graph_dual_population import fixture


class SensitivityTests(unittest.TestCase):
    def events(self, low=1):
        return {"B": [
            {"id": 0, "time": 0, "action": "init", "job": None, "before": 1, "after": 1, "cause": None},
            {"id": 1, "time": 2, "action": "take", "job": "A", "before": 1, "after": 0, "cause": 10},
            {"id": 2, "time": 8, "action": "put", "job": "B", "before": 0, "after": 1, "cause": 20},
            {"id": 3, "time": 10, "action": "take", "job": "B", "before": 1, "after": 0, "cause": 30},
        ]}, {"B": {"low": low, "start": 0, "end": 10}}

    def fd(self, events, windows, node):
        return (v.objective(events, windows, node, .001)-v.objective(events, windows))/.001

    def test_take_deficit_zero_to_one(self):
        events, windows = self.events()
        g, _, _ = v.event_geometry(events, windows)
        self.assertEqual(g[10], -1)
        self.assertAlmostEqual(self.fd(events, windows, 10), -1, places=8)

    def test_take_deficit_one_to_two_same_unit_derivative(self):
        events, windows = self.events(2)
        g, _, _ = v.event_geometry(events, windows)
        self.assertEqual(g[10], -1)
        self.assertAlmostEqual(self.fd(events, windows, 10), -1, places=8)

    def test_relief_deficit_two_to_one(self):
        events, windows = self.events(2)
        g, _, _ = v.event_geometry(events, windows)
        self.assertEqual(g[20], 1)
        self.assertAlmostEqual(self.fd(events, windows, 20), 1, places=8)

    def test_unchanged_deficit_zero(self):
        events, windows = self.events()
        for e in events["B"]:
            e["before"] += 2
            e["after"] += 2
        g, _, _ = v.event_geometry(events, windows)
        self.assertEqual(g[10], 0)
        self.assertEqual(self.fd(events, windows, 10), 0)

    def test_same_time_deficit_changes_nonsmooth(self):
        events, windows = self.events()
        events["B"][2]["time"] = 2
        _, nonsmooth, rows = v.event_geometry(events, windows)
        self.assertIn(10, nonsmooth)
        self.assertIn(20, nonsmooth)
        self.assertTrue(all(e["nonsmooth"] for e in rows if e["node"] in (10, 20)))

    def test_zero_g_transition_at_same_time_still_changes_logical_order(self):
        events, windows = self.events()
        events["B"].insert(3, {"id": 4, "time": 8, "action": "put", "job": "C", "before": 1, "after": 2, "cause": 40})
        g, nonsmooth, _ = v.event_geometry(events, windows)
        self.assertEqual(g[40], 0)
        self.assertIn(40, nonsmooth)
        self.assertIn(20, nonsmooth)

    def test_moving_active_end_boundary_correction(self):
        events, windows = self.events(2)
        g, _, rows = v.event_geometry(events, windows)
        self.assertEqual(g[30], 1)
        last = next(e for e in rows if e["node"] == 30)
        self.assertEqual(last["raw_g"], -1)
        self.assertEqual(last["moving_end_correction"], 2)
        self.assertAlmostEqual(self.fd(events, windows, 30), 1, places=8)

    def test_fixed_start_clipping(self):
        events, windows = self.events()
        windows["B"]["start"] = 3
        g, _, _ = v.event_geometry(events, windows)
        self.assertEqual(g[10], 0)
        self.assertEqual(self.fd(events, windows, 10), 0)

    def graph(self, edges, g):
        return {"order": sorted(g), "edges": [{"src": a, "dst": b, "alpha": alpha, "potential": bool(alpha)} for a, b, alpha in edges], "g": g, "nonsmooth": {}}

    def test_processing_s_to_c(self):
        graph = self.graph([(1, 2, 1)], {1: 0, 2: 3})
        self.assertEqual(v.reverse(graph)[1], 3)

    def test_unblocked_c_to_r(self):
        kind, rules, tied = v.release_dependency(5, 5, None)
        self.assertEqual((kind, rules, tied), ("unblocked", [("completion", 1)], False))

    def test_strict_blocked_c_to_r_zero(self):
        kind, rules, tied = v.release_dependency(5, 9, 10)
        self.assertEqual(kind, "strict_blocked")
        self.assertIn(("completion", 0), rules)
        self.assertFalse(tied)

    def test_unblocking_to_r_one(self):
        _, rules, _ = v.release_dependency(5, 9, 10)
        self.assertIn(("unblocking", 1), rules)

    def test_completion_unblocking_exact_tie(self):
        kind, rules, tied = v.release_dependency(5, 5, 10)
        self.assertTrue(tied)
        self.assertEqual(kind, "completion_unblocking_tie")
        self.assertTrue(all(alpha == 0 for name, alpha in rules))

    def test_unique_max_readiness(self):
        edges, tied, unknown = v.choose_binding([(1, 3, "job"), (2, 7, "machine")], 7)
        self.assertFalse(tied or unknown)
        self.assertEqual(next(e["alpha"] for e in edges if e["src"] == 2), 1)

    def test_nonbinding_readiness_zero(self):
        edges, _, _ = v.choose_binding([(1, 3, "job"), (2, 7, "machine")], 7)
        self.assertEqual(next(e["alpha"] for e in edges if e["src"] == 1), 0)

    def test_distinct_readiness_tie(self):
        edges, tied, _ = v.choose_binding([(1, 7, "job"), (2, 7, "machine")], 7)
        self.assertTrue(tied)
        self.assertTrue(all(e["alpha"] == 0 for e in edges))

    def test_same_variable_aliases_no_double_count(self):
        edges, tied, _ = v.choose_binding([(1, 7, "job"), (1, 7, "wip")], 7)
        self.assertFalse(tied)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["alpha"], 1)

    def test_forward_reverse_consistency(self):
        graph = self.graph([(1, 2, 1), (2, 3, 1)], {1: -1, 2: 0, 3: 4})
        shifts = v.forward(graph, {1: .001})
        self.assertAlmostEqual(sum(graph["g"][n]*dt for n, dt in shifts.items()), v.reverse(graph)[1]*.001)

    def test_two_independent_chains_no_double_count(self):
        graph = self.graph([(1, 2, 1), (3, 4, 1)], {1: 0, 2: 2, 3: 0, 4: 3})
        shifts = v.forward(graph, {1: 1})
        self.assertEqual(shifts[3], 0)
        self.assertEqual(shifts[4], 0)
        self.assertEqual(v.reverse(graph)[1], 2)

    def test_tied_potential_path_invalid_even_alpha_zero(self):
        graph = self.graph([(1, 2, 0)], {1: 0, 2: 1})
        graph["edges"][0]["potential"] = True
        graph["nonsmooth"][2] = ["tie"]
        self.assertTrue(v.invalid_path(graph, 1))

    def fixture_decode(self):
        ops, buffers, seq, ms = fixture()
        ind = Individual(seq, ms)
        decoded = v.normalized_decode(ops, buffers, ind)
        return ops, buffers, ind, decoded

    def test_real_graph_dag_time_logical_order_and_detachment(self):
        ops, buffers, ind, decoded = self.fixture_decode()
        old = copy.deepcopy(decoded)
        graph = v.graph_from_decode(decoded, ops, buffers, propagate=True)
        self.assertEqual(decoded, old)
        position = {node: i for i, node in enumerate(graph["order"])}
        self.assertTrue(all(position[e["src"]] < position[e["dst"]] for e in graph["edges"] if e["potential"]))

    def test_uniform_integer_grid_equivalence(self):
        ops, buffers, ind, decoded = self.fixture_decode()
        scaled = v.normalized_decode(v.scaled_operations(ops), buffers, ind, True)
        self.assertEqual(decoded["schedule"], scaled["schedule"])
        self.assertEqual(decoded["trace"], scaled["trace"])
        self.assertEqual(decoded["shortage"], scaled["shortage"])

    def test_literal_subinteger_processing_is_quantized(self):
        ops, buffers, ind, decoded = self.fixture_decode()
        perturbed = copy.deepcopy(ops)
        key = ind.os_seq[0]
        perturbed[key[0]][key[1]]["machines"][ind.ms_map[key]] += .001
        after = v.normalized_decode(perturbed, buffers, ind)
        self.assertEqual(decoded["schedule"], after["schedule"])

    def test_read_only_observer_off_on_trajectory_equivalence(self):
        ops, buffers, _, _ = fixture()
        off = WIPGraphDualPopulation(ops, buffers, N=3, N_A=12, FE_max=24, seed=7)
        on = WIPGraphDualPopulation(ops, buffers, N=3, N_A=12, FE_max=24, seed=7)
        checkpoints_off, checkpoints_on = [], []
        off.run(diagnostic_callback=lambda e: checkpoints_off.append(local.formal_state_signature(e)))
        original = on.decoder.decode
        observations = []
        def observer(*args, **kwargs):
            result = original(*args, **kwargs)
            m, schedule, trace, prov = result
            stats = on.decoder.analyze(schedule, trace, m)
            observations.append(v.graph_from_decode({"schedule": schedule, "provenance": prov, "stats": stats}, ops, buffers))
            return result
        with patch.object(on.decoder, "decode", side_effect=observer):
            on.run(diagnostic_callback=lambda e: checkpoints_on.append(local.formal_state_signature(e)))
        self.assertEqual(checkpoints_off, checkpoints_on)
        self.assertEqual(local.formal_state_signature(off), local.formal_state_signature(on))
        self.assertEqual(len(observations), 24)

    def test_source_operations_not_modified_by_grid_copy(self):
        ops, buffers, ind, decoded = self.fixture_decode()
        before = copy.deepcopy(ops)
        key = ind.os_seq[0]
        v.scaled_operations(ops, key, ind.ms_map[key], 1)
        self.assertEqual(ops, before)

    def test_parameter_boundary_input_is_not_lambda_c_only(self):
        ops, buffers, ind, _ = self.fixture_decode()
        excluded = v.parameter_boundary_keys(ops, buffers)
        self.assertTrue(excluded)


if __name__ == "__main__":
    unittest.main()
