"""Exact contracts of pure shortcuts; no statistical trajectory comparison."""

import itertools
import unittest

from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from tests.test_wip_graph_dual_population import fixture


class EquivalentHelperTests(unittest.TestCase):
    def setUp(self):
        operations, buffers, os_seq, ms = fixture()
        self.search = WIPGraphDualPopulation(operations, buffers, N=3, N_A=12,
                                             FE_max=6, seed=1)
        self.os_seq, self.ms = os_seq, ms

    def test_existence_equals_complete_units_with_os_and_ms_evidence(self):
        self.search.initialize()
        for x in self.search.P_S:
            diagnosis = self.search.diagnose_shortage(x)
            high = self.search.high_influence_set(x, diagnosis)
            evidence = self.search._machine_shortage_evidence(x, diagnosis)
            for y in self.search.P_S + self.search.A:
                before = self.search.rng.getstate()
                expected = bool(self.search._build_relink_units(x, y, high, diagnosis, evidence))
                self.assertEqual(self.search._has_relink_unit(x, y, high, diagnosis, evidence), expected)
                self.assertEqual(self.search.rng.getstate(), before)
            self.assertFalse(self.search._has_relink_unit(x, x, set(), diagnosis, evidence))

    def test_self_guide_matches_full_filter_for_every_population_order(self):
        self.search.initialize()
        pool = self.search.P_S
        for x in pool:
            diagnosis = self.search.diagnose_shortage(x)
            high = self.search.high_influence_set(x, diagnosis)
            evidence = self.search._machine_shortage_evidence(x, diagnosis)
            for order in itertools.permutations(pool):
                eligible = [y for y in order if y is not x and y.shortage <= x.shortage
                            and self.search._build_relink_units(x, y, high, diagnosis, evidence)]
                better = [y for y in eligible if y.shortage < x.shortage]
                candidates = better or eligible
                expected = candidates[0] if candidates else None
                self.assertIs(self.search.choose_guide(x, order, high, diagnosis, evidence), expected)

    def test_reused_backtrace_index_preserves_node_edge_and_depth_order(self):
        self.search.initialize()
        for x in self.search.P_S:
            incoming = {}
            for edge in x.provenance.active_dependencies:
                incoming.setdefault(edge.dst_event_id, []).append(edge)
            anchors = set(list(x.provenance.events)[-4:])
            self.assertEqual(self.search._backtrace(x.provenance, anchors),
                             self.search._backtrace(x.provenance, anchors, incoming))

    def test_distance_same_job_fast_path_matches_explicit_pairs_even_for_nonlegal_os(self):
        operations = {job: [{"machines": {"M": 1}} for _ in range(2)] for job in ("A", "B")}
        search = WIPGraphDualPopulation(operations, {}, N=1, N_A=2, FE_max=2)
        keys = search.encoder.ms_index_order
        ms = {key: "M" for key in keys}
        pairs = [(a, b) for a, b in itertools.combinations(keys, 2) if a[0] != b[0]]
        for x_os in itertools.permutations(keys):
            for y_os in itertools.permutations(keys):
                rx = {key: i for i, key in enumerate(x_os)}
                ry = {key: i for i, key in enumerate(y_os)}
                count = sum((rx[a] < rx[b]) != (ry[a] < ry[b]) for a, b in pairs)
                self.assertEqual(search.structural_distance(Individual(list(x_os), ms),
                                                            Individual(list(y_os), ms)),
                                 0.5 * count / len(pairs))

    def test_distance_cache_observes_in_place_os_and_ms_changes(self):
        x = Individual(self.os_seq[:], self.ms.copy())
        y = x.copy()
        self.assertEqual(self.search.structural_distance(x, y), 0)
        y.os_seq[0], y.os_seq[1] = y.os_seq[1], y.os_seq[0]
        changed = self.search.structural_distance(x, y)
        self.assertGreater(changed, 0)
        y.ms_map[("J0", 0)] = "M4"
        self.assertEqual(self.search.structural_distance(x, y), changed + 0.5 / len(self.os_seq))
        y.os_seq[:] = x.os_seq
        y.ms_map.update(x.ms_map)
        self.assertEqual(self.search.structural_distance(x, y), 0)

    def test_distance_rank_cache_is_bounded(self):
        x = Individual(self.os_seq[:], self.ms.copy())
        for order in itertools.islice(itertools.permutations(self.os_seq), 180):
            self.search.structural_distance(x, Individual(list(order), self.ms.copy()))
        self.assertEqual(len(self.search._distance_rank_cache), 128)


if __name__ == "__main__":
    unittest.main()
