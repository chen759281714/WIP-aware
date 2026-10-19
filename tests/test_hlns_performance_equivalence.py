"""Exact code-optimization contracts; no performance-quality experiment."""

import random
import unittest
from unittest.mock import patch

from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from src.solution.decoder import ActiveDependency, StageBufferWIPScheduler
from src.solution.encoder import Encoder
from tests.hlns_test_support import fixture, make


def greedy_reference(e, pool, objective):
    ordered = sorted(pool, key=lambda z: getattr(z, objective))
    selected = []
    index = 0
    while len(selected) < e.N:
        value = getattr(ordered[index], objective)
        group = []
        while index < len(ordered) and getattr(ordered[index], objective) == value:
            group.append(ordered[index]); index += 1
        slots = min(len(group), e.N-len(selected))
        for _ in range(slots):
            if selected:
                choice = max(group, key=lambda z: (
                    min(e.structural_distance(z, prior) for prior in selected), e.genotype_key(z)))
            else:
                choice = min(group, key=e.genotype_key)
            selected.append(choice)
            group = [z for z in group if z is not choice]
    return selected


class IncrementalDiversityTests(unittest.TestCase):
    def test_identical_order_with_ties_and_prior_groups(self):
        ops, buffers, _, _ = fixture()
        for mode in ('all_tied', 'groups', 'unique'):
            e = WIPGraphDualPopulation(ops, buffers, variant='HLNS_FULL', N=8, N_A=16, FE_max=16, seed=4)
            pool = [e.random_individual() for _ in range(16)]
            for i, ind in enumerate(pool):
                ind.shortage = 0 if mode == 'all_tied' else (i // 4 if mode == 'groups' else i)
            expected = greedy_reference(e, pool, 'shortage')
            actual = e._diverse_initial_population(pool, 'shortage')
            self.assertEqual([id(z) for z in expected], [id(z) for z in actual])

    def test_fewer_distance_calls_without_rng(self):
        ops, buffers, _, _ = fixture()
        e = WIPGraphDualPopulation(ops, buffers, variant='HLNS_FULL', N=8, N_A=16, FE_max=16, seed=4)
        pool = [e.random_individual() for _ in range(16)]
        for ind in pool:
            ind.shortage = 0
        state = e.rng.getstate()
        with patch.object(e, 'structural_distance', wraps=e.structural_distance) as distance:
            greedy_reference(e, pool, 'shortage'); before = distance.call_count
            distance.reset_mock()
            e._diverse_initial_population(pool, 'shortage'); after = distance.call_count
        self.assertLess(after, before)
        self.assertEqual(e.rng.getstate(), state)


class MGSFeasibilityTests(unittest.TestCase):
    def test_predicate_matches_actual_neighbor_all_pairs(self):
        e, _ = make()
        for _ in range(60):
            ind = e.random_individual()
            positions = {key: i for i, key in enumerate(ind.os_seq)}
            for key in ind.os_seq:
                for other in ind.os_seq:
                    self.assertEqual(e._relative_move_is_legal(key, other, positions, True),
                                     e._move_before(ind, key, other) is not None)
                    self.assertEqual(e._relative_move_is_legal(key, other, positions, False),
                                     e._move_after(ind, key, other) is not None)

    def test_enumeration_does_not_construct_children(self):
        e, p = make(); e.evaluate(p)
        with patch.object(e, '_move_before', side_effect=AssertionError('temporary child')), \
             patch.object(e, '_move_after', side_effect=AssertionError('temporary child')):
            e.makespan_actions(p)


class CacheTests(unittest.TestCase):
    def test_OS_and_MS_inplace_changes_observed(self):
        e, x = make()
        y = x.copy()
        self.assertEqual(e.structural_distance(x, y), 0)
        y.os_seq[0], y.os_seq[1] = y.os_seq[1], y.os_seq[0]
        count = e._os_discordant_pairs(x, y)
        self.assertEqual(count, 1)
        self.assertEqual(e._os_discordant_pairs(y, x), count)
        y.ms_map[('J0', 0)] = 'M4'
        self.assertEqual(e.structural_distance(x, y), .5*(1/e._distance_pair_count+1/6))
        y.os_seq = x.os_seq[:]
        self.assertEqual(e.structural_distance(x, y), .5/6)

    def test_no_job_ordering_requirement_added(self):
        ops = {j: [{'machines': {'M': 1}}] for j in (0, 'A')}
        e = WIPGraphDualPopulation(ops, {}, variant='HLNS_FULL', N=1, N_A=2, FE_max=2)
        x = Individual([(0, 0), ('A', 0)], {(0, 0): 'M', ('A', 0): 'M'})
        y = Individual(x.os_seq[::-1], x.ms_map.copy())
        self.assertEqual(e.structural_distance(x, y), .5)
        self.assertEqual(e.structural_distance(y, x), .5)

    def test_OS_pair_cache_bounded(self):
        ops = {j: [{'machines': {'M': 1}} for _ in range(3)] for j in ('A', 'B', 'C', 'D')}
        e = WIPGraphDualPopulation(ops, {}, variant='HLNS_FULL', N=1, N_A=2, FE_max=2, seed=0)
        x = e.random_individual()
        for _ in range(350):
            e.structural_distance(x, e.random_individual())
        self.assertLessEqual(len(e._os_pair_cache), 256)

    def test_dependency_index_invalidated_on_append_and_replace(self):
        e, p = make(); e.evaluate(p)
        prov = p.provenance
        first = e._incoming_dependencies(prov)
        self.assertIs(e._incoming_dependencies(prov), first)
        edge = prov.active_dependencies[0]
        prov.active_dependencies.append(edge)
        second = e._incoming_dependencies(prov)
        self.assertIsNot(second, first)
        prov.active_dependencies[0] = ActiveDependency(edge.src_event_id, edge.dst_event_id, 'test')
        third = e._incoming_dependencies(prov)
        expected = {}
        for edge in prov.active_dependencies:
            expected.setdefault(edge.dst_event_id, []).append(edge)
        self.assertEqual(third, expected)
        self.assertIsNot(second, third)

    def test_dependency_index_cache_bounded(self):
        e, p = make()
        for _ in range(6):
            ind = p.copy(); e.evaluate(ind)
            e._incoming_dependencies(ind.provenance)
        self.assertLessEqual(len(e._dependency_index_cache), 4)

    def test_repeated_diagnosis_still_updates_counters(self):
        e, p = make(); e.evaluate(p)
        before = e.diagnostics['shortage_diagnosis']['calls']
        first = e.diagnose_shortage(p)
        second = e.diagnose_shortage(p)
        self.assertEqual(first, second)
        self.assertEqual(e.diagnostics['shortage_diagnosis']['calls'], before+2)


class DecoderEquivalenceTests(unittest.TestCase):
    def test_lookup_fast_path_matches_live_lookup_path(self):
        class OriginalLookup(StageBufferWIPScheduler):
            def _select_startable(self, *args, selection_data=None, **kwargs):
                return super()._select_startable(*args, selection_data=None, **kwargs)
        ops, buffers, _, _ = fixture()
        encoder = Encoder(ops, rng=random.Random(17))
        before, after = OriginalLookup(ops, buffers), StageBufferWIPScheduler(ops, buffers)
        for _ in range(12):
            os_seq = encoder.generate_random_os()
            ms_map = encoder.build_ms_map(encoder.generate_random_ms())
            for ms in (ms_map, None):
                b = before.decode(os_seq, ms, return_provenance=True)
                a = after.decode(os_seq, ms, return_provenance=True)
                self.assertEqual(b, a)
                self.assertEqual(before.analyze(b[1], b[2], b[0]), after.analyze(a[1], a[2], a[0]))

    def test_lookup_is_rebuilt_after_MS_edit(self):
        ops, buffers, os_seq, ms = fixture()
        d = StageBufferWIPScheduler(ops, buffers)
        d.decode(os_seq, ms, True)
        ms[('J0', 0)] = 'M4'
        self.assertEqual(d.decode(os_seq, ms, True),
                         StageBufferWIPScheduler(ops, buffers).decode(os_seq, ms, True))

    def test_analysis_machine_totals_for_unsorted_records(self):
        ops, buffers, os_seq, ms = fixture()
        d = StageBufferWIPScheduler(ops, buffers)
        m, schedule, trace = d.decode(os_seq, ms)
        original = d.analyze(schedule, trace, m)
        schedule.reverse()
        stats = d.analyze(schedule, trace, m)
        self.assertEqual(stats['machines'], original['machines'])
        self.assertEqual(stats['shortage'], original['shortage'])
        for machine in d.machines:
            records = [r for r in schedule if r['machine'] == machine]
            busy = sum(int(r['end'])-int(r['start']) for r in records)
            span = max(0, max(int(r['release']) for r in records)-min(int(r['start']) for r in records)) if records else 0
            self.assertEqual(stats['machines']['per_machine_busy_time'][machine], busy)
            self.assertEqual(stats['machines']['per_machine_span'][machine], span)

    def test_invalid_genotypes_still_fail_before_scheduling(self):
        ops, buffers, os_seq, ms = fixture()
        d = StageBufferWIPScheduler(ops, buffers)
        for bad in (os_seq[:-1], os_seq[::-1], [os_seq[0]]*len(os_seq)):
            with self.assertRaises(ValueError):
                d.decode(bad, ms, True)
        with self.assertRaises(ValueError):
            d.decode(os_seq, {**ms, ('J0', 0): 'not_a_machine'}, True)


if __name__ == '__main__':
    unittest.main()
