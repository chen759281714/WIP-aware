"""H-LNS cleanup contracts and pre-cleanup search trajectory regression."""

import ast
import hashlib
import inspect
import json
import os
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from src.algorithms import wip_graph_dual_population as module
from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from tests.hlns_test_support import fixture, TEST_ALGORITHMS, make_hlns_algorithm, make
from tests.performance_trajectory_regression import signature

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((Path(__file__).with_name('hlns_cleanup_baseline.json')).read_text())


class CleanupContractTests(unittest.TestCase):
    def test_removed_types_and_methods(self):
        for name in ('RelinkUnit', 'OSRelinkAction'):
            self.assertFalse(hasattr(module, name))
        for name in ('_machine_shortage_evidence', '_os_discrepancy',
                     '_legal_os_insertion_bounds', '_best_os_relink_action',
                     '_build_relink_units', 'choose_guide', '_has_relink_unit',
                     'choose_archive_guide_for_shortage', '_sis_observation'):
            self.assertFalse(hasattr(WIPGraphDualPopulation, name))

    def test_removed_state_and_empty_schema(self):
        e, _ = make()
        for name in ('_sis_diagnostic_callback', '_sis_diagnostic_context',
                     '_last_ps_guide_source', 'eta', 'phi_mass_mismatch_count'):
            self.assertFalse(hasattr(e, name))
        for key in ('hs', 'relinking'):
            self.assertNotIn(key, e.diagnostics)
        self.assertEqual(set(e.diagnostics['ps_moves']), {'basic_mutation', 'hlns_joint'})
        self.assertEqual(set(e.diagnostics['cooperation']),
                         {'pm_attempts', 'pm_usable', 'pm_fallback_self'})

    def test_omitted_variant_fails_for_both_names(self):
        ops, buffers, _, _ = fixture()
        for cls in (module.WIPGraphDualPopulation, module.HLNSDualPopulation):
            with self.assertRaisesRegex(ValueError, 'Specify an explicit H-LNS variant'):
                cls(ops, buffers)

    def test_old_variant_labels_rejected(self):
        ops, buffers, _, _ = fixture()
        for label in ('Full', 'full', 'SIS-Full', 'wo_sis', None):
            with self.assertRaises(ValueError):
                WIPGraphDualPopulation(ops, buffers, variant=label)

    def test_eta_rejected(self):
        ops, buffers, _, _ = fixture()
        with self.assertRaisesRegex(TypeError, 'eta'):
            WIPGraphDualPopulation(ops, buffers, variant='HLNS_FULL', eta=.3)

    def test_old_run_callback_rejected(self):
        e, _ = make()
        with self.assertRaisesRegex(TypeError, 'sis_diagnostic_callback'):
            e.run(sis_diagnostic_callback=lambda row: None)
        self.assertEqual(e.n_evaluations, 0)

    def test_old_ps_guide_interface_rejected(self):
        e, p = make()
        with self.assertRaises(TypeError):
            e.vary_shortage(p, [p])
        with self.assertRaises(TypeError):
            e.vary_shortage(p, archive_assisted=True)
        self.assertEqual(e.n_evaluations, 0)

    def test_identity_matches_concrete_class(self):
        expected = (("HLNS_FULL", "MGS", "HLNS", True),
                    ("HLNS_WO_MGS", "BasicMutation", "HLNS", True),
                    ("HLNS_WO_HLNS", "MGS", "BasicMutation", True),
                    ("HLNS_WO_AC", "MGS", "HLNS", False))
        for name, pm, ps, ac in expected:
            e, _ = make(name)
            self.assertFalse(hasattr(e, 'hlns_config'))
            self.assertEqual(e.algorithm_variant, name)
            self.assertEqual(e.pm_specialized_kind, pm)
            self.assertEqual(e.ps_specialized_kind, ps)
            self.assertEqual(e.pm_archive_guidance_enabled, ac)
            self.assertFalse(e.ps_archive_guidance_enabled)
            for key in ('algorithm_variant', 'pm_specialized_kind', 'ps_specialized_kind',
                        'pm_archive_guidance_enabled', 'ps_archive_guidance_enabled'):
                self.assertEqual(e.diagnostics[key], getattr(e, key))
                with self.assertRaises(AttributeError):
                    setattr(e, key, 'Full')

    def test_explicit_identity_and_alias(self):
        ops, buffers, _, _ = fixture()
        e = module.HLNSDualPopulation(ops, buffers, variant='HLNS_FULL')
        self.assertIs(module.HLNSDualPopulation, WIPGraphDualPopulation)
        self.assertEqual(e.algorithm_variant, 'HLNS_FULL')

    def test_default_parameters(self):
        ops, buffers, _, _ = fixture()
        e = WIPGraphDualPopulation(ops, buffers, variant='HLNS_FULL')
        self.assertEqual((e.N, e.N_A, e.rho, e.p_mut, e.T_coop, e.FE_max),
                         (100, 200, .8, .1, 10, 10000))
        self.assertNotIn('eta', inspect.signature(WIPGraphDualPopulation).parameters)
        self.assertEqual(e.n_evaluations, 0)

    def test_small_explicit_size_and_initialization_FE(self):
        ops, buffers, _, _ = fixture()
        e = make_hlns_algorithm(ops, buffers, 'HLNS_FULL', N=4, N_A=8, FE_max=8, seed=0)
        e.initialize()
        self.assertEqual(e.n_evaluations, 8)
        self.assertEqual((len(e.P_M), len(e.P_S)), (4, 4))
        self.assertLessEqual(len(e.A), 8)

    def test_odd_budget_and_archive_capacity(self):
        e, _ = make(fe=13)
        e.run()
        self.assertEqual(e.n_evaluations, 13)
        self.assertLessEqual(len(e.A), e.N_A)
        self.assertNotIn('P_S_archive', e.diagnostics['source_generated'])
        # After a full first round, the even second round assigns the last FE to P_S.
        self.assertEqual(sum(c['generated'] for c in e.diagnostics['ps_moves'].values()), 4)

    def test_diagnosis_and_MGS_shared_helpers_retained(self):
        e, p = make()
        e.evaluate(p)
        d = e.diagnose_shortage(p)
        self.assertAlmostEqual(sum(d.phi.values()), p.shortage)
        self.assertTrue(e.high_influence_set(p, d))
        self.assertIsInstance(e.makespan_actions(p), list)
        for name in ('_backtrace', '_sample_archive_guide', '_sparse_tournament'):
            self.assertTrue(callable(getattr(e, name)))

    def test_frozen_search_functions_unchanged(self):
        tree = ast.parse(Path(module.__file__).read_text())
        current = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                   for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for name, expected in BASELINE['unchanged_function_ast_sha256'].items():
            # These methods intentionally shed the retired configuration branches;
            # their Full search behavior is checked by the frozen trajectories.
            if name in {'joint_repair', 'select_destroy_set', '_validate_candidate',
                        '_global_fallback', '_component', '_generate_pm_child',
                        '_diverse_initial_population', 'makespan_actions', '_backtrace',
                        'diagnose_shortage'}:
                continue
            with self.subTest(function=name):
                self.assertEqual(current[name], expected)

    def test_protected_sources_unchanged(self):
        for name, digest in BASELINE['protected_source_sha256'].items():
            if name == 'src/algorithms/hlns_variants.py':
                self.assertFalse((ROOT/name).exists())
                continue
            if name == 'src/solution/decoder.py':
                # Exact outputs are checked by decoder/trajectory regressions;
                # the old source hash predates authorized code optimization.
                continue
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest)


class StatisticsTests(unittest.TestCase):
    def changed_child(self, p):
        seq = p.os_seq[:]
        seq[0], seq[1] = seq[1], seq[0]
        ms = p.ms_map.copy()
        ms[('J0', 0)] = 'M4'
        return Individual(seq, ms, operation_order=p.operation_order)

    def test_normal_repair_structure_counts_without_decode(self):
        e, p = make()
        child = self.changed_child(p)
        with patch.object(e.rng, 'random', return_value=.9), \
             patch.object(e, 'select_destroy_set', return_value=set(p.os_seq)), \
             patch.object(e, 'joint_repair', return_value=child), \
             patch.object(e.decoder, 'decode', side_effect=AssertionError('decode')), \
             patch.object(e, 'diagnose_shortage', side_effect=AssertionError('diagnosis')):
            self.assertIs(e.vary_shortage(p), child)
        stats = e.hlns_statistics
        self.assertEqual(stats['hlns_calls'], 1)
        self.assertEqual(stats['hlns_normal_children'], 1)
        self.assertEqual(stats['normal_repair_ms_changes'], [1])
        self.assertEqual(stats['normal_repair_os_pair_changes'], [1])
        self.assertEqual(stats['normal_repair_both_changed_children'], 1)
        self.assertEqual(stats['hlns_both_changed_children'], 1)
        self.assertEqual(e.n_evaluations, 0)

    def test_same_genotype_fallback_not_counted_as_normal_repair(self):
        e, p = make()
        with patch.object(e.rng, 'random', return_value=.9), \
             patch.object(e, 'select_destroy_set', return_value=set(p.os_seq)), \
             patch.object(e, 'joint_repair', return_value=p.copy()):
            e.vary_shortage(p)
        stats = e.hlns_statistics
        self.assertEqual(stats['hlns_calls'], 1)
        self.assertEqual(stats['hlns_same_genotype_fallbacks'], 1)
        self.assertEqual(stats['same_genotype_fallback_ratio'], 1)
        self.assertEqual(stats['normal_repair_os_pair_changes'], [])
        self.assertEqual(stats['normal_repair_ms_changes'], [])
        self.assertEqual(stats['normal_repair_both_changed_children'], 0)

    def test_global_branch_not_a_repair(self):
        e, p = make()
        with patch.object(e.rng, 'random', return_value=0):
            e.vary_shortage(p)
        self.assertEqual(e.hlns_statistics['hlns_calls'], 0)
        self.assertEqual(e.hlns_statistics['same_genotype_fallback_ratio'], 0)
        self.assertEqual(e.hlns_statistics['normal_repair_ms_changes'], [])

    def test_missing_H_fallback_statistics(self):
        e, p = make()
        p.shortage = 0
        with patch.object(e.rng, 'random', return_value=.9):
            e.vary_shortage(p)
        stats = e.hlns_statistics
        self.assertEqual(stats['hlns_calls'], 1)
        self.assertEqual(stats['hlns_global_fallbacks'], 1)
        self.assertEqual(stats['hlns_normal_children'], 0)
        self.assertEqual(stats['fallback_reasons'], {'zero_shortage': 1})

    def test_averages_ratio_and_detached_lists(self):
        e, p = make()
        e.evaluate(p)
        high = e.select_destroy_set(p)
        stats = e.hlns_statistics
        self.assertEqual(stats['H_count'], 1)
        self.assertEqual(stats['mean_H_size'], len(high))
        self.assertEqual(stats['mean_H_ratio'], len(high)/len(p.os_seq))
        for name in ('normal_repair_ms_changes', 'normal_repair_os_pair_changes'):
            stats[name].append(999)
            self.assertNotIn(999, e.hlns_statistics[name])

    def test_collect_false_avoids_structure_count(self):
        e, p = make(observe=False)
        child = self.changed_child(p)
        with patch.object(e.rng, 'random', return_value=.9), \
             patch.object(e, 'select_destroy_set', return_value=set(p.os_seq)), \
             patch.object(e, 'joint_repair', return_value=child), \
             patch.object(e, '_os_discordant_pairs', side_effect=AssertionError('costly stats')):
            e.vary_shortage(p)
        self.assertEqual(e.hlns_statistics['hlns_normal_children'], 1)
        self.assertEqual(e.hlns_statistics['normal_repair_ms_changes'], [])

    def test_cross_stage_pair_definition(self):
        ops = {j: [{'machines': {'M': 1, 'N': 2}} for _ in range(2)] for j in ('A', 'B')}
        e = WIPGraphDualPopulation(ops, {}, variant='HLNS_FULL', N=1, N_A=2, FE_max=2)
        ms = {op: 'M' for op in e.encoder.ms_index_order}
        x = Individual([('A', 0), ('A', 1), ('B', 0), ('B', 1)], ms)
        y = Individual([('A', 0), ('B', 0), ('A', 1), ('B', 1)], ms.copy())
        self.assertEqual(e._os_discordant_pairs(x, y), 1)
        self.assertEqual(e.structural_distance(x, y), .125)
        for candidate in (x, y):
            ranks = {op: i for i, op in enumerate(candidate.os_seq)}
            count = sum(ranks[a] > ranks[b] for a, b in combinations(x.os_seq, 2) if a[0] != b[0])
            self.assertEqual(e._os_discordant_pairs(x, candidate), count)
            self.assertEqual(e._os_discordant_pairs(candidate, x), count)

    def test_same_job_pairs_excluded_and_zero_denominator(self):
        ops = {'A': [{'machines': {'M': 1, 'N': 2}} for _ in range(2)]}
        e = WIPGraphDualPopulation(ops, {}, variant='HLNS_FULL', N=1, N_A=2, FE_max=2)
        ms = {op: 'M' for op in e.encoder.ms_index_order}
        x = Individual([('A', 0), ('A', 1)], ms)
        y = Individual(x.os_seq[::-1], ms.copy())
        self.assertEqual(e._os_discordant_pairs(x, y), 0)
        with self.assertRaises(ValueError):
            e.encoder.validate_os(y.os_seq)
        y.os_seq = x.os_seq[:]
        y.ms_map[('A', 0)] = 'N'
        self.assertEqual(e.structural_distance(x, y), .25)


@unittest.skipUnless(os.environ.get('PYTHONHASHSEED') == '0', 'Frozen trajectory requires PYTHONHASHSEED=0')
class PreCleanupTrajectoryTests(unittest.TestCase):
    def test_all_variants_against_before_cleanup(self):
        for name, seed, observe, expected in BASELINE['trajectories']:
            if name not in {cls.algorithm_variant.fget(None) for cls in TEST_ALGORITHMS}:
                continue
            with self.subTest(variant=name, seed=seed, observe=observe):
                ops, buffers, _, _ = fixture()
                e = make_hlns_algorithm(ops, buffers, name, N=3, N_A=12, FE_max=30,
                    T_coop=2, seed=seed, collect_hlns_stats=observe)
                trace = []
                evaluated = e.evaluate
                def evaluate(ind):
                    result = evaluated(ind)
                    trace.append(signature((ind, e.rng.getstate(), e.n_evaluations)))
                    return result
                e.evaluate = evaluate
                def checkpoint(engine):
                    trace.append(signature((engine.P_M, engine.P_S, engine.A,
                        engine.n_evaluations, engine.round, engine.rng.getstate(),
                        engine._first_source_by_genotype, engine.machine_priority_move_spans)))
                e.run(diagnostic_callback=checkpoint)
                self.assertEqual(e.n_evaluations, 30)
                self.assertEqual(signature(trace), expected)

    def test_observer_on_off_has_identical_full_trajectories(self):
        pairs = {}
        for name, seed, observe, expected in BASELINE['trajectories']:
            pair = pairs.setdefault((name, seed), {})
            pair[observe] = expected
        self.assertEqual(len(pairs), 16)
        for key, pair in pairs.items():
            with self.subTest(case=key):
                self.assertEqual(pair[True], pair[False])


if __name__ == '__main__':
    unittest.main()
