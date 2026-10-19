"""Functional tests for Full plus three concrete inherited paper ablations."""

import inspect
import json
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from src.algorithms.wip_graph_dual_population_wo_mgs import WIPGraphDualPopulationWithoutMGS
from src.algorithms.wip_graph_dual_population_wo_hlns import WIPGraphDualPopulationWithoutHLNS
from src.algorithms.wip_graph_dual_population_wo_ac import WIPGraphDualPopulationWithoutAC
from tests.hlns_test_support import fixture, TEST_ALGORITHMS
from tests.performance_trajectory_regression import signature

ROOT = Path(__file__).resolve().parents[1]


def create(cls, *, observe=True, fe=18):
    ops, buffers, seq, ms = fixture()
    kwargs = dict(N=3, N_A=12, FE_max=fe, T_coop=1, seed=7, collect_hlns_stats=observe)
    if cls is WIPGraphDualPopulation:
        kwargs['variant'] = 'HLNS_FULL'
    e = cls(ops, buffers, **kwargs)
    p = Individual(seq, ms, operation_order=tuple(e.encoder.ms_index_order))
    return e, p


class PaperArchitectureTests(unittest.TestCase):
    def test_three_separate_direct_subclasses(self):
        for cls in TEST_ALGORITHMS[1:]:
            self.assertEqual(cls.__bases__, (WIPGraphDualPopulation,))
            self.assertTrue(Path(inspect.getfile(cls)).exists())
        self.assertEqual(len({inspect.getfile(cls) for cls in TEST_ALGORITHMS}), 4)

    def test_shared_framework_not_copied(self):
        shared = ('run', 'initialize', 'update_archive', 'tournament', 'structural_distance',
                  'similarity_replace', 'evaluate', 'diagnose_shortage', 'high_influence_set',
                  'basic_mutation', '_generate_pm_child', 'makespan_actions', 'joint_repair')
        for cls in TEST_ALGORITHMS[1:]:
            for name in shared:
                self.assertNotIn(name, cls.__dict__)
                self.assertIs(getattr(cls, name), getattr(WIPGraphDualPopulation, name))

    def test_old_modules_deleted_not_aliased(self):
        for name in ('wip_graph_dual_population_wo_sis.py', 'hlns_variants.py'):
            self.assertFalse((ROOT/'src/algorithms'/name).exists())
        for cls in TEST_ALGORITHMS:
            self.assertNotIn('SIS', cls.__name__)

    def test_no_runtime_configuration_matrix(self):
        source = Path(inspect.getfile(WIPGraphDualPopulation)).read_text()
        for token in ('hlns_config', 'get_hlns_config', 'HLNS_WO_', '_SPECS'):
            self.assertNotIn(token, source)
        for cls in TEST_ALGORITHMS:
            e, _ = create(cls)
            self.assertFalse(hasattr(e, 'hlns_config'))

    def test_full_requires_explicit_identity(self):
        ops, buffers, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'explicit H-LNS variant'):
            WIPGraphDualPopulation(ops, buffers)
        for label in ('Full', 'full', 'SIS-Full', 'HLNS_WO_DIAGNOSIS', 'HLNS_WO_PARENT_BIAS',
                      'HLNS_WO_OS_REPAIR', 'HLNS_WO_MS_REPAIR', 'HLNS_WO_MGS'):
            with self.assertRaises(ValueError):
                WIPGraphDualPopulation(ops, buffers, variant=label)

    def test_subclass_automatic_identity_and_matching_override(self):
        ops, buffers, _, _ = fixture()
        for cls in TEST_ALGORITHMS[1:]:
            e = cls(ops, buffers)
            same = cls(ops, buffers, variant=e.algorithm_variant)
            self.assertEqual(e.algorithm_variant, same.algorithm_variant)
            for wrong in ('HLNS_FULL', 'Full', None, 'HLNS_WO_DIAGNOSIS'):
                with self.assertRaises(ValueError):
                    cls(ops, buffers, variant=wrong)

    def test_four_versions_same_defaults(self):
        ops, buffers, _, _ = fixture()
        for cls in TEST_ALGORITHMS:
            e = cls(ops, buffers, variant=cls.algorithm_variant.fget(None))
            self.assertEqual((e.N, e.N_A, e.rho, e.p_mut, e.T_coop, e.FE_max),
                             (100, 200, .8, .1, 10, 10000))

    def test_metadata_matches_mechanism_matrix(self):
        expected = (('HLNS_FULL', 'MGS', 'HLNS', True),
                    ('HLNS_WO_MGS', 'BasicMutation', 'HLNS', True),
                    ('HLNS_WO_HLNS', 'MGS', 'BasicMutation', True),
                    ('HLNS_WO_AC', 'MGS', 'HLNS', False))
        for cls, values in zip(TEST_ALGORITHMS, expected):
            e, _ = create(cls)
            self.assertEqual((e.algorithm_variant, e.pm_specialized_kind,
                              e.ps_specialized_kind, e.pm_archive_guidance_enabled), values)
            self.assertFalse(e.ps_archive_guidance_enabled)
            for name in ('algorithm_variant', 'pm_specialized_kind', 'ps_specialized_kind',
                         'pm_archive_guidance_enabled', 'ps_archive_guidance_enabled'):
                self.assertEqual(e.diagnostics[name], getattr(e, name))


class WithoutMGSTests(unittest.TestCase):
    def test_specialized_branch_keeps_archive_seed(self):
        e, p = create(WIPGraphDualPopulationWithoutMGS)
        e.P_M = [p]
        with patch.object(e.rng, 'random', return_value=.9), \
             patch.object(e, 'archive_guide', return_value=p) as guide, \
             patch.object(e, 'basic_mutation', wraps=e.basic_mutation) as basic, \
             patch.object(e, 'tournament', side_effect=AssertionError('lost archive seed')), \
             patch.object(e, '_mgs_child', side_effect=AssertionError('MGS')), \
             patch.object(e, 'makespan_actions', side_effect=AssertionError('MGS actions')):
            seed, child, source = e._generate_pm_child(True)
        self.assertIs(seed, p)
        self.assertIs(basic.call_args.args[0], p)
        self.assertEqual(guide.call_count, 1)
        self.assertEqual(source, 'P_M_archive')
        self.assertEqual(e.diagnostics['cooperation']['pm_attempts'], 1)
        self.assertEqual(e.diagnostics['cooperation']['pm_usable'], 1)
        self.assertEqual(e._last_pm_move_kind, 'basic_mutation')
        self.assertIsNone(child.shortage)

    def test_global_branch_uses_self_parent(self):
        e, p = create(WIPGraphDualPopulationWithoutMGS)
        e.P_M = [p]
        with patch.object(e.rng, 'random', return_value=0), \
             patch.object(e, 'archive_guide', side_effect=AssertionError('global branch AC')):
            seed, _, source = e._generate_pm_child(True)
        self.assertIs(seed, p)
        self.assertEqual(source, 'P_M_self')
        self.assertEqual(e.diagnostics['cooperation']['pm_attempts'], 0)

    def test_empty_archive_falls_back_to_self(self):
        e, p = create(WIPGraphDualPopulationWithoutMGS)
        e.P_M = [p]
        with patch.object(e.rng, 'random', return_value=.9):
            seed, _, source = e._generate_pm_child(True)
        self.assertIs(seed, p)
        self.assertEqual(source, 'P_M_self')
        self.assertEqual(e.diagnostics['cooperation']['pm_fallback_self'], 1)

    def test_run_never_generates_MGS_but_keeps_HLNS(self):
        e, _ = create(WIPGraphDualPopulationWithoutMGS)
        with patch.object(e, 'makespan_actions', side_effect=AssertionError('MGS')):
            e.run()
        self.assertEqual(e.n_evaluations, 18)
        self.assertEqual(e.diagnostics['pm_moves']['basic_mutation']['generated'], 6)
        for kind in ('machine_priority', 'machine_reassign'):
            self.assertEqual(e.diagnostics['pm_moves'][kind]['generated'], 0)
        self.assertGreater(e.hlns_statistics['hlns_calls'], 0)
        self.assertTrue(e.A)


class WithoutHLNSTests(unittest.TestCase):
    def test_branch_draw_and_basic_child_exactly_match_reference(self):
        for value in (0, .9):
            e, p = create(WIPGraphDualPopulationWithoutHLNS)
            expected = random.Random()
            expected.setstate(e.rng.getstate())
            ref, _ = create(WIPGraphDualPopulationWithoutHLNS)
            ref.rng = expected
            # Both values take exactly one formal branch draw before BasicMutation.
            with patch.object(e.rng, 'random', return_value=value) as draw, \
                 patch.object(expected, 'random', return_value=value) as reference_draw, \
                 patch.object(e, 'diagnose_shortage', side_effect=AssertionError('diagnosis')), \
                 patch.object(e, 'high_influence_set', side_effect=AssertionError('H')), \
                 patch.object(e, 'select_destroy_set', side_effect=AssertionError('D')), \
                 patch.object(e, 'joint_repair', side_effect=AssertionError('repair')):
                expected.random()
                reference = ref.basic_mutation(p)
                child = e.vary_shortage(p)
                self.assertEqual(draw.call_count, 1)
                self.assertEqual(reference_draw.call_count, 1)
            self.assertEqual(e.genotype_key(child), ref.genotype_key(reference))
            self.assertEqual(e.rng.getstate(), expected.getstate())
            self.assertEqual(e.hlns_statistics['hlns_calls'], 0)

    def test_actual_branch_rng_advance(self):
        e, p = create(WIPGraphDualPopulationWithoutHLNS)
        ref, _ = create(WIPGraphDualPopulationWithoutHLNS)
        ref.rng.random()
        expected = ref.basic_mutation(p)
        child = e.vary_shortage(p)
        self.assertEqual(e.genotype_key(child), ref.genotype_key(expected))
        self.assertEqual(e.rng.getstate(), ref.rng.getstate())

    def test_PM_MGS_and_archive_pipeline_retained(self):
        e, p = create(WIPGraphDualPopulationWithoutHLNS)
        with patch.object(e.rng, 'random', return_value=.9), \
             patch.object(e, 'archive_guide', return_value=p) as guide, \
             patch.object(e, '_mgs_child', return_value=p.copy()) as mgs:
            seed, _, source = e._generate_pm_child(True)
        self.assertIs(seed, p)
        self.assertEqual(source, 'P_M_archive')
        self.assertEqual(guide.call_count, 1)
        self.assertIs(mgs.call_args.args[0], p)

    def test_run_no_HLNS_or_diagnosis(self):
        e, _ = create(WIPGraphDualPopulationWithoutHLNS)
        with patch.object(e, 'diagnose_shortage', side_effect=AssertionError('diagnosis')), \
             patch.object(e, 'joint_repair', side_effect=AssertionError('repair')):
            e.run()
        self.assertEqual(e.diagnostics['ps_moves']['basic_mutation']['generated'], 6)
        self.assertEqual(e.diagnostics['ps_moves']['hlns_joint']['generated'], 0)
        self.assertEqual(e.hlns_statistics['hlns_calls'], 0)
        self.assertEqual(e.diagnostics['shortage_diagnosis']['calls'], 0)
        self.assertTrue(e.A)


class WithoutACTests(unittest.TestCase):
    def test_no_archive_sampling_self_seed_with_MGS(self):
        e, p = create(WIPGraphDualPopulationWithoutAC)
        e.P_M = [p]
        e.A = [p.copy()]
        with patch.object(e.rng, 'random', return_value=.9), \
             patch.object(e, 'archive_guide', side_effect=AssertionError('AC')), \
             patch.object(e, '_mgs_child', return_value=p.copy()) as mgs:
            seed, _, source = e._generate_pm_child(True)
        self.assertIs(seed, p)
        self.assertEqual(source, 'P_M_self')
        self.assertIs(mgs.call_args.args[0], p)
        self.assertEqual(e.diagnostics['cooperation']['pm_attempts'], 0)

    def test_archive_direct_guide_disabled(self):
        e, _ = create(WIPGraphDualPopulationWithoutAC)
        with patch.object(e, '_sample_archive_guide', side_effect=AssertionError('sample')):
            self.assertIsNone(e.archive_guide())

    def test_run_archive_update_and_HLNS_retained(self):
        e, _ = create(WIPGraphDualPopulationWithoutAC)
        with patch.object(e, 'update_archive', wraps=e.update_archive) as archive:
            e.run()
        self.assertEqual(archive.call_count, 3)
        self.assertTrue(e.A)
        self.assertGreater(e.hlns_statistics['hlns_calls'], 0)
        self.assertEqual(e.diagnostics['cooperation']['pm_attempts'], 0)
        self.assertNotIn('P_M_archive', e.diagnostics['source_generated'])


class SharedCorrectnessTests(unittest.TestCase):
    def test_FE_decode_and_every_child_archive_eligibility(self):
        for cls in TEST_ALGORITHMS:
            e, _ = create(cls, fe=13)
            with patch.object(e.decoder, 'decode', wraps=e.decoder.decode) as decoded, \
                 patch.object(e, 'update_archive', wraps=e.update_archive) as archive:
                e.run()
            self.assertEqual(decoded.call_count, 13)
            self.assertEqual(e.n_evaluations, 13)
            batches = [call.args[0] for call in archive.call_args_list]
            self.assertEqual([len(batch) for batch in batches], [6, 6, 1])
            self.assertTrue(all(ind.shortage is not None for batch in batches for ind in batch))
            self.assertEqual((len(e.P_M), len(e.P_S)), (3, 3))
            self.assertLessEqual(len(e.A), 12)
            self.assertEqual(sum(c['generated'] for group in ('pm_moves', 'ps_moves')
                                 for c in e.diagnostics[group].values()), 7)

    def test_nearest_strict_replacement_no_parent_gate(self):
        for cls in TEST_ALGORITHMS:
            e, p = create(cls)
            p.makespan, p.shortage = 1, 0
            child = p.copy()
            child.makespan, child.shortage = 2, 2
            for objective in ('makespan', 'shortage'):
                nearest = child.copy()
                setattr(nearest, objective, 3)
                population = [nearest]
                self.assertTrue(e.similarity_replace(population, child, objective))
                self.assertIs(population[0], child)
                self.assertFalse(e.similarity_replace(population, child.copy(), objective))

    def test_rejected_but_nondominated_child_can_enter_archive(self):
        for cls in TEST_ALGORITHMS:
            e, p = create(cls)
            p.makespan, p.shortage = 10, 5
            child = p.copy()
            child.ms_map = child.ms_map.copy()
            child.ms_map[('J0', 0)] = 'M4'
            child.makespan, child.shortage = 5, 6
            nearest = child.copy()
            nearest.shortage = 4
            e.A = [p]
            self.assertFalse(e.similarity_replace([nearest], child, 'shortage'))
            e.update_archive([child])
            self.assertIn(child, e.A)

    def test_identical_initialization(self):
        states = []
        for cls in TEST_ALGORITHMS:
            e, _ = create(cls, fe=6)
            e.initialize()
            states.append(signature((e.P_M, e.P_S, e.A, e.rng.getstate(), e.n_evaluations)))
        self.assertEqual(len(set(states)), 1)

    def test_each_version_reproducible_and_statistics_neutral(self):
        for cls in TEST_ALGORITHMS:
            results = []
            for observe in (True, True, False):
                e, _ = create(cls, observe=observe)
                e.run()
                results.append(signature((e.P_M, e.P_S, e.A, e.rng.getstate(), e.n_evaluations)))
            self.assertEqual(len(set(results)), 1)

    @unittest.skipUnless(os.environ.get('PYTHONHASHSEED') == '0', 'Frozen Full requires PYTHONHASHSEED=0')
    def test_Full_exact_pre_refactor_trajectory_including_diagnostics(self):
        records = json.loads(Path(__file__).with_name('hlns_inheritance_baseline.json').read_text())
        for observe, expected in records:
            e, _ = create(WIPGraphDualPopulation, observe=observe)
            trace = []
            evaluated = e.evaluate
            def evaluate(ind):
                result = evaluated(ind)
                trace.append(signature((ind, e.rng.getstate(), e.n_evaluations)))
                return result
            e.evaluate = evaluate
            def checkpoint(engine):
                trace.append(signature((engine.P_M, engine.P_S, engine.A, engine.diagnostics,
                    engine.n_evaluations, engine.round, engine.rng.getstate(),
                    engine._first_source_by_genotype, engine.machine_priority_move_spans)))
            e.run(diagnostic_callback=checkpoint)
            self.assertEqual(signature(trace), expected)


if __name__ == '__main__':
    unittest.main()
