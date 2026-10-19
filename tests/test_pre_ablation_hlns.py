"""Temporary-output runner contracts, analytical checks and 8-FE smoke runs."""

import ast
import copy
from contextlib import ExitStack
from concurrent.futures import Future
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments import run_pre_ablation as r
from tests.hlns_test_support import fixture

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads(Path(__file__).with_name('pre_ablation_hlns_baseline.json').read_text())
INSTANCE = 'WIPHFSP_S1_P1_B1'
DEFAULTS = (tuple(r.SELECTED_INSTANCES), tuple(r.SELECTED_VARIANTS), tuple(r.SEEDS),
            r.current_settings(), r.USE_MULTIPROCESSING, r.N_WORKERS)
SMOKE_FE = []


@unittest.skipUnless(os.environ.get('PYTHONHASHSEED') == '0', 'Runner smoke requires PYTHONHASHSEED=0')
class PreAblationHLNSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory(prefix='pre_ablation_hlns_golden_')
        cls.root = Path(cls.storage.name)
        ops, buffers, _, _ = fixture()
        cls.instance_text = json.dumps(dict(operations=ops, buffers=buffers), sort_keys=True)
        (cls.root/'instances').mkdir()
        (cls.root/'instances'/f'{INSTANCE}.json').write_text(cls.instance_text)
        settings = r.Settings(cls.root/'instances', cls.root/'output', 8, 2, 4, .1, .8, 1)
        cls.golden = {}
        for variant in r.VARIANT_REGISTRY:
            task = r.Task(INSTANCE, variant, 0)
            result = r.run_single_task(task, settings)
            if result['status'] != 'completed':
                raise AssertionError(result)
            cls.golden[variant] = r.load_valid_raw(r.raw_path(task, settings), task, settings)
            SMOKE_FE.append((variant, 8))

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='pre_ablation_hlns_test_')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        (self.path/'instances').mkdir()
        (self.path/'instances'/f'{INSTANCE}.json').write_text(self.instance_text)
        self.settings = r.Settings(self.path/'instances', self.path/'output', 8, 2, 4, .1, .8, 1)
        self.task = r.Task(INSTANCE, 'full', 0)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(r, MODE='run', SELECTED_INSTANCES=[INSTANCE],
            SELECTED_VARIANTS=list(r.VARIANT_REGISTRY), SEEDS=[0], USE_MULTIPROCESSING=False,
            N_WORKERS=2, OVERWRITE_EXISTING=False))
        self.stack.enter_context(patch.object(r, 'current_settings', return_value=self.settings))

    def install(self, variants=None):
        for variant in (list(r.VARIANT_REGISTRY) if variants is None else variants):
            task = r.Task(INSTANCE, variant, 0)
            r.write_json_atomic(r.raw_path(task, self.settings), copy.deepcopy(self.golden[variant]))

    def invalid(self, field, value):
        record = copy.deepcopy(self.golden['full'])
        record[field] = value
        with self.assertRaises((ValueError, KeyError, TypeError)):
            r.validate_raw_record(record, self.task, self.settings)

    def main_mode(self, mode, complete=True):
        events = []
        with patch.object(r, 'MODE', mode), patch.object(sys, 'argv', [str(r.__file__)]), \
             patch.object(r, 'ensure_fixed_hash_seed', side_effect=lambda: events.append('hash')), \
             patch.object(r, 'run_experiments', side_effect=lambda s: events.append('run') or {'complete': complete}), \
             patch.object(r, 'run_analysis', side_effect=lambda s: events.append('analyze')):
            if complete or mode == 'analyze':
                r.main()
            else:
                with self.assertRaises(RuntimeError):
                    r.main()
        return events

    def test_run_mode_does_not_analyze(self):
        self.assertEqual(self.main_mode('run'), ['hash', 'run'])

    def test_analyze_mode_does_not_run(self):
        self.assertEqual(self.main_mode('analyze'), ['analyze'])

    def test_all_mode_order(self):
        self.assertEqual(self.main_mode('all'), ['hash', 'run', 'analyze'])

    def test_all_incomplete_blocks_analysis(self):
        self.assertEqual(self.main_mode('all', complete=False), ['hash', 'run'])

    def test_all_exception_blocks_analysis(self):
        with patch.object(r, 'MODE', 'all'), patch.object(sys, 'argv', [str(r.__file__)]), \
             patch.object(r, 'run_experiments', side_effect=RuntimeError('task failed')), \
             patch.object(r, 'run_analysis') as analyze:
            with self.assertRaises(RuntimeError):
                r.main()
        analyze.assert_not_called()

    def test_invalid_and_old_modes_rejected(self):
        self.assertEqual(r.ALLOWED_MODES, {'run', 'analyze', 'all'})
        for mode in ('unknown', 'sis_diagnostic', 'sis_prefix_replay', 'sis_eta_screen',
                     'sis_local_viability', 'sis_role_direction_test', 'sis_active_lever_diagnostic'):
            with patch.object(r, 'MODE', mode), self.assertRaises(ValueError):
                r.validate_configuration(self.settings)

    def test_unknown_and_old_cli_arguments_rejected(self):
        for arg in ('--unknown', '--sis-diagnostic', '--sis-prefix-replay', '--sis-eta-screen',
                    '--sis-local-viability', '--sis-role-direction-test', '--sis-active-lever-diagnostic',
                    '--sis-selection-aware-audit'):
            with patch.object(sys, 'argv', [str(r.__file__), arg]), \
                 patch.object(r, 'run_experiments') as run, self.assertRaises(ValueError):
                r.main()
            run.assert_not_called()

    def test_actual_cli_rejects_old_flag_without_running(self):
        result = subprocess.run([sys.executable, str(r.__file__), '--sis-diagnostic'],
                                 capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unsupported arguments', result.stderr)

    def test_default_design_and_dynamic_tasks(self):
        instances, variants, seeds, settings, multiprocessing, workers = DEFAULTS
        self.assertEqual((len(instances), len(variants), len(seeds)), (12, 4, 5))
        self.assertEqual(variants, ('full', 'wo_mgs', 'wo_hlns', 'wo_ac'))
        self.assertEqual((settings.fe_max, settings.n, settings.n_a, settings.p_mu, settings.rho, settings.t_coop),
                         (10000, 100, 200, .1, .8, 10))
        self.assertTrue(multiprocessing)
        self.assertEqual(workers, 5)
        with patch.multiple(r, SELECTED_INSTANCES=list(instances), SELECTED_VARIANTS=list(variants), SEEDS=list(seeds)):
            self.assertEqual(len(r.build_tasks()), 240)
        self.assertEqual(len(r.build_tasks()), 4)

    def test_registry_classes_and_formal_identities(self):
        self.assertEqual(set(r.VARIANT_REGISTRY), {'full', 'wo_mgs', 'wo_hlns', 'wo_ac'})
        for variant, record in self.golden.items():
            self.assertEqual(record['variant'], variant)
            self.assertEqual(record['algorithm_variant'], r.EXPECTED_ALGORITHM_IDS[variant])
            r.validate_identity(record['diagnostics'], r.Task(INSTANCE, variant, 0))
        with self.assertRaises(TypeError):
            r.EXPECTED_ALGORITHM_IDS['full'] = 'full'

    def test_constructor_parameters_remove_eta_and_explicit_full_identity(self):
        self.assertNotIn('eta', self.settings.algorithm_parameters(0))
        self.assertNotIn('eta', r.Settings.__dataclass_fields__)
        self.assertEqual(self.golden['full']['constructor_parameters']['variant'], 'HLNS_FULL')
        for variant in ('wo_mgs', 'wo_hlns', 'wo_ac'):
            self.assertNotIn('variant', self.golden[variant]['constructor_parameters'])

    def test_wrong_registered_mechanism_fails_before_optimization(self):
        wrong = r.VARIANT_REGISTRY['wo_ac']
        task = r.Task(INSTANCE, 'wo_mgs', 0)
        with patch.dict(r.VARIANT_REGISTRY, {'wo_mgs': wrong}), \
             patch.object(wrong, 'run', side_effect=AssertionError('wrong algorithm ran')) as run:
            result = r.run_single_task(task, self.settings)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('identity mismatch', result['exception'])
        run.assert_not_called()

    def test_all_four_smoke_records_validate(self):
        self.install()
        records, missing = r.inspect_raw(r.build_tasks(), self.settings)
        self.assertFalse(missing)
        self.assertEqual(len(records), 4)
        for record in records.values():
            self.assertEqual(record['actual_FE'], 8)
            self.assertEqual(record['population_sizes'], {'P_M': 2, 'P_S': 2})

    def test_task_identity_corruption(self):
        for field, value in (('instance', 'wrong'), ('variant', 'wo_hlns'), ('seed', 1)):
            self.invalid(field, value)

    def test_formal_identity_and_mechanism_corruption(self):
        self.invalid('algorithm_variant', 'full')
        record = copy.deepcopy(self.golden['full'])
        record['diagnostics']['ps_specialized_kind'] = 'BasicMutation'
        with self.assertRaises(ValueError):
            r.validate_raw_record(record, self.task, self.settings)

    def test_FE_and_population_corruption(self):
        for field, value in (('FE_max', 7), ('actual_FE', 7), ('actual_FE', True),
                             ('population_sizes', {'P_M': 1, 'P_S': 2})):
            self.invalid(field, value)

    def test_parameter_and_constructor_corruption(self):
        self.invalid('algorithm_parameters', {**self.golden['full']['algorithm_parameters'], 'N': 3})
        self.invalid('constructor_parameters', {**self.golden['full']['constructor_parameters'], 'variant': 'full'})

    def test_archive_empty_nonfinite_negative_and_dominated(self):
        for archive in ([], [{'makespan': math.inf, 'shortage': 1}],
                        [{'makespan': 1, 'shortage': math.nan}], [{'makespan': -1, 'shortage': 1}]):
            self.invalid('final_archive', archive)
        record = copy.deepcopy(self.golden['full'])
        record.update(final_archive=[{'makespan': 1, 'shortage': 1}, {'makespan': 2, 'shortage': 2}],
                      final_archive_size=2, best_makespan=1, best_shortage=1)
        with self.assertRaises(ValueError):
            r.validate_raw_record(record, self.task, self.settings)

    def test_archive_count_capacity_and_extremes(self):
        self.invalid('final_archive_size', 999)
        self.invalid('best_makespan', -1)
        self.invalid('best_shortage', -1)
        record = copy.deepcopy(self.golden['full'])
        record['final_archive'] *= 5
        record['final_archive_size'] = len(record['final_archive'])
        with self.assertRaises(ValueError):
            r.validate_raw_record(record, self.task, self.settings)

    def test_runtime_timestamp_and_move_counts(self):
        for value in (-1, math.inf, True):
            self.invalid('runtime_seconds', value)
        self.invalid('timestamp_utc', '')
        record = copy.deepcopy(self.golden['full'])
        record['diagnostics']['ps_moves']['basic_mutation']['generated'] += 1
        with self.assertRaises(ValueError):
            r.validate_raw_record(record, self.task, self.settings)

    def test_source_fingerprint_checks_all_implementations(self):
        source = r.task_source(self.task, self.settings)
        self.assertEqual(set(source['sources']), set(r.SOURCE_FILES))
        record = copy.deepcopy(self.golden['full'])
        record['source']['sources'][r.SOURCE_FILES[0]] = '0'*64
        record['fingerprint'] = r.source_fingerprint(record['source'])
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            r.validate_raw_record(record, self.task, self.settings)

    def test_instance_version_changes_rejected(self):
        (self.settings.instance_dir/f'{INSTANCE}.json').write_text(self.instance_text+'\n')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            r.validate_raw_record(self.golden['full'], self.task, self.settings)

    def test_loaded_source_version_change_requires_restart(self):
        with patch.object(r, 'IMPORTED_SOURCE_HASHES', {}), self.assertRaisesRegex(ValueError, 'restart'):
            r.task_source(self.task, self.settings)

    def test_fingerprint_ignores_mode_workers_and_output_location(self):
        old = r.task_source(self.task, self.settings)
        with patch.multiple(r, MODE='analyze', N_WORKERS=7):
            new = r.task_source(self.task, replace(self.settings, output_root=self.path/'elsewhere'))
        self.assertEqual(old, new)

    def test_valid_resume_has_zero_new_runs(self):
        self.install()
        with patch.object(r, 'run_single_task', side_effect=AssertionError('rerun')), \
             patch.object(r, 'run_analysis', side_effect=AssertionError('run-only analysis')):
            manifest = r.run_experiments(self.settings)
        self.assertTrue(manifest['complete'])
        self.assertEqual((manifest['completed'], manifest['skipped']), (0, 4))

    def test_corrupt_existing_never_silently_overwritten(self):
        path = r.raw_path(self.task, self.settings)
        r.write_json_atomic(path, {'algorithm_variant': 'old'})
        before = path.read_bytes()
        with patch.object(r, 'run_single_task', side_effect=AssertionError('launched')), self.assertRaises(RuntimeError):
            r.run_experiments(self.settings)
        self.assertEqual(before, path.read_bytes())
        manifest = json.loads((self.settings.output_root/'run_manifest.json').read_text())
        self.assertFalse(manifest['complete'])
        self.assertEqual(manifest['failed'], 1)

    def test_explicit_overwrite_only_replaces_new_output(self):
        self.install()
        path = r.raw_path(self.task, self.settings)
        r.write_json_atomic(path, {'algorithm_variant': 'old'})
        with patch.object(r, 'OVERWRITE_EXISTING', True):
            result = r.run_experiments(self.settings)
        self.assertTrue(result['complete'])
        self.assertEqual(result['completed'], 4)
        for variant in r.VARIANT_REGISTRY:
            SMOKE_FE.append((variant, 8))

    def test_worker_failure_is_in_manifest_and_run_fails(self):
        def fail(task, *_):
            return {'task': task.__dict__, 'status': 'failed', 'exception': 'intentional'}
        with patch.object(r, 'run_single_task', side_effect=fail), self.assertRaises(RuntimeError):
            r.run_experiments(self.settings)
        manifest = json.loads((self.settings.output_root/'run_manifest.json').read_text())
        self.assertEqual((manifest['failed'], manifest['missing']), (4, 4))
        self.assertFalse(manifest['complete'])

    def test_spawn_smoke_and_raw_reproducibility(self):
        with patch.multiple(r, USE_MULTIPROCESSING=True, N_WORKERS=2), \
             patch.object(r, 'get_context', wraps=r.get_context) as context:
            manifest = r.run_experiments(self.settings)
        context.assert_called_with('spawn')
        self.assertTrue(manifest['complete'])
        self.assertEqual(manifest['completed'], 4)
        for variant in r.VARIANT_REGISTRY:
            task = r.Task(INSTANCE, variant, 0)
            record = r.load_valid_raw(r.raw_path(task, self.settings), task, self.settings)
            for key in ('final_archive', 'best_makespan', 'best_shortage', 'diagnostics', 'algorithm_parameters'):
                self.assertEqual(record[key], self.golden[variant][key])
            SMOKE_FE.append((variant, 8))

    def test_executor_start_failure_is_recorded(self):
        with patch.object(r, 'USE_MULTIPROCESSING', True), \
             patch.object(r, 'ProcessPoolExecutor', side_effect=RuntimeError('executor startup')), \
             self.assertRaises(RuntimeError):
            r.run_experiments(self.settings)
        manifest = json.loads((self.settings.output_root/'run_manifest.json').read_text())
        self.assertEqual(manifest['failed'], 4)
        self.assertFalse(manifest['complete'])
        self.assertIn('executor startup', manifest['failed_tasks'][0]['exception'])

    def test_future_exception_is_recorded(self):
        class FailedExecutor:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def submit(self, *args):
                future = Future()
                future.set_exception(RuntimeError('worker future'))
                return future
        with patch.object(r, 'USE_MULTIPROCESSING', True), \
             patch.object(r, 'ProcessPoolExecutor', FailedExecutor), self.assertRaises(RuntimeError):
            r.run_experiments(self.settings)
        manifest = json.loads((self.settings.output_root/'run_manifest.json').read_text())
        self.assertEqual(manifest['failed'], 4)
        self.assertTrue(all('worker future' in row['exception'] for row in manifest['failed_tasks']))

    def test_historical_and_other_repository_outputs_refused(self):
        for root in (ROOT/'experiments/results/pre_ablation', ROOT/'experiments/results/pre_ablation/raw',
                     ROOT/'experiments/results', ROOT/'experiments/results/other'):
            with self.assertRaises(ValueError):
                r.validate_output_root(replace(self.settings, output_root=root))

    def test_nested_symlink_cannot_escape_output(self):
        self.settings.output_root.mkdir()
        (self.settings.output_root/'raw').symlink_to(self.path/'outside', target_is_directory=True)
        with self.assertRaises(ValueError):
            r.raw_path(self.task, self.settings)

    def test_atomic_write_cleans_temporary_file(self):
        path = self.path/'atomic.json'
        r.write_json_atomic(path, {'valid': 1})
        with patch.object(r.os, 'replace', side_effect=OSError('interrupted')), self.assertRaises(OSError):
            r.write_json_atomic(path, {'valid': 2})
        self.assertEqual(json.loads(path.read_text()), {'valid': 1})
        self.assertEqual(list(self.path.glob('.*.tmp')), [])

    def test_hash_seed_guard(self):
        with patch.dict(os.environ, {'PYTHONHASHSEED': '1'}), self.assertRaises(RuntimeError):
            r.run_experiments(self.settings)

    def test_metric_definitions_and_shared_normalization(self):
        r.metric_sanity_checks()
        self.assertAlmostEqual(r.compute_hv_2d([(0, 0)])/1.21, 1)
        records = {}
        for variant, point in zip(r.VARIANT_REGISTRY, ((10, 30), (20, 20), (30, 10), (20, 30))):
            record = copy.deepcopy(self.golden[variant])
            record.update(final_archive=[dict(zip(('makespan', 'shortage'), point))],
                          final_archive_size=1, best_makespan=point[0], best_shortage=point[1])
            records[r.Task(INSTANCE, variant, 0)] = record
        rows = r.build_run_metrics(records)
        full = next(row for row in rows if row['variant'] == 'full')
        self.assertAlmostEqual(full['raw_hv'], 1.1*.1)
        self.assertTrue(all(0 <= row['nhv'] <= 1 for row in rows))

    def test_paired_delta_directions_and_matching_seed(self):
        rows = []
        for variant in r.VARIANT_REGISTRY:
            rows.append(dict(instance=INSTANCE, **r.instance_dimensions(INSTANCE), variant=variant,
                seed=0, nhv=.9 if variant == 'full' else .6, igd=.1 if variant == 'full' else .2,
                best_makespan=10 if variant == 'full' else 11, best_shortage=1 if variant == 'full' else 2))
        deltas = r.build_pairwise_deltas(rows)
        self.assertEqual({row['variant'] for row in deltas}, {'wo_mgs', 'wo_hlns', 'wo_ac'})
        self.assertTrue(all(row['delta_'+metric] > 0 for row in deltas for metric in r.DELTA_METRICS))
        grouped = r.build_group_summary(deltas)
        self.assertTrue(all(row['wins'] == 1 for row in grouped))
        self.assertEqual(len(r.build_wilcoxon(deltas)), 12)

    def test_analyze_never_constructs_or_runs_algorithms(self):
        self.install()
        before = {p: p.read_bytes() for p in (self.settings.output_root/'raw').rglob('*.json')}
        with ExitStack() as stack:
            for cls in r.VARIANT_REGISTRY.values():
                stack.enter_context(patch.object(cls, '__init__', side_effect=AssertionError('construct')))
                stack.enter_context(patch.object(cls, 'run', side_effect=AssertionError('optimize')))
            result = r.run_analysis(self.settings)
        self.assertEqual((result['runs'], result['pairs']), (4, 3))
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        output = self.settings.output_root/'analysis'
        expected = {'run_metrics.csv', 'instance_summary.csv', 'pairwise_deltas.csv',
                    'wtl_summary.csv', 'group_summary.csv', 'wilcoxon.csv', 'mechanism_summary.txt'}
        self.assertEqual({p.name for p in output.iterdir()}, {'pre_ablation_'+name for name in expected})
        text = (output/'pre_ablation_mechanism_summary.txt').read_text()
        self.assertIn('wo_hlns', text)
        self.assertNotIn('wo_sis', text)
        self.assertIn('not independent', text)

    def test_incomplete_and_incompatible_analysis_refused(self):
        self.install(['full', 'wo_mgs', 'wo_ac'])
        with patch.object(r, 'run_single_task', side_effect=AssertionError('optimization')), self.assertRaises(RuntimeError):
            r.run_analysis(self.settings)
        self.assertFalse((self.settings.output_root/'analysis').exists())

    def test_missing_seed_analysis_refused(self):
        self.install()
        with patch.object(r, 'SEEDS', [0, 1]), self.assertRaises(RuntimeError):
            r.run_analysis(self.settings)
        self.assertFalse((self.settings.output_root/'analysis').exists())

    def test_legacy_identity_analysis_refused(self):
        self.install()
        record = copy.deepcopy(self.golden['full'])
        record['algorithm_variant'] = 'full'
        r.write_json_atomic(r.raw_path(self.task, self.settings), record)
        with self.assertRaises(RuntimeError):
            r.run_analysis(self.settings)
        self.assertFalse((self.settings.output_root/'analysis').exists())

    def test_analysis_math_AST_unchanged(self):
        tree = ast.parse(Path(r.__file__).read_text())
        hashes = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                  for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name, expected in BASELINE['analysis_ast_sha256'].items():
            self.assertEqual(hashes[name], expected, name)

    def test_no_retired_code_or_dispatch(self):
        tree = ast.parse(Path(r.__file__).read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertFalse(any(n.startswith(('sis_', 'prefix_', 'eta_', 'run_sis', 'run_eta')) for n in names))
        self.assertNotIn('EtaTask', names)
        self.assertNotIn('SISPrefixCaptureAlgorithm', names)
        source = Path(r.__file__).read_text()
        for token in ('SIS_', '--sis-', 'wo_sis', 'sis_settings', 'eta:'):
            self.assertNotIn(token, source)

    def test_production_source_hashes_unchanged(self):
        for name, expected in BASELINE['production_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)


if __name__ == '__main__':
    unittest.main()
