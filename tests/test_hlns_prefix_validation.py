"""Numeric prefix consistency only; never construct/run an optimizer or decode."""
import math
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import src.algorithms.wip_graph_dual_population as module


def fixture(values, shortage, rho):
    engine = object.__new__(module.WIPGraphDualPopulation)
    ops = [(f'J{i:02d}', 0) for i in range(len(values))]
    engine.rho = rho
    engine.encoder = SimpleNamespace(ms_index_order=ops)
    engine.diagnostics = {'hlns': {'hlns_diagnosis_errors': 0,
                                 'diagnosis_error_reasons': Counter()}}
    diagnosis = module.ShortageDiagnosis([('B', 0, 1, shortage, set())],
                                         dict(zip(ops, values)), set(), [])
    engine.diagnose_shortage = lambda parent: diagnosis
    parent = SimpleNamespace(shortage=shortage, stats={}, provenance=SimpleNamespace(events={}))
    return engine, parent, diagnosis


class PrefixValidationTests(unittest.TestCase):
    def test_higher_accuracy_sum_cannot_disagree_with_incremental_H(self):
        engine, parent, diagnosis = fixture([.05]*20, 1., .5)
        expected = engine.high_influence_set(parent, diagnosis)
        self.assertEqual(len(expected), 11)
        # Reproduce the relevant higher-accuracy sum behavior without requiring 3.12.
        with patch.object(module, 'sum', math.fsum, create=True):
            actual, reason = engine._diagnostic_high(parent)
        self.assertEqual(actual, expected)
        self.assertIsNone(reason)
        self.assertEqual(engine.diagnostics['hlns']['hlns_diagnosis_errors'], 0)

    def test_all_tuning_rho_levels_preserve_existing_H(self):
        for rho in (.5, .65, .8, .95):
            engine, parent, diagnosis = fixture([.05]*20, 1., rho)
            expected = engine.high_influence_set(parent, diagnosis)
            for summation in (sum, math.fsum):
                with patch.object(module, 'sum', summation, create=True):
                    actual, reason = engine._diagnostic_high(parent)
                self.assertEqual(actual, expected)
                self.assertIsNone(reason)

    def test_exact_boundary(self):
        engine, parent, diagnosis = fixture([1., 1., 1., 1.], 4., .5)
        actual, reason = engine._diagnostic_high(parent)
        self.assertEqual(actual, {('J00', 0), ('J01', 0)})
        self.assertIsNone(reason)

    def test_too_short_prefix_still_rejected(self):
        engine, parent, diagnosis = fixture([1., 1., 1., 1.], 4., .75)
        engine.high_influence_set = lambda *_: {('J00', 0), ('J01', 0)}
        with self.assertRaisesRegex(module.HLNSDiagnosisError, 'H_not_minimal_rho_prefix'):
            engine._diagnostic_high(parent)

    def test_too_long_prefix_still_rejected(self):
        engine, parent, diagnosis = fixture([1., 1., 1., 1.], 4., .5)
        engine.high_influence_set = lambda *_: set(diagnosis.phi)
        with self.assertRaisesRegex(module.HLNSDiagnosisError, 'H_not_minimal_rho_prefix'):
            engine._diagnostic_high(parent)

    def test_wrong_operations_still_rejected(self):
        engine, parent, diagnosis = fixture([1., 1., 1., 1.], 4., .5)
        engine.high_influence_set = lambda *_: {('J01', 0), ('J02', 0)}
        with self.assertRaisesRegex(module.HLNSDiagnosisError, 'H_not_minimal_rho_prefix'):
            engine._diagnostic_high(parent)


if __name__ == '__main__':
    unittest.main()
