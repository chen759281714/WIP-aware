"""Frozen global partial path relinking semantics, checked against exhaustive gaps."""

import copy
import math
import random
import unittest
from itertools import permutations
from unittest.mock import patch

from src.algorithms.wip_graph_dual_population import (
    Individual, RelinkUnit, ShortageDiagnosis, WIPGraphDualPopulation,
)
from src.solution.decoder import ActiveDependency, DecodeProvenance, ProvenanceEvent
from tests.test_wip_graph_dual_population import fixture


def make_search(jobs="AB", stages=2, **kwargs):
    ops = {job: [{"machines": {"M0": 1, "M1": 2}, "buffer_in": None,
                  "buffer_out": None} for _ in range(stages)] for job in jobs}
    return WIPGraphDualPopulation(ops, {}, N=1, FE_max=100, seed=7, **kwargs)


def individual(seq, shortage=10, makespan=10, machine="M0"):
    return Individual(seq[:], {key: machine for key in seq}, makespan=makespan,
                      shortage=shortage, provenance=DecodeProvenance(),
                      operation_order=tuple(seq))


def diagnosis(phi):
    return ShortageDiagnosis([], phi.copy(), set(), [])


def explicit_d(seq, guide, key):
    return sum((seq.index(key) < seq.index(other)) !=
               (guide.index(key) < guide.index(other))
               for other in seq if other[0] != key[0])


def exhaustive_action(search, x, y, key):
    old = x.os_seq.index(key)
    remaining = [other for other in x.os_seq if other != key]
    candidates = []
    for gap in range(len(remaining) + 1):
        seq = remaining[:gap] + [key] + remaining[gap:]
        try:
            search.encoder.validate_os(seq)
        except ValueError:
            continue
        candidates.append((explicit_d(seq, y.os_seq, key), abs(gap - old), gap))
    best = min(candidates)
    return best if best[0] < explicit_d(x.os_seq, y.os_seq, key) else None


class PSRelinkingTests(unittest.TestCase):
    def setUp(self):
        self.search = make_search()
        self.x = individual([("A", 0), ("A", 1), ("B", 0), ("B", 1)])
        self.y = individual([("A", 0), ("B", 0), ("A", 1), ("B", 1)], shortage=9)

    def vary(self, search, x, y, phi, evidence=frozenset()):
        with patch.object(search, "diagnose_shortage", return_value=diagnosis(phi)), \
             patch.object(search, "_machine_shortage_evidence", return_value=evidence):
            return search.vary_shortage(x, [y])

    def test_ps01_cross_stage_discrepancy(self):
        key = ("A", 1)
        self.assertEqual(self.search._os_discrepancy(self.x, self.y, key), 1)
        action = self.search._best_os_relink_action(self.x, self.y, key)
        self.assertEqual((action.target_index, action.d_before, action.d_after), (2, 1, 0))
        units = self.search._build_relink_units(self.x, self.y, {key}, diagnosis({key: 1}))
        self.assertEqual(units, (RelinkUnit(key, "OS"),))

    def test_ps02_same_job_pairs_excluded(self):
        # Even deliberately reversing a same-job pair does not contribute to D_o.
        y = individual([("A", 1), ("A", 0), ("B", 0), ("B", 1)])
        self.assertEqual(self.search._os_discrepancy(self.x, y, ("A", 0)), 0)
        self.assertEqual(self.search._os_discrepancy(self.x, y, ("A", 1)), 0)
        with self.assertRaises(ValueError):
            self.search.encoder.validate_os(y.os_seq)

    def test_ps03_global_best_crosses_non_h_and_other_stage(self):
        search = make_search("ABC")
        x = individual([("A", 0), ("B", 0), ("B", 1), ("C", 0), ("C", 1), ("A", 1)])
        y = individual([("A", 0), ("A", 1), ("B", 0), ("B", 1), ("C", 0), ("C", 1)])
        action = search._best_os_relink_action(x, y, ("A", 1))
        self.assertEqual((action.target_index, action.d_after, action.displacement), (1, 0, 4))
        self.assertEqual(search._build_relink_units(x, y, {("A", 1)}, diagnosis({("A", 1): 1})),
                         (RelinkUnit(("A", 1), "OS"),))

    def test_ps04_only_strict_reducing_gap(self):
        action = self.search._best_os_relink_action(self.x, self.y, ("A", 1))
        self.assertEqual(action.target_index, 2)
        self.assertEqual(exhaustive_action(self.search, self.x, self.y, ("A", 1)), (0, 1, 2))

    def test_ps05_displacement_and_stable_index_ties(self):
        search = make_search("ABCD", stages=1)
        x = individual([(j, 0) for j in "CBAD"])
        y = individual([(j, 0) for j in "ABCD"])
        action = search._best_os_relink_action(x, y, ("B", 0))
        self.assertEqual((action.d_after, action.displacement, action.target_index), (1, 1, 0))
        # The exhaustive oracle checks unequal and equal displacement ties over all permutations.
        for seq in permutations(x.os_seq):
            current = individual(list(seq))
            expected = exhaustive_action(search, current, y, ("B", 0))
            actual = search._best_os_relink_action(current, y, ("B", 0))
            self.assertEqual(None if actual is None else
                             (actual.d_after, actual.displacement, actual.target_index), expected)

    def test_ps06_equal_d_pair_correction_is_rejected(self):
        search = make_search("ABC")
        x = individual([("A", 0), ("B", 0), ("A", 1), ("C", 0), ("B", 1), ("C", 1)])
        y = individual([("A", 0), ("A", 1), ("C", 0), ("B", 0), ("B", 1), ("C", 1)])
        key = ("C", 0)
        old_move = search._move_before(x, key, ("B", 0))
        self.assertEqual(explicit_d(x.os_seq, y.os_seq, key), 1)
        self.assertEqual(explicit_d(old_move.os_seq, y.os_seq, key), 1)
        self.assertIsNone(search._best_os_relink_action(x, y, key))
        self.assertEqual(search._build_relink_units(x, y, {key}, diagnosis({key: 1})), ())

    def test_ps07_unique_unit_for_multiple_insertions(self):
        search = make_search("ABC", stages=1)
        x = individual([(j, 0) for j in "CBA"])
        y = individual([(j, 0) for j in "ABC"])
        key = ("B", 0)
        self.assertEqual(search._build_relink_units(x, y, {key}, diagnosis({key: 1})),
                         (RelinkUnit(key, "OS"),))

    def test_ps08_initial_unit_at_most_once(self):
        search = make_search("ABCDEF", stages=1, p_mut=0)
        x = individual([(j, 0) for j in "ABCDEF"], shortage=21)
        y = individual(list(reversed(x.os_seq)), shortage=20)
        phi = {(j, 0): 6-i for i, j in enumerate("ABCDEF")}
        self.vary(search, x, y, phi)
        record = search.diagnostics["relinking"][-1]
        applied = [(item["key"], item["kind"]) for item in record["applied_units"]]
        self.assertEqual(record["K"], 2)
        self.assertEqual(applied.count((("A", 0), "OS")), 1)
        self.assertEqual(len(applied), len(set(applied)))

    def test_ps09_never_moves_non_h_target(self):
        y = individual([("B", 0), ("B", 1), ("A", 0), ("A", 1)])
        self.assertEqual(self.search._build_relink_units(self.x, y, {("A", 0)},
                                                       diagnosis({("A", 0): 10})), ())

    def test_ps10_k_uses_unique_initial_units(self):
        search = make_search("ABCDEF", stages=1, rho=1, eta=0.4, p_mut=0)
        x = individual([(j, 0) for j in "ABCDEF"], shortage=6)
        y = individual(list(reversed(x.os_seq)), shortage=5)
        self.vary(search, x, y, dict.fromkeys(x.os_seq, 1))
        record = search.diagnostics["relinking"][-1]
        self.assertEqual(record["initial_unit_count"], 6)
        self.assertEqual(record["K"], max(1, math.ceil(0.4 * 6)))
        self.assertEqual(record["applied_unit_count"], 3)
        self.assertEqual([item["key"] for item in record["applied_units"]],
                         [("A", 0), ("B", 0), ("C", 0)])

    def test_ps11_dynamic_best_action_fixed_initial_units(self):
        search = make_search("ABC", stages=1, rho=1, eta=1, p_mut=0)
        x = individual([(j, 0) for j in "ABC"], shortage=15)
        y = individual(list(reversed(x.os_seq)), shortage=14)
        self.assertEqual(search._best_os_relink_action(x, y, ("B", 0)).target_index, 0)
        child = self.vary(search, x, y, {("A", 0): 10, ("B", 0): 5})
        record = search.diagnostics["relinking"][-1]
        self.assertEqual(record["initial_units"], [(("A", 0), "OS"), (("B", 0), "OS")])
        self.assertEqual([item["target_index"] for item in record["applied_units"]], [2, 1])
        self.assertEqual(child.os_seq, y.os_seq)

    def test_ps12_ms_difference_and_machine_evidence(self):
        y = individual(self.x.os_seq, machine="M1")
        key = ("A", 0)
        prov = self.x.provenance
        prov.events = {1: ProvenanceEvent(1, 0, "release", "A", 0, "M0"),
                       2: ProvenanceEvent(2, 0, "start", "B", 0, "M0")}
        diag = diagnosis({key: 1})
        diag.influence_dependencies = [ActiveDependency(1, 2, "machine", machine="M0")]
        self.assertEqual(self.search._build_relink_units(self.x, y, {key}, diag),
                         (RelinkUnit(key, "MS"),))

    def test_ps13_ms_difference_without_evidence(self):
        y = individual(self.x.os_seq, machine="M1")
        self.assertEqual(self.search._build_relink_units(self.x, y, {("A", 0)},
                                                       diagnosis({("A", 0): 1})), ())

    def test_ps14_same_ms_with_evidence(self):
        self.assertEqual(self.search._build_relink_units(self.x, self.x, {("A", 0)},
                         diagnosis({("A", 0): 1}), frozenset({("A", 0)})), ())

    def test_ps15_self_prefers_strictly_better_shortage(self):
        equal = individual(self.y.os_seq, shortage=10)
        self.assertIs(self.search.choose_guide(self.x, [equal, self.y], {("A", 1)},
                                               diagnosis({("A", 1): 1})), self.y)

    def test_ps16_self_equal_shortage_fallback(self):
        equal = individual(self.y.os_seq, shortage=10)
        self.assertIs(self.search.choose_guide(self.x, [equal], {("A", 1)},
                                               diagnosis({("A", 1): 1})), equal)

    def test_ps17_better_shortage_empty_u0_is_invalid(self):
        equal_genotype = individual(self.x.os_seq, shortage=1)
        self.assertIsNone(self.search.choose_guide(self.x, [equal_genotype], {("A", 1)},
                                                   diagnosis({("A", 1): 1})))

    def test_ps18_archive_allows_worse_shortage_and_keeps_sparse_rule(self):
        low_m = individual(self.x.os_seq, shortage=40, makespan=1)
        better = individual(self.y.os_seq, shortage=5, makespan=10)
        worse = individual(self.y.os_seq, shortage=15, makespan=20)
        low_s = individual(self.x.os_seq, shortage=1, makespan=30)
        self.search.A = [low_m, better, worse, low_s]
        with patch.object(self.search.rng, "sample", return_value=[better, worse]) as sample, \
             patch.object(self.search, "_crowding", return_value={id(better): 1, id(worse): 100}):
            self.assertIs(self.search.choose_archive_guide_for_shortage(
                self.x, diagnosis({("A", 1): 1}), {("A", 1)}), worse)
        self.assertEqual(sample.call_args.args[0], [better, worse])

    def test_ps19_no_intermediate_decode_final_decode_once(self):
        ops, buffers, os_seq, ms = fixture()
        search = WIPGraphDualPopulation(ops, buffers, N=1, FE_max=10, seed=0,
                                        rho=1, eta=1, p_mut=0)
        x = search.evaluate(Individual(os_seq, ms))
        x.shortage = len(os_seq)
        y = individual([(job, op) for job in reversed(tuple(ops)) for op in range(2)],
                       shortage=0)
        y.ms_map = {key: next(m for m in ops[key[0]][key[1]]["machines"] if m != ms[key])
                    for key in os_seq}
        before = search.n_evaluations
        with patch.object(search.decoder, "decode", wraps=search.decoder.decode) as decode:
            child = self.vary(search, x, y, dict.fromkeys(os_seq, 1), frozenset(os_seq))
            self.assertEqual(decode.call_count, 0)
            self.assertEqual(search.n_evaluations, before)
            self.assertGreater(search.diagnostics["relinking"][-1]["applied_unit_count"], 1)
            search.evaluate(child)
            self.assertEqual(decode.call_count, 1)
        self.assertEqual(search.n_evaluations, before + 1)

    def test_ps20_diagnosis_h_evidence_and_u0_fixed(self):
        search = make_search("ABC", stages=1, rho=1, eta=1, p_mut=0)
        x = individual([(j, 0) for j in "ABC"], shortage=15)
        y = individual(list(reversed(x.os_seq)), shortage=14, machine="M1")
        phi = {("A", 0): 10, ("B", 0): 5}
        diag = diagnosis(phi)
        saved = copy.deepcopy(diag)
        evidence = frozenset({("A", 0)})
        with patch.object(search, "diagnose_shortage", return_value=diag) as diagnose, \
             patch.object(search, "high_influence_set", wraps=search.high_influence_set) as high, \
             patch.object(search, "_machine_shortage_evidence", return_value=evidence) as machine:
            self.assertEqual(search.vary_shortage(x, [y]).os_seq, y.os_seq)
        self.assertEqual((diagnose.call_count, high.call_count, machine.call_count), (1, 1, 1))
        self.assertEqual(diag, saved)
        record = search.diagnostics["relinking"][-1]
        self.assertEqual(record["initial_units"], [(("A", 0), "OS"), (("A", 0), "MS"),
                                                    (("B", 0), "OS")])
        self.assertEqual([(item["key"], item["kind"]) for item in record["applied_units"]],
                         record["initial_units"])
        self.assertEqual(record["machine_evidence"], sorted(evidence))

    def test_exhaustive_oracle_all_targets_random_legal_os(self):
        search = make_search("ABC", stages=3)
        rng = random.Random(19)
        for _ in range(50):
            search.encoder.rng = rng
            x = individual(search.encoder.generate_random_os())
            y = individual(search.encoder.generate_random_os())
            for key in x.os_seq:
                self.assertEqual(search._os_discrepancy(x, y, key), explicit_d(x.os_seq, y.os_seq, key))
                action = search._best_os_relink_action(x, y, key)
                actual = None if action is None else (action.d_after, action.displacement, action.target_index)
                self.assertEqual(actual, exhaustive_action(search, x, y, key))

    def test_specialized_ties_do_not_consume_rng(self):
        search = make_search("ABC", stages=1, rho=1, eta=1, p_mut=0)
        x = individual([(j, 0) for j in "ABC"], shortage=3)
        y = individual(list(reversed(x.os_seq)), shortage=2)
        expected = random.Random(7)
        expected.random()  # Existing mutation-vs-specialized branch draw.
        self.vary(search, x, y, dict.fromkeys(x.os_seq, 1))
        self.assertEqual(search.rng.getstate(), expected.getstate())

    def test_empty_u0_falls_back_without_fabricated_unit(self):
        self.search.p_mut = 0
        marker = self.x.copy()
        with patch.object(self.search, "diagnose_shortage", return_value=diagnosis({("A", 0): 10})), \
             patch.object(self.search, "basic_mutation", return_value=marker):
            self.assertIs(self.search.vary_shortage(self.x, [self.x]), marker)
        self.assertEqual(self.search.diagnostics["relinking"], [])

    def test_initial_unit_can_become_unactionable_and_is_skipped(self):
        search = make_search("ABC", stages=1, rho=1, eta=1, p_mut=0)
        x = individual([(j, 0) for j in "ABC"], shortage=6)
        y = individual(list(reversed(x.os_seq)), shortage=5)
        self.vary(search, x, y, {("A", 0): 3, ("B", 0): 2, ("C", 0): 1})
        record = search.diagnostics["relinking"][-1]
        self.assertEqual(record["initial_unit_count"], 3)
        self.assertEqual(record["K"], 3)
        self.assertEqual(record["applied_unit_count"], 2)
        self.assertIn((("C", 0), "OS"), record["skipped_units"])

    def test_ms_move_rejects_illegal_guide_machine(self):
        self.search.p_mut = 0
        y = individual(self.x.os_seq, shortage=0, machine="illegal")
        with self.assertRaisesRegex(ValueError, "Illegal guide machine"):
            self.vary(self.search, self.x, y, {("A", 0): 10}, frozenset({("A", 0)}))

    def test_same_seed_reproduces_full_trace(self):
        ops, buffers, _, _ = fixture()
        def run():
            search = WIPGraphDualPopulation(ops, buffers, N=3, N_A=8, FE_max=80,
                                            seed=31, T_coop=2)
            evaluated = []
            original = search.evaluate
            def record(ind):
                result = original(ind)
                evaluated.append((search.genotype_key(result), result.makespan,
                                  result.shortage, copy.deepcopy(result.provenance)))
                return result
            search.evaluate = record
            search.run()
            populations = [[(search.genotype_key(ind), ind.makespan, ind.shortage)
                            for ind in pop] for pop in (search.P_M, search.P_S, search.A)]
            return evaluated, populations, search.diagnostics, search.rng.getstate()
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
