"""DISVA fidelity, frozen sets and exact neutral restricted atomic semantics."""

import copy
import random
import unittest
from unittest.mock import patch

from experiments.diagnostics import diagnosis_subproblem_viability as d
from experiments import sis_local_viability as local
from experiments import run_pre_ablation as r
from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from tests.test_wip_graph_dual_population import fixture


class DISVATests(unittest.TestCase):
    def setUp(self):
        ops, buffers, seq, ms = fixture()
        self.engine = WIPGraphDualPopulation(ops, buffers, N=3, N_A=12, FE_max=6, seed=7)
        self.ind = d.offline_decode(self.engine.decoder, Individual(seq, ms,
                                    operation_order=tuple(self.engine.encoder.ms_index_order)), full=True)
        diagnosis = self.engine.diagnose_shortage(self.ind)
        high = self.engine.high_influence_set(self.ind, diagnosis)
        self.engine.local_instance, self.engine.local_seed = "fixture", 0
        self.state = local.capture_state(self.engine, self.ind, diagnosis, high)
        self.assertIsNotNone(self.state)
        self.features = d.opportunity_features(self.engine, self.ind)
        self.ops = self.engine.encoder.ms_index_order

    def test_phi_H_reconstruction(self):
        rebuilt = d.verify_state(self.engine, self.state)
        self.assertEqual(rebuilt.shortage, self.ind.shortage)
        modified = copy.deepcopy(self.state)
        modified["diagnosis"]["influence_dependencies"].pop()
        with self.assertRaises(AssertionError):
            d.verify_state(self.engine, modified)

    def test_positive_support_equality(self):
        diagnosis = self.engine.diagnose_shortage(self.ind)
        expected = {op for op, value in diagnosis.phi.items() if value > 0}
        actual = {tuple(op) for op, value in self.state["phi"] if value > 0}
        self.assertEqual(expected, actual)

    def test_H_mass_coverage(self):
        phi = {tuple(op): value for op, value in self.state["phi"]}
        mass = sum(phi[tuple(op)] for op in self.state["H"])
        self.assertGreaterEqual(mass/self.state["S_seed"], self.engine.rho)

    def test_H_minimality(self):
        phi = {tuple(op): value for op, value in self.state["phi"]}
        ordered = sorted(phi, key=lambda op: (-phi[op], op))
        n = len(self.state["H"])
        self.assertLess(sum(phi[op] for op in ordered[:n-1])/self.state["S_seed"], self.engine.rho)

    def test_all_O_bit_exact_formal_mutation_and_RNG(self):
        for trial in range(100):
            left, right = random.Random(trial), random.Random(trial)
            prior = self.engine.rng
            try:
                self.engine.rng = left
                formal = self.engine.basic_mutation(self.ind)
            finally:
                self.engine.rng = prior
            child, metadata = d.restricted_basic_mutation(self.engine, self.ind, set(self.ops), right)
            self.assertEqual(self.engine.genotype_key(formal), self.engine.genotype_key(child))
            self.assertEqual(left.getstate(), right.getstate())
            d.assert_restricted(self.ind, child, set(self.ops), metadata)

    def test_non_D_never_actively_modified(self):
        allowed = {("J0", 0)}
        for trial in range(40):
            child, meta = d.restricted_basic_mutation(self.engine, self.ind, allowed, random.Random(trial))
            d.assert_restricted(self.ind, child, allowed, meta)
            self.assertEqual(tuple(meta["operation"]), ("J0", 0))

    def test_OS_can_cross_non_D(self):
        allowed = {("J0", 0)}
        os_children = []
        for trial in range(30):
            child, meta = d.restricted_basic_mutation(self.engine, self.ind, allowed, random.Random(trial))
            if meta["move_type"] == "OS":
                self.engine.encoder.validate_os(child.os_seq)
                self.assertNotEqual(child.os_seq, self.ind.os_seq)
                os_children.append(child)
        self.assertTrue(os_children)

    def test_MS_changes_one_assignment_only(self):
        for trial in range(30):
            child, meta = d.restricted_basic_mutation(self.engine, self.ind, set(self.ops), random.Random(trial))
            if meta["move_type"] == "MS":
                self.assertEqual([op for op in self.ops if child.ms_map[op] != self.ind.ms_map[op]],
                                 [tuple(meta["operation"])])

    def fake_better_decode(self, child):
        child.shortage = -100 if self.ind.shortage >= 0 else -200
        child.makespan = self.ind.makespan+10000
        return child

    def test_subset_frozen_across_acceptance(self):
        allowed = frozenset({("J0", 0)})
        observed = []
        original = d.restricted_basic_mutation
        def recorder(engine, ind, subset, rng):
            observed.append(frozenset(subset))
            return original(engine, ind, subset, rng)
        with patch.object(d, "restricted_basic_mutation", recorder):
            rows = d.hill_climb(self.engine, self.ind, allowed, random.Random(0), decode=self.fake_better_decode)
        self.assertEqual(set(observed), {allowed})
        self.assertGreater(rows[-1]["accepted_moves"], 0)
        self.assertGreater(rows[-1]["Cmax_delta"], 0)

    def test_no_rediagnosis_in_hill_climb(self):
        with patch.object(self.engine, "diagnose_shortage", side_effect=AssertionError("No re-diagnosis")):
            rows = d.hill_climb(self.engine, self.ind, set(self.ops), random.Random(0), decode=self.fake_better_decode)
        self.assertEqual([z["budget"] for z in rows], [8,16,32])
        self.assertEqual(rows[-1]["actual_offline_decodes"], 32)

    def test_subset_construction_never_reads_child_objective(self):
        class SeedOnly(dict):
            def __getitem__(self, key):
                if key in ("shortage", "S_child", "Cmax_child", "gain"):
                    raise AssertionError("Child outcome used")
                return super().__getitem__(key)
        d.construct_subsets(SeedOnly(self.state), self.ops, self.features)

    def test_diagnostic_rng_isolation(self):
        before = local.formal_state_signature(self.engine)
        d.construct_subsets(self.state, self.ops, self.features)
        d.restricted_basic_mutation(self.engine, self.ind, set(self.ops), d.diagnostic_rng(self.state, "test"))
        self.assertEqual(before, local.formal_state_signature(self.engine))

    def test_support_random_exact_size(self):
        subsets = d.construct_subsets(self.state, self.ops, self.features)
        for subset in subsets:
            if subset["kind"] == d.KINDS[1]:
                self.assertEqual(len(subset["D"]), len(self.state["H"]))

    def test_global_random_exact_size(self):
        subsets = d.construct_subsets(self.state, self.ops, self.features)
        for subset in subsets:
            if subset["kind"] == d.KINDS[2]:
                self.assertEqual(len(subset["D"]), len(self.state["H"]))

    def test_degenerate_support_H(self):
        state = {**self.state, "phi": [[op, 1] for op in self.state["H"]]}
        sets = d.construct_subsets(state, self.ops, self.features)
        for subset in sets:
            if subset["kind"] == d.KINDS[1]:
                self.assertTrue(subset["degenerate"])
                self.assertEqual(subset["D"], frozenset(tuple(op) for op in state["H"]))

    def test_exact_match_failure_no_relaxation(self):
        chosen, unmatched = d.matched_subset([("J0",0)], [("J0",1)], self.features, random.Random(0))
        self.assertIsNone(chosen)
        self.assertEqual(unmatched, 1)

    def test_formal_sources_unchanged(self):
        expected = {"src/algorithms/wip_graph_dual_population.py": "40961aa219e32bc23673c47dc66ec393b08baa234e52c0e8f961900e1257022a",
                    "src/solution/encoder.py": "d759879b9a6e285a5b7a2355e0306c517be7368538e5460436422a57c1015d40",
                    "src/solution/decoder.py": "f0139652b6f236e2bf9dac338933af5257220a86339e7103b7dca529f59e3b10"}
        for path, digest in expected.items():
            self.assertEqual(d.role.file_hash(r.PROJECT_ROOT/path), digest)

    def test_exhaustion_no_fallback_or_decode(self):
        rows = d.hill_climb(self.engine, self.ind, frozenset(), random.Random(0),
                            decode=lambda _: self.fail("Should not decode"))
        self.assertTrue(all(z["exhausted"] for z in rows))
        self.assertEqual(rows[-1]["actual_offline_decodes"], 0)

    def test_four_replicates_are_one_state_unit(self):
        rows = [{"instance":"test", "seed":0, "fe_index":1, "kind":d.KINDS[2],
                 "trial":rep, "gain":rep, "Cmax_delta":0, "exhausted":False,
                 "actual_offline_decodes":1} for rep in range(4)]
        states = d.state_metrics(rows, 1)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["replicates"], 4)
        self.assertEqual(states[0]["EAG"], 1.5)

    def test_strict_acceptance_no_equal_move(self):
        def neutral(child):
            child.shortage, child.makespan = self.ind.shortage, 0
            return child
        rows = d.hill_climb(self.engine, self.ind, set(self.ops), random.Random(0), decode=neutral)
        self.assertEqual(rows[-1]["accepted_moves"], 0)
        self.assertEqual(rows[-1]["Cmax_delta"], 0)


if __name__ == "__main__":
    unittest.main()
