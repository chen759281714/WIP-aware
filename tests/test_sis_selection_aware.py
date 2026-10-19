"""Selection-aware diagnostic fidelity; formal algorithm remains untouched."""

import copy
import itertools
import random
import unittest
from unittest.mock import patch

from experiments import run_pre_ablation as r
from experiments import sis_local_viability as local
from experiments import sis_selection_aware as v
from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from tests.test_wip_graph_dual_population import fixture


class SelectionAwareTests(unittest.TestCase):
    def setUp(self):
        ops, buffers, seq, ms = fixture()
        self.engine = WIPGraphDualPopulation(ops, buffers, N=3, N_A=3, FE_max=24, seed=1)
        self.seed = Individual(seq, ms, makespan=100, shortage=100,
                               operation_order=tuple(self.engine.encoder.ms_index_order))

    def test_basic_fixed_rng_exact_genes_and_rng_consumption(self):
        original = self.engine.rng
        original_state = original.getstate()
        for seed in range(40):
            actual_rng, expected_rng = random.Random(seed), random.Random(seed)
            actual = v.exact_basic(self.engine, self.seed, actual_rng)
            other = WIPGraphDualPopulation(self.engine.operations, self.engine.buffers, N=3, N_A=3, FE_max=24)
            other.rng = expected_rng
            expected = other.basic_mutation(self.seed)
            self.assertEqual(self.engine.genotype_key(actual), other.genotype_key(expected))
            self.assertEqual(actual_rng.getstate(), expected_rng.getstate())
        self.assertIs(self.engine.rng, original)
        self.assertEqual(original.getstate(), original_state)

    def test_basic_rng_restored_on_error(self):
        original = self.engine.rng
        with patch.object(self.engine, "basic_mutation", side_effect=RuntimeError("probe")):
            with self.assertRaises(RuntimeError):
                v.exact_basic(self.engine, self.seed, random.Random(3))
        self.assertIs(self.engine.rng, original)

    def test_streams_deterministic_independent(self):
        state = {"instance": "tiny", "seed": 0, "fe_index": 7}
        self.assertEqual(v.rng_for(state, 1, "BasicMutation").getstate(), v.rng_for(state, 1, "BasicMutation").getstate())
        self.assertNotEqual(v.rng_for(state, 1, "BasicMutation").getstate(), v.rng_for(state, 1, "SupplyAMP_choice").getstate())

    def test_global_distance_explicit_all_incomparable_pairs(self):
        x = self.seed
        pairs = [(a, b) for a, b in itertools.combinations(x.os_seq, 2) if a[0] != b[0]]
        self.assertEqual(len(pairs), len(x.os_seq) * (len(x.os_seq) - 1) // 2 - 3)
        for i in range(25):
            y = v.exact_basic(self.engine, x, random.Random(i))
            rx, ry = ({key: p for p, key in enumerate(z.os_seq)} for z in (x, y))
            d_os = sum((rx[a] < rx[b]) != (ry[a] < ry[b]) for a, b in pairs) / len(pairs)
            d_ms = sum(x.ms_map[k] != y.ms_map[k] for k in x.os_seq) / len(x.os_seq)
            self.assertEqual(self.engine.structural_distance(x, y), .5 * (d_os + d_ms))

    def test_nearest_first_tie_and_strict_acceptance(self):
        population = [self.seed.copy(), self.seed.copy()]
        population[1].shortage = 1
        child = self.seed.copy()
        child.shortage = 99
        replay = v.nearest_result(self.engine, population, child, population[0], 0)
        self.assertEqual(replay["nearest_id"], 0)
        self.assertTrue(replay["nearest_is_seed"])
        detached = list(population)
        self.assertEqual(self.engine.similarity_replace(detached, child, "shortage"), replay["accepted_to_PS"])
        self.assertIs(detached[0], child)
        child.shortage = 100
        child.makespan = 1
        self.assertFalse(v.nearest_result(self.engine, population, child, population[0], 0)["accepted_to_PS"])
        self.assertFalse(self.engine.similarity_replace(list(population), child, "shortage"))

    def test_nearest_random_proposals_exact_replacement(self):
        population = [v.exact_basic(self.engine, self.seed, random.Random(i)) for i in range(8)]
        for i, z in enumerate(population):
            z.shortage, z.makespan = i * 5, 100 - i
        for i in range(30):
            child = v.exact_basic(self.engine, self.seed, random.Random(100 + i))
            child.shortage, child.makespan = i % 20, 80
            replay = v.nearest_result(self.engine, population, child, self.seed)
            detached = list(population)
            self.assertEqual(replay["accepted_to_PS"], self.engine.similarity_replace(detached, child, "shortage"))
            if replay["accepted_to_PS"]:
                self.assertIs(detached[replay["nearest_id"]], child)

    def test_archive_replay_exact_with_cap_and_isolation(self):
        archive = []
        for i in range(3):
            z = v.exact_basic(self.engine, self.seed, random.Random(5 + i))
            z.makespan, z.shortage = 10 + i * 10, 40 - i * 10
            archive.append(z)
        child = v.exact_basic(self.engine, self.seed, random.Random(777))
        child.makespan, child.shortage = 25, 25
        before = local.formal_state_signature(self.engine)
        original_genes = [self.engine.genotype_key(z) for z in archive]
        result = v.archive_result(self.engine, archive, child)
        other = WIPGraphDualPopulation(self.engine.operations, self.engine.buffers, N=3, N_A=3, FE_max=24)
        other.A = list(archive)
        other.update_archive([child])
        self.assertEqual(result["child_survives_cap"], any(z is child for z in other.A))
        self.assertEqual(before, local.formal_state_signature(self.engine))
        self.assertEqual(original_genes, [self.engine.genotype_key(z) for z in archive])

    def test_archive_duplicate_is_not_new_child(self):
        child = self.seed.copy()
        result = v.archive_result(self.engine, [self.seed], child)
        self.assertTrue(result["child_nondominated_vs_archive"])
        self.assertTrue(result["child_genotype_already_archived"])
        self.assertFalse(result["child_inserted"])
        self.assertFalse(result["child_survives_cap"])

    def test_archive_trials_independent(self):
        good = self.seed.copy()
        good.ms_map[next(iter(good.ms_map))] = "different"
        good.shortage = 50
        baseline = copy.deepcopy([self.seed])
        first = v.archive_result(self.engine, baseline, good)
        second = v.archive_result(self.engine, baseline, good)
        self.assertEqual(first, second)
        self.assertEqual(baseline[0].shortage, 100)

    def test_metrics_rejected_negative_gain_truncated(self):
        child = self.seed.copy()
        child.shortage = 150
        out = v.nearest_result(self.engine, [self.seed], child, self.seed, 0)
        self.assertEqual(out["replacement_gain"], 0)
        self.assertFalse(out["accepted_to_PS"])

    def test_state_paired_unit_not_trial_unit(self):
        values = [{"EAG_FE_difference": z} for z in (2, 0, -1)]
        self.assertEqual(v.paired(values), {"states": 3, "W": 1, "T": 1, "L": 1,
                         "mean_difference": 1 / 3, "median_difference": 0})

    def test_quantiles(self):
        self.assertEqual(v.quantile([0, 10, 20, 30], .25), 7.5)
        self.assertEqual(v.quantile([0, 10, 20, 30], .75), 22.5)
        self.assertIsNone(v.quantile([], .5))

    def test_repeated_cached_proposals_keep_trial_weights(self):
        child = self.seed.copy()
        child.shortage = 80
        near = v.nearest_result(self.engine, [self.seed], child, self.seed, 0)
        row = {**near, "child_survives_cap": False, "delta_S_seed": 20,
               "population_or_archive_useful": True}
        rows = [dict(row) for _ in range(8)]
        result = v.metrics(rows)
        self.assertEqual(result["generated"], 8)
        self.assertEqual(result["accepted"], 8)
        self.assertEqual(result["EAG_FE"], 20)

    def test_preregistered_pass_fail_and_ambiguous(self):
        states = [{"instance": name, "seed": seed, "EAG_FE_difference": 1}
                  for name in ("P2", "P3") for seed in range(3) for _ in range(2)]
        groups = [{"instance": name, "seed": seed, "mean_difference": 1}
                  for name in ("P2", "P3") for seed in range(3)]
        policies = [{"policy": p, "archive_survival_rate": .2,
                     "mean_relative_makespan_penalty_accepted": .01} for p in v.POLICIES]
        self.assertEqual(v.decide(states, groups, policies)["verdict"], "PASS")
        # Positive pooled mean cannot override an unfavourable state W/L count.
        bad = copy.deepcopy(states)
        for s in bad:
            s["EAG_FE_difference"] = -1
        bad[0]["EAG_FE_difference"] = 20
        self.assertEqual(v.decide(bad, groups, policies)["verdict"], "FAIL")
        policies[0]["archive_survival_rate"] = 0
        self.assertEqual(v.decide(states, groups, policies)["verdict"], "AMBIGUOUS")

    def test_roundtrip_genotype_and_objectives(self):
        record = {"genome": r.prefix_genome(self.seed), "makespan": 100, "shortage": 100}
        restored = v.restore_ind(record)
        self.assertEqual(self.engine.genotype_key(restored), self.engine.genotype_key(self.seed))
        self.assertEqual((restored.makespan, restored.shortage), (100, 100))

    def test_capture_off_on_rng_and_full_checkpoint_sequence(self):
        rows = []
        baseline = WIPGraphDualPopulation(self.engine.operations, self.engine.buffers, N=3, N_A=3, FE_max=24, seed=1)
        baseline.run(sis_diagnostic_callback=rows.append)
        expected = {"children": rows}
        runs = []
        for capture in (False, True):
            observed = v.FidelityEngine(self.engine.operations, self.engine.buffers, N=3, N_A=3, FE_max=24, seed=1)
            observed.configure(r.Task("tiny", "full", 1), [], expected, capture)
            observed.run(diagnostic_callback=lambda e: e.checkpoint("round"), sis_diagnostic_callback=lambda row: None)
            runs.append((observed.checkpoints, local.formal_state_signature(observed)))
            self.assertEqual(local.formal_state_signature(baseline), local.formal_state_signature(observed))
        self.assertEqual(runs[0], runs[1])


if __name__ == "__main__":
    unittest.main()
