import random
import unittest

from src.algorithms.wip_graph_dual_population import (HLNSRepairError, alternative_probabilities,
    biased_machine_repair, biased_topological_os_repair, parent_deviation_probability,
    restricted_atomic_fallback, validate_priority_os)
from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation
from tests.hlns_test_support import restricted_basic_mutation
from tests.hlns_test_support import fixture


class ScriptedRNG:
    def __init__(self, values=(), choice_index=0):
        self.values=iter(values);self.choice_index=choice_index;self.choices=[]

    def random(self):
        return next(self.values)

    def choice(self, options):
        self.choices.append(list(options))
        return options[self.choice_index]


class OSRepairTests(unittest.TestCase):
    parent=[("A",0),("A",1),("B",0),("B",1),("C",0),("C",1)]

    def test_complete_unique_legal(self):
        for seed in range(80):
            child=biased_topological_os_repair(self.parent,self.parent,.4,random.Random(seed))
            self.assertEqual(set(child),set(self.parent));self.assertEqual(len(child),len(set(child)))
            validate_priority_os(child)

    def test_fixed_projection(self):
        high={self.parent[1],self.parent[4]}
        for seed in range(50):
            child=biased_topological_os_repair(self.parent,high,.5,random.Random(seed))
            self.assertEqual([op for op in child if op not in high],[op for op in self.parent if op not in high])

    def test_cross_stage_no_blocks(self):
        parent=self.parent[:4]
        child=biased_topological_os_repair(parent,parent,1,ScriptedRNG([0,.99,0,.99]))
        self.assertEqual(child,[("B",0),("B",1),("A",0),("A",1)])

    def test_empty_D_keeps_skeleton(self):
        self.assertEqual(biased_topological_os_repair(self.parent,set(),.5,ScriptedRNG()),self.parent)

    def test_q_zero_parent(self):
        self.assertEqual(biased_topological_os_repair(self.parent,self.parent,0,random.Random(4)),self.parent)

    def test_single_ready_no_rng(self):
        parent=[("A",0),("A",1),("A",2)]
        self.assertEqual(biased_topological_os_repair(parent,parent,1,ScriptedRNG()),parent)

    def test_weights_normalized_exact_definition(self):
        probs=alternative_probabilities([0,2,5])
        self.assertAlmostEqual(sum(probs),1)
        self.assertAlmostEqual(probs[0],2/3);self.assertAlmostEqual(probs[1],1/3)

    def test_positive_parent_and_remote_probabilities(self):
        probs=alternative_probabilities([2,5,99])
        self.assertTrue(all(p>0 for p in probs))
        self.assertGreater(1-parent_deviation_probability(3),0)

    def test_multiple_reconstructions(self):
        outcomes={tuple(biased_topological_os_repair(self.parent,self.parent,.4,random.Random(i))) for i in range(80)}
        self.assertGreater(len(outcomes),3)

    def test_seed_reproducibility(self):
        a,b=random.Random(9),random.Random(9)
        self.assertEqual(biased_topological_os_repair(self.parent,self.parent,.4,a),biased_topological_os_repair(self.parent,self.parent,.4,b))
        self.assertEqual(a.getstate(),b.getstate())

    def test_unbiased_choice_all_ready_not_q_one(self):
        rng=ScriptedRNG(choice_index=0)
        child=biased_topological_os_repair(self.parent,self.parent,.4,rng,parent_bias=False)
        self.assertEqual(child,self.parent)
        self.assertEqual(rng.choices[0],[0,2,4])

    def test_invalid_parent_not_retried(self):
        with self.assertRaises(HLNSRepairError):biased_topological_os_repair([("A",1),("A",0)],[],.5,ScriptedRNG())

    def test_duplicate_rejected(self):
        with self.assertRaises(HLNSRepairError):biased_topological_os_repair([("A",0),("A",0)],[],.5,ScriptedRNG())

    def test_unknown_relaxed_rejected(self):
        with self.assertRaises(HLNSRepairError):biased_topological_os_repair(self.parent,{("D",0)},.5,ScriptedRNG())

    def test_q_validation(self):
        for q in (-.1,1.1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):biased_topological_os_repair(self.parent,self.parent,q,ScriptedRNG())

    def test_q_formula_no_additional_parameters(self):
        self.assertEqual(parent_deviation_probability(1),.5)
        self.assertEqual(parent_deviation_probability(9),.25)
        for n in (0,-1,True,1.5):
            with self.assertRaises(ValueError):parent_deviation_probability(n)


class MachineRepairTests(unittest.TestCase):
    legal={("A",0):("slow","fast","middle"),("B",0):("only",),("C",0):("m1","m2")}
    parent={("A",0):"slow",("B",0):"only",("C",0):"m1"}

    def test_fixed_and_legal(self):
        for seed in range(30):
            child=biased_machine_repair(self.parent,{("A",0)},self.legal,.5,random.Random(seed))
            self.assertEqual(child[("C",0)],"m1")
            self.assertTrue(all(m in self.legal[op] for op,m in child.items()))

    def test_single_machine_no_draw(self):
        self.assertEqual(biased_machine_repair(self.parent,{("B",0)},self.legal,1,ScriptedRNG()),self.parent)

    def test_parent_retention(self):
        self.assertEqual(biased_machine_repair(self.parent,{("A",0)},self.legal,.4,ScriptedRNG([.9])),self.parent)

    def test_alternative_uniform_list_no_speed_bias(self):
        rng=ScriptedRNG([0],choice_index=-1)
        child=biased_machine_repair(self.parent,{("A",0)},self.legal,.4,rng)
        self.assertEqual(rng.choices,[["fast","middle"]]);self.assertEqual(child[("A",0)],"middle")

    def test_unbiased_includes_current(self):
        rng=ScriptedRNG(choice_index=0)
        self.assertEqual(biased_machine_repair(self.parent,{("A",0)},self.legal,.4,rng,parent_bias=False),self.parent)
        self.assertEqual(rng.choices,[list(self.legal[("A",0)])])

    def test_reproducible(self):
        a,b=random.Random(2),random.Random(2)
        self.assertEqual(biased_machine_repair(self.parent,self.parent,self.legal,.5,a),biased_machine_repair(self.parent,self.parent,self.legal,.5,b))
        self.assertEqual(a.getstate(),b.getstate())

    def test_inputs_unchanged(self):
        before=self.parent.copy();biased_machine_repair(self.parent,self.parent,self.legal,1,random.Random(2))
        self.assertEqual(self.parent,before)

    def test_invalid_machine(self):
        with self.assertRaises(HLNSRepairError):biased_machine_repair({**self.parent,("A",0):"bad"},self.parent,self.legal,.5,random.Random(1))


class RestrictedAdapterTests(unittest.TestCase):
    def setUp(self):
        ops,buffers,seq,ms=fixture()
        self.engine=WIPGraphDualPopulation(ops,buffers,variant="HLNS_FULL",N=3,N_A=12,FE_max=6,seed=0)
        self.parent=Individual(seq,ms,operation_order=tuple(self.engine.encoder.ms_index_order))

    def test_matches_verified_restricted_child_and_rng(self):
        for allowed in (set(self.parent.os_seq),set(self.parent.os_seq[1:4]),{self.parent.os_seq[0]}):
            for seed in range(80):
                a,b=random.Random(seed),random.Random(seed)
                old,meta=restricted_basic_mutation(self.engine,self.parent,allowed,a)
                new=restricted_atomic_fallback(self.engine,self.parent,allowed,b)
                self.assertFalse(meta["exhausted"])
                self.assertEqual(self.engine.genotype_key(old),self.engine.genotype_key(new))
                self.assertEqual(a.getstate(),b.getstate())

    def test_OS_only_adapter(self):
        child=restricted_atomic_fallback(self.engine,self.parent,set(self.parent.os_seq),random.Random(4),component="OS")
        self.assertEqual(child.ms_map,self.parent.ms_map);self.assertNotEqual(child.os_seq,self.parent.os_seq)

    def test_MS_only_adapter(self):
        child=restricted_atomic_fallback(self.engine,self.parent,set(self.parent.os_seq),random.Random(4),component="MS")
        self.assertEqual(child.os_seq,self.parent.os_seq);self.assertNotEqual(child.ms_map,self.parent.ms_map)

    def test_empty_no_rng(self):
        rng=random.Random(4);before=rng.getstate()
        self.assertIsNone(restricted_atomic_fallback(self.engine,self.parent,set(),rng))
        self.assertEqual(rng.getstate(),before)

    def test_restores_formal_rng(self):
        original=self.engine.rng
        restricted_atomic_fallback(self.engine,self.parent,self.parent.os_seq,random.Random(4))
        self.assertIs(self.engine.rng,original)

    def test_exact_with_only_one_available_move_type(self):
        for jobs,machines in ((1,{"m":1,"n":2}),(2,{"m":1})):
            ops={str(j):[{"machines":machines.copy()} for _ in range(2)] for j in range(jobs)}
            engine=WIPGraphDualPopulation(ops,{},variant="HLNS_FULL",N=1,N_A=2,FE_max=2,seed=0)
            seq=[(j,i) for j in ops for i in range(2)]
            parent=Individual(seq,{op:"m" for op in seq},operation_order=tuple(engine.encoder.ms_index_order))
            for seed in range(30):
                a,b=random.Random(seed),random.Random(seed)
                old,_=restricted_basic_mutation(engine,parent,set(seq),a)
                new=restricted_atomic_fallback(engine,parent,set(seq),b)
                self.assertEqual(engine.genotype_key(old),engine.genotype_key(new));self.assertEqual(a.getstate(),b.getstate())

    def test_restore_rng_on_operator_failure(self):
        from unittest.mock import patch
        original=self.engine.rng
        with patch.object(self.engine,"basic_mutation",side_effect=RuntimeError("bug")):
            with self.assertRaises(RuntimeError):restricted_atomic_fallback(self.engine,self.parent,self.parent.os_seq,random.Random(4))
        self.assertIs(self.engine.rng,original)
