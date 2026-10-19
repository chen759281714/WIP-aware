"""Residual identities, structural boundary rules and leakage-safe utilities."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.diagnostics import action_event_bridge as v
from experiments.diagnostics import shortage_sensitivity as s
from experiments import sis_local_viability as local
from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation
from tests.test_wip_graph_dual_population import fixture


def chain():
    return {"order":[1,2,3],"edges":[{"src":1,"dst":2,"alpha":1,"potential":True},
            {"src":2,"dst":3,"alpha":1,"potential":True}],"g":{1:0,2:0,3:1},"nonsmooth":{}}


def topology_fixture():
    nodes=[]
    for i,job in enumerate(("P","A","B","X","Y")):
        for j,kind in enumerate(("start","complete","release")):
            node=i*3+j+1
            nodes.append({"id":node,"uid":[job,0,kind],"kind":kind,"time":i*10+j,"logical_id":node})
    mapping={tuple(n["uid"]):n["id"] for n in nodes}
    edge=lambda a,b:{"src":mapping[(a,0,"release")],"dst":mapping[(b,0,"start")],"kind":"start_machine","alpha":1,"potential":True}
    seed={"nodes":nodes,"machine_sequences":{"M1":[[j,0] for j in ("P","A","B")],"M2":[[j,0] for j in ("X","Y")]},
          "edges":[edge("P","A"),edge("A","B"),edge("X","Y")],
          "release_types":{str(n["id"]):{"type":"unblocked","trigger":None} for n in nodes if n["kind"]=="release"}}
    child=copy.deepcopy(seed)
    child["machine_sequences"]={"M1":[[j,0] for j in ("P","B")],"M2":[[j,0] for j in ("X","A","Y")]}
    child["edges"]=[edge("P","B"),edge("X","A"),edge("A","Y")]
    return seed,child,mapping


class BridgeTests(unittest.TestCase):
    def test_processing_residual_only_complete(self):
        graph=chain()
        e=v.residual(graph,{1:0,2:.001,3:.001})
        self.assertEqual(e,{1:0,2:.001,3:0})

    def test_single_machine_parent_change_shock_at_start(self):
        graph=chain()
        self.assertEqual(v.residual(graph,{1:4,2:4,3:4}),{1:4,2:0,3:0})

    def test_downstream_deltas_not_double_counted(self):
        graph=chain();graph["lambda"]=s.reverse(graph)
        dt={1:2,2:5,3:5};e=v.residual(graph,dt)
        self.assertEqual(e,{1:2,2:3,3:0})
        self.assertEqual(-sum(graph["lambda"][n]*x for n,x in e.items()),-5)
        self.assertEqual(-sum(graph["lambda"][n]*dt[n] for n in (1,2)),-7)

    def test_two_independent_roots_no_double_count(self):
        graph={"order":[1,2,3,4],"edges":[{"src":1,"dst":2,"alpha":1},{"src":3,"dst":4,"alpha":1}],"g":{1:0,2:1,3:0,4:2}}
        shifts={1:2,2:2,3:3,4:3};e=v.residual(graph,shifts)
        lam=s.reverse(graph)
        self.assertEqual(-sum(lam[n]*e[n] for n in e),-8)

    def test_old_machine_removal_boundary(self):
        before,after,ids=topology_fixture()
        roots=v.boundary_roots(before,after,("A",0),"M1","M2",2)
        self.assertIn("old_machine_removal_successor",roots[ids[("B",0,"start")]])

    def test_new_machine_insertion_boundary(self):
        before,after,ids=topology_fixture()
        roots=v.boundary_roots(before,after,("A",0),"M1","M2",2)
        self.assertIn("new_machine_insertion_successor",roots[ids[("Y",0,"start")]])

    def test_machine_processing_change_is_exact_input(self):
        graph=chain()
        self.assertEqual(v.residual(graph,{1:3,2:8,3:8})[2],5)

    def test_topology_diff_deterministic(self):
        before,after,_=topology_fixture()
        first=v.edge_diff(before,after,True)
        before["edges"].reverse();after["edges"].reverse()
        self.assertEqual(first,v.edge_diff(before,after,True))
        self.assertEqual(len(first["deleted"]),3)
        self.assertEqual(len(first["added"]),3)

    def test_residual_forward_reconstruction(self):
        graph=chain();dt={1:-7,2:11,3:19}
        self.assertEqual(s.forward(graph,v.residual(graph,dt)),dt)

    def test_seed_only_slot_scoring_does_not_read_objective(self):
        class SeedOnly(dict):
            def __getitem__(self,key):
                if key in ("actual","shortage","S_child"):
                    raise AssertionError("Outcome access")
                return super().__getitem__(key)
        schedule=[SeedOnly(job="P",op=0,machine="M1",start=0,release=5),
                  SeedOnly(job="Q",op=0,machine="M1",start=10,release=15)]
        slot=v.ect_slot_metadata(schedule,("A",0),"M1",3,2)
        self.assertEqual(slot["predicted_start"],5)
        self.assertEqual(slot["predicted_completion"],7)
        self.assertEqual(slot["seed_slot_predecessor"],["P",0])
        self.assertEqual(slot["seed_slot_successor"],["Q",0])
        self.assertFalse(slot["factual_child_neighbors_guaranteed"])

    def test_score_freeze_before_decode_and_no_outcome_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            cert=v.freeze_scores([{"candidate":"M2","score":3}],path)
            def decode():
                self.assertTrue((path/"score_freeze_certificate.json").exists())
                self.assertEqual(v.role.file_hash(path/"candidate_scores_frozen.csv"),cert["sha256"])
                return 5
            self.assertEqual(decode(),5)
            self.assertFalse(cert["child_objectives_loaded"])
            with self.assertRaises(ValueError):
                v.freeze_scores([{"score":3,"actual":5}],path)

    def test_read_only_formal_state_equivalence(self):
        ops,buffers,_,_=fixture()
        engine=WIPGraphDualPopulation(ops,buffers,N=3,N_A=12,FE_max=6,seed=1)
        before=local.formal_state_signature(engine)
        graph=chain();graph["lambda"]=s.reverse(graph)
        s.invalid_path(graph,1)
        old=copy.deepcopy(graph)
        v.residual(graph,{1:1,2:2,3:2})
        v.shock_statistics(graph,{1:1,2:1,3:0})
        self.assertEqual(graph,old)
        self.assertEqual(before,local.formal_state_signature(engine))

    def test_zero_weighted_mass_not_high_smooth(self):
        graph=chain();graph["g"]={n:0 for n in graph["order"]};graph["lambda"]=s.reverse(graph)
        self.assertEqual(v.shock_statistics(graph,{1:2,2:0,3:0})["coverage_bin"],"zero_weighted")

    def test_total_lambda_path_smooth_not_only_local(self):
        graph=chain();graph["lambda"]=s.reverse(graph);graph["nonsmooth"][3]=["tie"]
        result=v.shock_statistics(graph,{1:2,2:0,3:0})
        self.assertEqual(result["shock_nodes_smooth"],1)
        self.assertEqual(result["shock_nodes_path_valid"],0)
        self.assertEqual(result["coverage_bin"],"Low")

    def test_boundary_membership_has_no_benefit_input(self):
        before,after,_=topology_fixture()
        old=copy.deepcopy((before,after))
        v.boundary_roots(before,after,("A",0),"M1","M2",1)
        self.assertEqual((before,after),old)

    def test_zero_processing_change_excludes_complete_root(self):
        before,after,ids=topology_fixture()
        roots=v.boundary_roots(before,after,("A",0),"M1","M2",0)
        self.assertNotIn(ids[("A",0,"complete")],roots)

    def test_shared_family_genotype_deduplicated(self):
        rows=[{"instance":"P","seed":0,"fe_index":3,"child_id":"a","family":f} for f in ("ECT-AMR","FastestProcessingMS")]
        self.assertEqual(len(v.unique_rows(rows)),1)

    def test_scalar_ect_does_not_supply_counterfactual_release(self):
        audit=v.ect_capability_audit()
        self.assertTrue(audit["target_completion"])
        self.assertFalse(audit["target_release_under_blocking"])
        self.assertFalse(audit["factual_counterfactual_machine_neighbors"])


if __name__=="__main__":
    unittest.main()
