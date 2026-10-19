"""Localization, pre-move diagnosis, hierarchy and isolation, no formal search."""

import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.diagnostics import accepted_move_h_drift_audit as a
from experiments.tests.test_coupled_reoptimization_audit import setup_case


def toy():
    operations={job:[{"machines":{"m"+str(i):2,"alternative"+str(i):3}} for i in range(4)] for job in "ABC"}
    engine=SimpleNamespace(operations=operations)
    ms={(job,i):"m"+str(i) for job in operations for i in range(4)}
    node={"ms":ms,"phi":{("A",1):8,("A",2):2},"H":{("A",1)},"S":10,"Cmax":20}
    return engine,node


class AcceptedDriftTests(unittest.TestCase):
    def test_01_membership_four_quadrants(self):
        h0={1,2};ht={1,3}
        self.assertEqual([a.quadrant(i,h0,ht) for i in (1,2,3,4)],list("ABCD"))

    def test_02_overlap(self):
        row=a.overlap({1,2},{2,3,4})
        self.assertEqual(row["jaccard"],.25)
        self.assertEqual(row["retained_H0"],.5)
        self.assertEqual(row["Ht_from_H0"],1/3)
        self.assertEqual((row["added_Ht_not_H0"],row["removed_H0_not_Ht"]),(2,1))

    def test_03_empty_overlap_safe(self):
        self.assertEqual(a.overlap(set(),set())["jaccard"],1)
        self.assertEqual(a.overlap({1},set())["jaccard"],0)

    def test_04_direct_predecessor(self):
        engine,node=toy();rel=a.Region(engine,{("A",1)},node,node).relation(("A",0))
        self.assertTrue(rel["direct_predecessor"]);self.assertEqual(rel["tech_distance"],1)

    def test_05_direct_successor(self):
        engine,node=toy();rel=a.Region(engine,{("A",1)},node,node).relation(("A",2))
        self.assertTrue(rel["direct_successor"])

    def test_06_two_hop(self):
        engine,node=toy();rel=a.Region(engine,{("A",1)},node,node).relation(("A",3))
        self.assertTrue(rel["tech2"]);self.assertEqual(rel["tech_distance"],2)

    def test_07_different_jobs_no_technology_path(self):
        engine,node=toy();self.assertIsNone(a.Region(engine,{("A",1)},node,node).relation(("B",3))["tech_distance"])

    def test_08_seed_same_machine(self):
        engine,node=toy();rel=a.Region(engine,{("A",1)},node,node).relation(("B",1))
        self.assertTrue(rel["seed_machine"])

    def test_09_current_same_machine_separate(self):
        engine,node=toy();current=copy.deepcopy(node);current["ms"][("B",1)]="alternative1"
        rel=a.Region(engine,{("A",1)},node,current).relation(("B",1))
        self.assertTrue(rel["seed_machine"]);self.assertFalse(rel["current_machine"])

    def test_10_eligible_overlap_without_assigned_same(self):
        engine,node=toy();current=copy.deepcopy(node);current["ms"][("B",1)]="alternative1"
        rel=a.Region(engine,{("A",1)},current,current).relation(("B",1))
        self.assertTrue(rel["eligible_machine"]);self.assertFalse(rel["narrow"])

    def test_11_self_excluded_machine(self):
        engine,node=toy();rel=a.Region(engine,{("A",1)},node,node).relation(("A",1))
        self.assertFalse(rel["seed_machine"]);self.assertFalse(rel["current_machine"]);self.assertFalse(rel["eligible_machine"])

    def test_12_remote(self):
        engine,node=toy();rel=a.Region(engine,{("A",1)},node,node).relation(("B",3))
        self.assertEqual(a.location(("B",3),{("A",1)},rel),"remote")

    def test_13_exclusive_priority(self):
        engine,node=toy();rel=a.Region(engine,{("A",1)},node,node).relation(("A",2))
        self.assertEqual(a.location(("A",2),{("A",1)},rel),"tech1")

    def test_14_no_formal_state_change_diagnosis(self):
        engine,parent,state=setup_case();before=a.c.local.formal_state_signature(engine)
        counts=Counter();a.diagnose_node(engine,parent,counts)
        self.assertEqual(before,a.c.local.formal_state_signature(engine))
        self.assertEqual(counts["offline_decodes"],1);self.assertEqual(counts["diagnosis_computations"],1)

    def test_15_source_hash_validation(self):
        relative=a.FROZEN[0];expected=a.c.disva.role.file_hash(a.c.r.PROJECT_ROOT/relative)
        a.verify_sources({"files":{relative:expected},"CRNA_artifacts":{}})
        with self.assertRaises(AssertionError):a.verify_sources({"files":{relative:"wrong"},"CRNA_artifacts":{}})

    def test_16_hierarchy_not_pooled_trials(self):
        rows=[{"instance":"I","seed":0,"checkpoint":1,"trial":0,"policy":"P","FinalGain":0},
              *[{"instance":"I","seed":0,"checkpoint":2,"trial":i,"policy":"P","FinalGain":10} for i in range(8)]]
        states,seeds,instances=a.hierarchical(rows)
        self.assertEqual(instances[0]["FinalGain"],5)

    def test_17_accepted_active_target_not_passive(self):
        engine,parent,state=setup_case()
        _,records,_=a.c.episode(engine,state,"Global-Sequential",0)
        counts=Counter();_,seed,_=a.diagnose_node(engine,parent,counts)
        current=seed
        for step in records:
            if step["accepted"]:
                ind=a.c.r.prefix_individual(step["proposal_genome"])
                _,child,_=a.diagnose_node(engine,ind,counts)
                row=a.accepted_record(engine,state,seed,current,child,step,1)
                self.assertEqual(row["operation"],step["operation"])
                self.assertGreater(row["incremental_gain"],0)
                self.assertEqual(row["in_Hpre"],tuple(step["operation"]) in current["H"])
                current=child

    def test_18_drift_mass(self):
        engine,parent,state=setup_case();counts=Counter();_,node,_=a.diagnose_node(engine,parent,counts)
        row=a.drift(node,node)
        self.assertEqual(row["jaccard"],1);self.assertEqual(row["stage_TV"],0)
        self.assertLessEqual(row["outside_H0_phi_mass"],.2+1e-9)

    def test_19_no_future_H_leakage(self):
        self.assertEqual(a.quadrant("o",{"x"},{"x"}),"D")
        self.assertEqual(a.quadrant("o",{"x"},{"x","o"}),"C")

    def test_20_unstable_positive_mean(self):
        ev=a.evidence([10,-1,-1,-1])
        self.assertEqual(ev["status"],"unstable")

    def test_21_leave_one_out_instability(self):
        ev=a.evidence([100,1,1,-9])
        self.assertFalse(ev["stable_positive"]);self.assertLess(ev["min_LOO_margin"],0)

    def test_22_no_zero_fill_absent_coverage(self):
        self.assertIsNone(a.mean([None,None]));self.assertEqual(a.mean([None,1]),1)

    def test_23_structural_coverage_not_sufficiency(self):
        rows=[]
        for name in a.c.gate.PRIMARY:
            rows.append({"instance":name,"policy":"Global-Sequential","dynamic_outside_gain_share":0,
                         "structural0_outside_gain_share":1,"structural0_enrichment":0,
                         "narrow0_outside_gain_share":.9,"narrow0_enrichment":0,
                         "D_of_outside_gain":1,"Global_minus_H":-1,"outside_H0_gain":10})
        label,_=a.classify(rows)
        self.assertEqual(label,"MIXED_MECHANISM")

    def test_24_tv_and_dominant_stable(self):
        self.assertEqual(a.tv({"1":1},{"2":1}),1)
        self.assertEqual(a.dominant({"b":.5,"a":.5}),"a")

    def test_25_no_proposals_during_audit(self):
        engine,parent,state=setup_case()
        source={"steps":[],"episodes":[]}
        for trial in range(8):
            for policy in a.POLICIES:
                summary,steps,final=a.c.episode(engine,state,policy,trial)
                source["steps"].extend(steps);source["episodes"].append(summary)
        before=a.c.local.formal_state_signature(engine)
        with tempfile.TemporaryDirectory() as tmp,patch.object(a,"ROOT",Path(tmp)), \
             patch.object(engine,"basic_mutation",side_effect=AssertionError("new proposals")), \
             patch.object(engine,"evaluate",side_effect=AssertionError("formal FE")):
            result=a.audit_state(engine,state,source,"unit-test")
            self.assertTrue(result["isolated"])
            self.assertEqual(len(result["episodes"]),16)
        self.assertEqual(before,a.c.local.formal_state_signature(engine))


if __name__=="__main__":unittest.main()
