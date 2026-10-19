"""Read-only CRNA accepted-move localization and H-drift audit.

PYTHONHASHSEED=0 python3 -m experiments.diagnostics.accepted_move_h_drift_audit
No trajectory replay, proposals, search, FE, RNG draws, or changes to CRNA.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from experiments.diagnostics import coupled_reoptimization_audit as c

ROOT = c.r.PROJECT_ROOT / "experiments/results/accepted_move_h_drift_audit"
POLICIES = ("H-Sequential", "Global-Sequential")
FROZEN = ("src/algorithms/wip_graph_dual_population.py", "src/solution/encoder.py",
          "src/solution/decoder.py", "experiments/core_ps_variants.py")
REGIONS = ("inside_H0", "outside_H0", "tech1", "tech2", "machine", "remote",
           "A", "B", "C", "D", "dynamic", "structural0", "combined", "narrow0",
           "narrow_current", "residual", "seed_machine", "current_machine", "eligible_machine")
RULES = {
    "formal_FE": 0, "replay_FE": 0, "rho": .8, "p_mu": .1,
    "sources": "All 288 original CRNA states and original sequential steps, including 240 unseen; no new proposals or selected outcomes",
    "target": "Original recorded active operation, not passive OS relation changes; verify atomic genotype and parent/proposal/incumbent hashes",
    "technology": "Fixed job route; undirected distance=abs(op_idx difference), infinity/None across jobs. Direct predecessor/successor of any core op; exact two-hop. Stage=op_idx+1",
    "machine": "Exclude self: same assigned machine in seed schedule; same assigned machine in pre-proposal incumbent schedule; eligible-machine intersection. All three flags separately, union for broad structural coverage",
    "structural0": "Distance1 or distance2 OR seed-H0 same-machine OR current-H0 same-machine OR eligible intersection with H0. This is an explanatory set, never an implemented search region",
    "structural_current": "Distance1 or distance2 OR current-Ht same-machine OR eligible intersection with Ht. Narrow sensitivity component omits eligible intersection, specified here before outcomes",
    "A3_partition": "Outside H0 exclusive priority: tech1, else tech2, else machine union, else remote. Flags/coverage sets overlap; never sum overlapping shares",
    "quadrants": "A=in H0,in Hpre; B=in H0,not Hpre; C=not H0,in Hpre; D=neither. Hpre is diagnosis of proposal parent, NEVER post-move H",
    "coverage": "All rates conditional on Global accepted moves outside H0. Dynamic=Hpre; Structural0=structural0; Combined=Hpre OR structural_current(Hpre), NOT union with structural0",
    "exposure": "Same operator candidate operations outside H0 from exact formal atomic_space; fraction covered before proposal. Gain-weight exposure baseline; enrichment is observed coverage minus baseline, descriptive not a randomized causal control",
    "phi_rank": "All operations, absent phi=0; rank=1+number strictly greater phi (ties share rank), percentile=(rank-1)/(O-1), 0=highest. Also save deterministic ordinal tie rank",
    "clusters": "Normalized phi mass by production stage, assigned machine, and job (technological cluster); total-variation shift and dominant-label change vs seed. No learned clustering or flow-causality claim",
    "counts": "One seed reconstruction per state; accepted genotypes decoded/diagnosed once per state with hash cache shared across trials/policies. Rejected proposals never decoded. Count unique actual calls and logical post-acceptance observations separately",
    "hierarchy": "Trial->state->seed->instance; all episodes retained including zero accepted. Additive counts/gains averaged first, instance shares=ratio of hierarchical mean numerators/denominators; no outside gains -> None, never zero coverage. Pooled counts descriptive only",
    "robustness": "Instance mean/median/W/T/L around stated null; all leave-one-instance directions. Positive mean but median<=0, W<=L or LOO<=0 labelled unstable. Seen excluded",
    "classification": "Dynamic supported only if outside gain coverage minus .5 stable. Structural supported only if broad and narrow coverage >.5 AND both exposure enrichments stable. Global supported only if D gain share>.5 stable, structural not supported, and Global-H C2 advantage stable. Multiple supported or conflicting region evidence -> MIXED_MECHANISM; no outside evidence -> INSUFFICIENT_EVIDENCE. These are explanations/next-test recommendations, not efficacy proofs",
}


def register():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 required")
    previous = json.loads((c.ROOT / "preregistration.json").read_text())
    if c.signature(previous) != json.loads((c.ROOT / "analysis/coupled_reoptimization_summary.json").read_text())["fingerprint"]:
        raise AssertionError("CRNA source fingerprint differs")
    files = dict(previous["files"])
    for path in (Path(__file__), c.r.PROJECT_ROOT / "experiments/tests/test_accepted_move_h_drift_audit.py"):
        files[str(path.relative_to(c.r.PROJECT_ROOT))] = c.disva.role.file_hash(path)
    # Protect every CRNA file, including raw, caches, reports, and progress logs.
    artifacts = {str(p.relative_to(c.r.PROJECT_ROOT)): c.disva.role.file_hash(p)
                 for p in sorted(c.ROOT.rglob("*")) if p.is_file()}
    payload = json.loads(json.dumps({"rules":RULES,"files":files,"CRNA_artifacts":artifacts,
                                    "CRNA_fingerprint":c.signature(previous),"python":sys.version}))
    verify_sources(payload)
    path = ROOT / "preregistration.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise AssertionError("Changed preregistration; do not mix audit versions")
    c.r.write_json_atomic(path,payload)
    return c.signature(payload),payload


def verify_sources(payload, all_artifacts=True):
    files = {**payload["files"], **(payload["CRNA_artifacts"] if all_artifacts else {})}
    for relative, expected in files.items():
        if c.disva.role.file_hash(c.r.PROJECT_ROOT/relative) != expected:
            raise AssertionError("Protected source/result changed: "+relative)


def genome_hash(ind):
    return c.selection.digest((tuple(ind.os_seq),tuple(ind.ms_map[k] for k in ind.operation_order)))


def overlap(h0, ht):
    a,b=set(h0),set(ht); intersection=len(a & b)
    return {"H0_size":len(a),"Ht_size":len(b),"jaccard":intersection/len(a|b) if a|b else 1.0,
            "retained_H0":intersection/len(a) if a else 1.0,
            "Ht_from_H0":intersection/len(b) if b else 1.0,
            "added_Ht_not_H0":len(b-a),"removed_H0_not_Ht":len(a-b)}


def quadrant(operation,h0,ht):
    return ("A" if operation in ht else "B") if operation in h0 else ("C" if operation in ht else "D")


class Region:
    """Fixed, indexed relation definitions, not a search neighborhood operator."""
    def __init__(self, engine, high, seed, current, include_seed_machine=True):
        self.high=set(high)
        self.jobs=defaultdict(set)
        self.seed=Counter();self.current=Counter();self.eligible=Counter()
        self.legal={op:set(engine.operations[op[0]][op[1]]["machines"]) for op in current["ms"]}
        self.seed_ms,self.current_ms=seed["ms"],current["ms"]
        self.include_seed_machine=include_seed_machine
        for op in self.high:
            self.jobs[op[0]].add(op[1]);self.seed[seed["ms"][op]]+=1
            self.current[current["ms"][op]]+=1
            self.eligible.update(self.legal[op])

    def relation(self,op):
        indices=self.jobs.get(op[0],set())
        dist=min((abs(op[1]-i) for i in indices),default=None)
        is_self=op in self.high
        seed_machine=self.seed[self.seed_ms[op]]-int(is_self)>0
        current_machine=self.current[self.current_ms[op]]-int(is_self)>0
        eligible=any(self.eligible[m]-int(is_self)>0 for m in self.legal[op])
        pred=op[1]+1 in indices
        succ=op[1]-1 in indices
        two=any(abs(op[1]-i)==2 for i in indices)
        machine=(seed_machine and self.include_seed_machine) or current_machine or eligible
        tech=pred or succ or two
        narrow=tech or (seed_machine and self.include_seed_machine) or current_machine
        return {"direct_predecessor":pred,"direct_successor":succ,"tech2":two,
                "tech_distance":dist,"seed_machine":seed_machine,"current_machine":current_machine,
                "eligible_machine":eligible,"machine":machine,"structural":tech or machine,"narrow":narrow}


def location(op,high,relation):
    if op in high:
        return "inside_H0"
    if relation["direct_predecessor"] or relation["direct_successor"]:
        return "tech1"
    if relation["tech2"]:
        return "tech2"
    return "machine" if relation["machine"] else "remote"


def distribution(phi,labels):
    mass=defaultdict(float)
    total=sum(phi.values())
    for op,value in phi.items():
        mass[str(labels(op))]+=value/total if total else 0
    return dict(mass)


def tv(a,b):
    return .5*sum(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))


def dominant(dist):
    return min(dist,key=lambda k:(-dist[k],k)) if dist else None


def diagnose_node(engine, ind, counts):
    """Offline decode + original diagnosis; all formal counters restored."""
    started=time.perf_counter()
    decoded=c.disva.offline_decode(engine.decoder,c.r.prefix_individual(c.r.prefix_genome(ind)),full=True)
    counts["offline_decodes"]+=1
    saved=copy.deepcopy((engine.diagnostics,engine.unanchored_shortage_intervals,engine.phi_mass_mismatch_count))
    try:
        diagnosis=engine.diagnose_shortage(decoded);counts["diagnosis_computations"]+=1
        high=engine.high_influence_set(decoded,diagnosis);counts["H_computations"]+=1
        if diagnosis.unanchored_intervals or not math.isclose(sum(diagnosis.phi.values()),decoded.shortage,abs_tol=1e-9,rel_tol=0):
            raise AssertionError("Offline diagnosis mass/anchor mismatch")
        order=sorted(diagnosis.phi,key=lambda op:(-diagnosis.phi[op],op))
        n=len(high)
        if decoded.shortage and (not n or set(order[:n])!=high or
            sum(diagnosis.phi[op] for op in order[:n])/decoded.shortage<.8 or
            sum(diagnosis.phi[op] for op in order[:n-1])/decoded.shortage>=.8):
            raise AssertionError("H is not the frozen minimal rho-prefix")
    finally:
        engine.diagnostics,engine.unanchored_shortage_intervals,engine.phi_mass_mismatch_count=saved
    phi=diagnosis.phi
    node={"hash":genome_hash(decoded),"genome":c.r.prefix_genome(decoded),"ms":decoded.ms_map,
          "S":decoded.shortage,"Cmax":decoded.makespan,"H":frozenset(high),"phi":phi,
          "stage_mass":distribution(phi,lambda op:op[1]+1),
          "machine_mass":distribution(phi,lambda op:decoded.ms_map[op]),
          "job_mass":distribution(phi,lambda op:op[0]),
          "schedule":decoded.schedule,"provenance_hash":c.signature(decoded.provenance),
          "diagnosis_hash":c.signature(diagnosis)}
    counts["diagnostic_seconds"]+=time.perf_counter()-started
    return decoded,node,diagnosis


def drift(seed,node):
    high=seed["H"];phi=node["phi"];total=sum(phi.values())
    positive0={op for op,v in seed["phi"].items() if v>0}
    positive={op for op,v in phi.items() if v>0}
    result={**overlap(high,node["H"]),"outside_H0_phi_mass":sum(v for op,v in phi.items() if op not in high)/total if total else 0,
            "new_positive_operations":len(positive-positive0),
            "positive_operation_jaccard":len(positive&positive0)/len(positive|positive0) if positive|positive0 else 1.0}
    for kind in ("stage","machine","job"):
        a,b=seed[kind+"_mass"],node[kind+"_mass"]
        result[kind+"_TV"]=tv(a,b)
        result[kind+"_dominant_changed"]=dominant(a)!=dominant(b)
        result[kind+"_new_mass"]=sum(v for k,v in b.items() if a.get(k,0)==0)
    return result


def archive_nd(node,archive):
    return not any(p["shortage"]<=node["S"] and p["Cmax"]<=node["Cmax"] and
                   (p["shortage"]<node["S"] or p["Cmax"]<node["Cmax"]) for p in archive)


def dominates(a,b):
    return a["S"]<=b["S"] and a["Cmax"]<=b["Cmax"] and (a["S"]<b["S"] or a["Cmax"]<b["Cmax"])


def coverage_exposure(engine,parent,seed,current,kind):
    ind=c.r.prefix_individual(parent["genome"])
    _,windows,ms=c.disva.atomic_space(engine,ind,set(ind.os_seq))
    candidates=[op for op in (windows if kind=="OS" else ms) if op not in seed["H"]]
    r0=Region(engine,seed["H"],seed,parent)
    rt=Region(engine,current["H"],seed,parent,include_seed_machine=False)
    values={key:0.0 for key in ("dynamic","structural0","combined","narrow0","narrow_current")}
    for op in candidates:
        a,b=r0.relation(op),rt.relation(op)
        values["dynamic"]+=op in current["H"]
        values["structural0"]+=a["structural"]
        values["combined"]+=op in current["H"] or b["structural"]
        values["narrow0"]+=a["narrow"]
        values["narrow_current"]+=op in current["H"] or b["narrow"]
    return {key:value/len(candidates) if candidates else 0 for key,value in values.items()}


def accepted_record(engine,state,seed,parent,child,step,accepted_index):
    op=tuple(step["operation"])
    r0=Region(engine,seed["H"],seed,parent).relation(op)
    rt=Region(engine,parent["H"],seed,parent,include_seed_machine=False).relation(op)
    outside=op not in seed["H"]
    phi=seed["phi"].get(op,0)
    allops=list(seed["ms"])
    rank=1+sum(seed["phi"].get(other,0)>phi for other in allops)
    stable=sorted(allops,key=lambda key:(-seed["phi"].get(key,0),key)).index(op)+1
    base={k:state[k] for k in ("instance","seed","checkpoint")}
    row={**base,"state_id":f"{state['instance']}/seed_{state['seed']}/cp_{state['checkpoint']}",
         "trial":step["trial"],"policy":step["policy"],"step":step["step"],"accepted_index":accepted_index,
         "operator":step["move_type"],"operation":list(op),"production_stage":op[1]+1,
         "parent_hash":parent["hash"],"child_hash":child["hash"],
         "S_before":parent["S"],"S_after":child["S"],"incremental_gain":parent["S"]-child["S"],
         "Cmax_delta":child["Cmax"]-parent["Cmax"],"Cmax_seed_delta":child["Cmax"]-seed["Cmax"],
         "dominates_incumbent":dominates(child,parent),"dominates_seed":dominates(child,seed),
         "snapshot_archive_nondominated":archive_nd(child,state["archive"]),
         "in_H0":not outside,"in_Hpre":op in parent["H"],"initial_phi":phi,"phi_rank":rank,
         "phi_rank_percentile":(rank-1)/(len(allops)-1) if len(allops)>1 else 0,
         "phi_stable_order_rank":stable,"quadrant":quadrant(op,seed["H"],parent["H"]),
         "location_H0":location(op,seed["H"],r0),
         **{key:value for key,value in r0.items()},
         "dynamic":op in parent["H"],"structural0":r0["structural"],"narrow0":r0["narrow"],
         "combined":op in parent["H"] or rt["structural"],
         "narrow_current":op in parent["H"] or rt["narrow"],
         "structural_Hpre":rt["structural"],
         "residual":op not in parent["H"] and not rt["structural"],
         "explanation":("H0" if not outside else "new_Hpre" if op in parent["H"] else
                        "structural_Hpre" if rt["structural"] else "remote_Hpre")}
    exposure=coverage_exposure(engine,parent,seed,parent,step["move_type"]) if outside else {}
    row.update({"exposure_"+key:value for key,value in exposure.items()})
    return row


def episode_metrics(state,policy,trial,records,post,summary):
    result={k:state[k] for k in ("instance","seed","checkpoint")}
    result.update(policy=policy,trial=trial,FinalGain=summary["FinalGain"],
                  accepted_count=len(records),accepted_gain=sum(row["incremental_gain"] for row in records))
    for region in REGIONS:
        rows=[]
        for row in records:
            if region=="inside_H0": included=row["in_H0"]
            elif region=="outside_H0": included=not row["in_H0"]
            elif region in ("A","B","C","D"): included=row["quadrant"]==region
            elif region in ("tech1","tech2","machine","remote"): included=row["location_H0"]==region
            else: included=not row["in_H0"] and row[region]
            if included: rows.append(row)
        result[region+"_count"]=len(rows)
        result[region+"_gain"]=sum(row["incremental_gain"] for row in rows)
        result[region+"_Cmax_delta"]=sum(row["Cmax_delta"] for row in rows)
        result[region+"_dominates_count"]=sum(row["dominates_incumbent"] for row in rows)
        result[region+"_dominates_seed_count"]=sum(row["dominates_seed"] for row in rows)
        result[region+"_archive_nd_count"]=sum(row["snapshot_archive_nondominated"] for row in rows)
    for key in ("dynamic","structural0","combined","narrow0","narrow_current"):
        result[key+"_exposure_gain"]=sum(row["incremental_gain"]*row.get("exposure_"+key,0) for row in records if not row["in_H0"])
    result.update(first_jaccard=post[0]["jaccard"] if post else None,
                  last_jaccard=post[-1]["jaccard"] if post else None,
                  first_outside_phi=post[0]["outside_H0_phi_mass"] if post else None,
                  last_outside_phi=post[-1]["outside_H0_phi_mass"] if post else None)
    if not math.isclose(result["accepted_gain"],result["FinalGain"],abs_tol=1e-9):
        raise AssertionError("Accepted incremental gains do not telescope to saved FinalGain")
    return result


def audit_state(engine,state,source,fingerprint):
    path=ROOT/"cache/states"/state["instance"]/f"seed_{state['seed']}"/f"cp_{state['checkpoint']}.pkl.gz"
    if path.exists():
        value=c.read_pickle(path)
        if value["fingerprint"]!=fingerprint or not value["isolated"]:
            raise AssertionError("Stale audit cache")
        return value
    before=c.local.formal_state_signature(engine)
    counts=Counter()
    decoded,seed,diagnosis=diagnose_node(engine,state["parent"],counts)
    for field in ("makespan","shortage","schedule","buffer_trace","stats","provenance"):
        if getattr(decoded,field)!=getattr(state["parent"],field):
            raise AssertionError("Saved seed reconstruction differs: "+field)
    if diagnosis!=state["diagnosis"] or seed["H"]!=state["H"]:
        raise AssertionError("Seed phi/H differs from frozen CRNA")
    counts["seed_reconstruction_decodes"]+=1
    nodes={seed["hash"]:seed}
    groups=defaultdict(list)
    for row in source["steps"]:
        if row["policy"] in POLICIES: groups[(row["trial"],row["policy"])].append(row)
    summaries={(s["trial"],s["policy"]):s for s in source["episodes"]}
    accepted,drifts,episodes,wins=[],[],[],[]
    for (trial,policy),steps in sorted(groups.items()):
        current=seed;records=[];post=[]
        base={k:state[k] for k in ("instance","seed","checkpoint")}
        drifts.append({**base,"trial":trial,"policy":policy,"accepted_index":0,"step":0,
                       "node_hash":seed["hash"],**drift(seed,seed)})
        for step in steps:
            if step["proposal_parent_hash"]!=current["hash"]:
                raise AssertionError("Saved sequential parent mismatch")
            strict=step["S_proposal"]<current["S"]
            if bool(step["accepted"])!=strict:
                raise AssertionError("Saved acceptance is not strict shortage decrease")
            if not strict:
                if step["incumbent_hash"]!=current["hash"]:
                    raise AssertionError("Rejection altered saved incumbent")
                continue
            proposal=c.r.prefix_individual(step["proposal_genome"])
            key=genome_hash(proposal)
            if key!=step["proposal_hash"] or step["incumbent_hash"]!=key:
                raise AssertionError("Saved proposal genotype/hash mismatch")
            parent_ind=c.r.prefix_individual(current["genome"])
            c.disva.assert_restricted(parent_ind,proposal,seed["H"] if policy==POLICIES[0] else set(seed["ms"]),step)
            if key not in nodes:
                _,nodes[key],_=diagnose_node(engine,proposal,counts)
            child=nodes[key]
            if (child["S"],child["Cmax"])!=(step["S_proposal"],step["Cmax_proposal"]):
                raise AssertionError("Accepted child objectives differ from CRNA")
            record=accepted_record(engine,state,seed,current,child,step,len(records)+1)
            records.append(record);current=child
            post.append({**base,"trial":trial,"policy":policy,"accepted_index":len(records),
                         "step":step["step"],"node_hash":key,**drift(seed,child)})
        summary=summaries[(trial,policy)]
        if current["S"]!=summary["S_final"] or current["Cmax"]!=summary["Cmax_final"] or len(records)!=summary["accepted_moves"]:
            raise AssertionError("Saved final chain differs")
        accepted.extend(records);drifts.extend(post)
        episodes.append(episode_metrics(state,policy,trial,records,post,summary))
        if policy=="Global-Sequential":
            h=summaries[(trial,"H-Sequential")]["FinalGain"];g=summary["FinalGain"]
            first=next((row for row in records if seed["S"]-row["S_after"]>h),None) if g>h else None
            wins.append({**base,"trial":trial,"outcome":"Global-win" if g>h else "H-win" if h>g else "tie",
                         "Global_minus_H":g-h,"H_FinalGain":h,"Global_FinalGain":g,
                         **({"crossing_step":first["step"],"operator":first["operator"],"operation":first["operation"],
                             "quadrant":first["quadrant"],"explanation":first["explanation"],
                             "structural0":first["structural0"],"combined":first["combined"],
                             "in_H0":first["in_H0"],"in_Hpre":first["in_Hpre"],
                             "crossing_incremental_gain":first["incremental_gain"],
                             "Cmax_delta":first["Cmax_delta"],
                             "production_stage":first["production_stage"]} if first else {})})
    if c.local.formal_state_signature(engine)!=before:
        raise AssertionError("Audit mutated formal FE/RNG/populations/archive/counters")
    value={"fingerprint":fingerprint,"isolated":True,"counts":dict(counts),"accepted":accepted,
           "drifts":drifts,"episodes":episodes,"wins":wins,"nodes":list(nodes.values())}
    c.write_pickle(path,value)
    return value


def run_task(instance,seed,fingerprint,payload):
    verify_sources(payload,all_artifacts=False)
    replay=c.read_pickle(c.replay_path(instance,seed))
    if replay["fingerprint"]!=payload["CRNA_fingerprint"]:
        raise AssertionError("Wrong CRNA capture")
    data=json.loads((c.r.INSTANCE_DIR/f"{instance}.json").read_text())
    engine=c.variants.CorePSGateAlgorithm(data["operations"],data["buffers"],core_ps_enabled=True,
                                         **c.gate.settings().algorithm_parameters(seed))
    total=Counter()
    for state in replay["states"]:
        source=c.read_pickle(c.ROOT/"cache/episodes"/instance/f"seed_{seed}"/f"checkpoint_{state['checkpoint']}.pkl.gz")
        if source["fingerprint"]!=payload["CRNA_fingerprint"]:
            raise AssertionError("Wrong CRNA proposal source")
        result=audit_state(engine,state,source,fingerprint)
        total.update(result["counts"])
    print(f"H-drift audit: {instance} seed={seed} decodes={total['offline_decodes']} diagnoses={total['diagnosis_computations']}",flush=True)
    return dict(total)


def mean(values):
    valid=[value for value in values if value is not None]
    return statistics.mean(valid) if valid else None


def group_mean(rows,keys,fields):
    groups=defaultdict(list)
    for row in rows: groups[tuple(row[key] for key in keys)].append(row)
    return [{**dict(zip(keys,key)),**{field:mean([r[field] for r in values]) for field in fields},
             "units":len(values)} for key,values in sorted(groups.items())]


def hierarchical(episodes):
    fields=[key for key,value in episodes[0].items() if key not in ("instance","seed","checkpoint","trial","policy")]
    states=group_mean(episodes,("instance","seed","checkpoint","policy"),fields)
    seeds=group_mean(states,("instance","seed","policy"),fields)
    instances=group_mean(seeds,("instance","policy"),fields)
    return states,seeds,instances


def rates(row):
    row=dict(row)
    gain=row["accepted_gain"];outside=row["outside_H0_gain"];outside_count=row["outside_H0_count"]
    for region in REGIONS:
        row[region+"_gain_share"]=row[region+"_gain"]/gain if gain else None
        conditional=region not in ("inside_H0","A","B")
        row[region+"_outside_gain_share"]=row[region+"_gain"]/outside if outside and conditional else None
        row[region+"_outside_count_share"]=row[region+"_count"]/outside_count if outside_count and conditional else None
    for key in ("dynamic","structural0","combined","narrow0","narrow_current"):
        row[key+"_exposure"]=row[key+"_exposure_gain"]/outside if outside else None
        row[key+"_enrichment"]=row[key+"_outside_gain_share"]-row[key+"_exposure"] if outside else None
    row["C_of_outside_gain"]=row["C_gain"]/outside if outside else None
    row["D_of_outside_gain"]=row["D_gain"]/outside if outside else None
    return row


def evidence(values,null=0):
    xs=[x-null for x in values if x is not None]
    if not xs: return {"n":0,"stable_positive":False,"status":"no evidence"}
    m=statistics.mean(xs);med=statistics.median(xs)
    w,t,l=sum(x>0 for x in xs),sum(x==0 for x in xs),sum(x<0 for x in xs)
    loo=[statistics.mean(xs[:i]+xs[i+1:]) for i in range(len(xs))] if len(xs)>1 else []
    stable=m>0 and med>0 and w>l and bool(loo) and min(loo)>0
    return {"n":len(xs),"null":null,"mean":m+null,"median":med+null,"W":w,"T":t,"L":l,
            "min_LOO_margin":min(loo) if loo else None,"stable_positive":stable,
            "status":"stable positive" if stable else "unstable" if m>0 else "not positive"}


def classify(instances):
    rows=[r for r in instances if r["instance"] in c.gate.PRIMARY and r["policy"]=="Global-Sequential"]
    specs={"dynamic":("dynamic_outside_gain_share",.5),"structural":("structural0_outside_gain_share",.5),
           "structural_enrichment":("structural0_enrichment",0),"narrow":("narrow0_outside_gain_share",.5),
           "narrow_enrichment":("narrow0_enrichment",0),"D":("D_of_outside_gain",.5),
           "global_advantage":("Global_minus_H",0)}
    ev={key:evidence([row[field] for row in rows],null) for key,(field,null) in specs.items()}
    dyn=ev["dynamic"]["stable_positive"]
    structural=all(ev[k]["stable_positive"] for k in ("structural","structural_enrichment","narrow","narrow_enrichment"))
    global_chain=ev["D"]["stable_positive"] and not structural and ev["global_advantage"]["stable_positive"]
    if not rows or not any(row["outside_H0_gain"]>0 for row in rows): label="INSUFFICIENT_EVIDENCE"
    elif sum((dyn,structural,global_chain))>1: label="MIXED_MECHANISM"
    elif dyn: label="DYNAMIC_H_SUPPORTED"
    elif structural: label="STRUCTURAL_RELAXATION_SUPPORTED"
    elif global_chain: label="GLOBAL_CHAINING_SUPPORTED"
    else: label="MIXED_MECHANISM"
    return label,ev


def analyze(fingerprint,payload):
    results=[]
    for name in c.gate.PRIMARY+c.gate.SECONDARY:
        for seed in c.gate.SEEDS:
            for cp in c.CHECKPOINTS:
                value=c.read_pickle(ROOT/"cache/states"/name/f"seed_{seed}"/f"cp_{cp}.pkl.gz")
                if value["fingerprint"]!=fingerprint or not value["isolated"]: raise AssertionError("Incomplete audit")
                results.append(value)
    accepted=[row for result in results for row in result["accepted"]]
    drifts=[row for result in results for row in result["drifts"]]
    episodes=[row for result in results for row in result["episodes"]]
    wins=[row for result in results for row in result["wins"]]
    outcomes={(r["instance"],r["seed"],r["checkpoint"],r["trial"]):r["outcome"] for r in wins}
    for row in accepted:
        row["paired_outcome"]=outcomes[(row["instance"],row["seed"],row["checkpoint"],row["trial"])]
    counts=Counter()
    for result in results: counts.update(result["counts"])
    states,seeds,instances=hierarchical(episodes)
    old={row["instance"]:row for row in __import__("csv").DictReader((c.ROOT/"analysis/instance_mean_summary.csv").open())}
    for row in accepted:
        row["instance_C2_negative"]=float(old[row["instance"]]["Delta_seq_region"])<0
    instances=[rates(row) for row in instances]
    for row in instances: row["Global_minus_H"]=-float(old[row["instance"]]["Delta_seq_region"])
    label,ev=classify(instances)
    global_rows=[row for row in accepted if row["policy"]=="Global-Sequential"]
    summary={"fingerprint":fingerprint,"classification":label,"formal_FE":0,"replay_FE":0,
             "states":len(results),"primary_states":sum(r["episodes"][0]["instance"] in c.gate.PRIMARY for r in results),
             "episodes":len(episodes),"accepted_observations":len(accepted),"counts":dict(counts),
             "instance_evidence":ev,"instance_results":instances,
             "sources":{p:payload["files"][p] for p in FROZEN},"CRNA_unchanged":True,
             "scope":"Observed and mechanism-consistent evidence only; no causal proof or new search-policy efficacy claim"}
    c.csv(ROOT/"analysis/accepted_move_localization.csv",accepted)
    c.csv(ROOT/"raw/episode_metrics.csv",episodes)
    c.csv(ROOT/"raw/step_H_diagnosis.csv",drifts)
    node_rows=[]
    for result in results:
        origin=result["episodes"][0]
        for node in result["nodes"]:
            node_rows.append({**{k:origin[k] for k in ("instance","seed","checkpoint")},
                              **{k:v for k,v in node.items() if k not in ("ms","phi","H")},
                              "H":[list(op) for op in sorted(node["H"])],
                              "phi":[[list(op),v] for op,v in sorted(node["phi"].items())]})
    c.csv(ROOT/"raw/unique_diagnosis_nodes.csv",node_rows)
    c.csv(ROOT/"analysis/global_win_episode_analysis.csv",wins)
    c.csv(ROOT/"analysis/instance_mean_summary.csv",instances)
    c.csv(ROOT/"analysis/instance_seed_summary.csv",seeds)
    c.csv(ROOT/"analysis/state_summary.csv",states)
    # Operator/cohort localization keeps zero contributions from all original
    # episodes; instance ratios condition on the corresponding outside gain.
    moves_by_episode=defaultdict(list)
    for row in accepted:
        moves_by_episode[(row["instance"],row["seed"],row["checkpoint"],row["trial"],row["policy"])].append(row)
    for dimension,levels in (("operator",("OS","MS")),("cohort",("Global-win","H-win","tie"))):
        stratified=[]
        for ep in episodes:
            key=(ep["instance"],ep["seed"],ep["checkpoint"],ep["trial"],ep["policy"])
            for level in levels:
                selected=[row for row in moves_by_episode[key] if row["operator"]==level] if dimension=="operator" else (
                    moves_by_episode[key] if outcomes[key[:4]]==level else [])
                item=episode_metrics(ep,ep["policy"],ep["trial"],selected,[],{"FinalGain":sum(r["incremental_gain"] for r in selected)})
                item[dimension]=level
                stratified.append(item)
        fields=[k for k in stratified[0] if k not in ("instance","seed","checkpoint","trial","policy",dimension)]
        st=group_mean(stratified,("instance","seed","checkpoint","policy",dimension),fields)
        sd=group_mean(st,("instance","seed","policy",dimension),fields)
        ins=[rates(row) for row in group_mean(sd,("instance","policy",dimension),fields)]
        c.csv(ROOT/f"analysis/{dimension}_localization_summary.csv",ins)
    paired=[]
    for row in wins:
        item={k:row[k] for k in ("instance","seed","checkpoint","trial")}
        item.update(global_win=int(row["outcome"]=="Global-win"),h_win=int(row["outcome"]=="H-win"),tie=int(row["outcome"]=="tie"))
        for category in ("H0","new_Hpre","structural_Hpre","remote_Hpre"):
            item[category]=int(row.get("explanation")==category)
        paired.append(item)
    fields=[k for k in paired[0] if k not in ("instance","seed","checkpoint","trial")]
    st=group_mean(paired,("instance","seed","checkpoint"),fields)
    sd=group_mean(st,("instance","seed"),fields)
    pi=group_mean(sd,("instance",),fields)
    for row in pi:
        for category in ("H0","new_Hpre","structural_Hpre","remote_Hpre"):
            row[category+"_crossing_fraction"]=row[category]/row["global_win"] if row["global_win"] else None
    c.csv(ROOT/"analysis/paired_crossing_instance_summary.csv",pi)
    # Primary summaries always average the already hierarchical instance rows.
    numeric=[key for key,value in instances[0].items() if isinstance(value,(int,float)) or value is None]
    breakdown=[]
    for scope,names in (("primary_unseen",c.gate.PRIMARY),("secondary_seen",c.gate.SECONDARY)):
        for dimension in ("all","size","profile"):
            levels=("all",) if dimension=="all" else sorted({name.split("_")[1 if dimension=="size" else 2] for name in names})
            for level in levels:
                for policy in POLICIES:
                    rows=[r for r in instances if r["instance"] in names and r["policy"]==policy and
                          (dimension=="all" or r["instance"].split("_")[1 if dimension=="size" else 2]==level)]
                    breakdown.append({"set":scope,"dimension":dimension,"level":level,"policy":policy,
                                      "n_instances":len(rows),**{key:mean([r[key] for r in rows]) for key in numeric}})
    checkpoints=[];checkpoint_instances=[]
    for scope,names in (("primary_unseen",c.gate.PRIMARY),("secondary_seen",c.gate.SECONDARY)):
        for cp in c.CHECKPOINTS:
            cp_seed=[row for row in states if row["checkpoint"]==cp]
            fields=[key for key in cp_seed[0] if key not in ("instance","seed","checkpoint","policy","units")]
            cp_inst=[rates(row) for row in group_mean(cp_seed,("instance","policy"),fields)]
            if scope=="primary_unseen":
                for row in cp_inst:
                    row["checkpoint"]=cp
                checkpoint_instances.extend(cp_inst)
            for policy in POLICIES:
                rows=[r for r in cp_inst if r["instance"] in names and r["policy"]==policy]
                checkpoints.append({"set":scope,"checkpoint":cp,"policy":policy,"n_instances":len(rows),
                                    **{key:mean([r[key] for r in rows]) for key in numeric if key not in ("units","Global_minus_H")}})
    c.csv(ROOT/"analysis/size_profile_summary.csv",breakdown)
    c.csv(ROOT/"analysis/checkpoint_summary.csv",checkpoints)
    c.csv(ROOT/"analysis/checkpoint_instance_summary.csv",checkpoint_instances)
    c.csv(ROOT/"analysis/structural_coverage.csv",[{k:r[k] for k in ("instance","policy","outside_H0_gain","outside_H0_count",
          *[f"{name}_{suffix}" for name in ("dynamic","structural0","combined","narrow0","narrow_current")
            for suffix in ("outside_gain_share","outside_count_share","exposure","enrichment")])} for r in instances])
    c.csv(ROOT/"analysis/dynamic_h_move_quadrants.csv",[{k:r[k] for k in ("instance","policy",*[f"{q}_{s}" for q in "ABCD" for s in ("count","gain","gain_share","outside_gain_share")])} for r in instances])
    drift_fields=[k for k,v in drifts[0].items() if isinstance(v,(float,int)) and k not in ("seed","checkpoint","trial","step","accepted_index")]
    d_state=group_mean(drifts,("instance","seed","checkpoint","policy","accepted_index"),drift_fields)
    d_seed=group_mean(d_state,("instance","seed","policy","accepted_index"),drift_fields)
    d_inst=group_mean(d_seed,("instance","policy","accepted_index"),drift_fields)
    c.csv(ROOT/"analysis/h_drift_summary.csv",d_inst)
    side=[]
    for scope,names in (("primary_unseen",c.gate.PRIMARY),("secondary_seen",c.gate.SECONDARY)):
        for policy in POLICIES:
            for region in REGIONS:
                rows=[r for r in instances if r["instance"] in names and r["policy"]==policy]
                n=mean([r[region+"_count"] for r in rows])
                side.append({"set":scope,"policy":policy,"region":region,
                             "mean_count_per_episode":n,"mean_incremental_gain_per_episode":mean([r[region+"_gain"] for r in rows]),
                             "mean_Cmax_delta_per_episode":mean([r[region+"_Cmax_delta"] for r in rows]),
                             **{key:mean([r[region+suffix] for r in rows])/n if n else None for key,suffix in
                                (("dominates_incumbent_fraction","_dominates_count"),("dominates_seed_fraction","_dominates_seed_count"),("archive_nd_fraction","_archive_nd_count"))}})
    c.csv(ROOT/"analysis/archive_side_effect_summary.csv",side)
    loo=[]
    fields=("dynamic_outside_gain_share","structural0_enrichment","narrow0_enrichment","D_of_outside_gain","Global_minus_H")
    primary=[r for r in instances if r["instance"] in c.gate.PRIMARY and r["policy"]=="Global-Sequential"]
    for name in c.gate.PRIMARY:
        rows=[r for r in primary if r["instance"]!=name]
        loo.append({"omitted":name,**{key:mean([r[key] for r in rows]) for key in fields}})
    c.csv(ROOT/"analysis/leave_one_instance_out.csv",loo)
    evidence_groups=[]
    specs=(("dynamic_outside_gain_share",.5),("structural0_outside_gain_share",.5),
           ("structural0_enrichment",0),("narrow0_enrichment",0),("D_of_outside_gain",.5),
           ("Global_minus_H",0))
    for dimension in ("size","profile","checkpoint"):
        candidates=primary if dimension!="checkpoint" else [r for r in checkpoint_instances if r["instance"] in c.gate.PRIMARY and r["policy"]=="Global-Sequential"]
        levels=sorted({r["checkpoint"] if dimension=="checkpoint" else r["instance"].split("_")[1 if dimension=="size" else 2] for r in candidates},key=str)
        for level in levels:
            rows=[r for r in candidates if (r["checkpoint"] if dimension=="checkpoint" else r["instance"].split("_")[1 if dimension=="size" else 2])==level]
            for field,null in specs:
                if field not in rows[0]:continue
                evidence_groups.append({"dimension":dimension,"level":level,"metric":field,**evidence([r[field] for r in rows],null)})
    c.csv(ROOT/"analysis/group_evidence.csv",evidence_groups)
    c.csv(ROOT/"analysis/negative_C2_crossing_summary.csv",[r for r in pi if r["instance"] in c.gate.PRIMARY and float(old[r["instance"]]["Delta_seq_region"])<0])
    # Descriptive pooled totals and operator/stage breakdowns never drive classification.
    pooled=[]
    for scope,names in (("primary_unseen",c.gate.PRIMARY),("secondary_seen",c.gate.SECONDARY)):
        for dimension in ("all","operator","production_stage","profile","size","checkpoint"):
            def val(row):
                return ("all" if dimension=="all" else row["instance"].split("_")[2 if dimension=="profile" else 1]
                        if dimension in ("profile","size") else row[dimension])
            rows=[row for row in global_rows if row["instance"] in names]
            for level in sorted({val(row) for row in rows},key=str):
                group=[row for row in rows if val(row)==level]
                for category in ("H0","outside","A","B","C","D","tech1","tech2","machine","remote","dynamic","structural0","combined","residual"):
                    subset=[row for row in group if (row["in_H0"] if category=="H0" else not row["in_H0"] if category=="outside" else
                            row["quadrant"]==category if category in "ABCD" else row["location_H0"]==category if category in ("tech1","tech2","machine","remote") else not row["in_H0"] and row[category])]
                    pooled.append({"set":scope,"dimension":dimension,"level":level,"category":category,
                                   "count":len(subset),"incremental_gain":sum(r["incremental_gain"] for r in subset),
                                   "Cmax_delta":sum(r["Cmax_delta"] for r in subset),"instance_coverage":len({r["instance"] for r in subset})})
    c.csv(ROOT/"analysis/descriptive_move_totals.csv",pooled)
    summary["pooled_totals"]=[row for row in pooled if row["dimension"]=="all"]
    summary["group_results"]=breakdown
    summary["checkpoint_results"]=checkpoints
    verify_sources(payload)
    c.r.write_json_atomic(ROOT/"analysis/accepted_move_h_drift_summary.json",summary)
    report(summary)
    figures(instances,global_rows,d_inst,wins)
    return summary


def report(summary):
    lines=["# Accepted-Move Localization & H-Drift Audit", "",f"Classification: **{summary['classification']}**",
           "\n## Fidelity and scope",f"Formal FE=0; replay FE=0. States={summary['states']} (unseen={summary['primary_states']}); episodes={summary['episodes']}; accepted observations={summary['accepted_observations']}. Actual new calls: {json.dumps(summary['counts'])}.",
           "Original CRNA proposals/acceptance are read only. No new mutation, search, selection, replacement, archive update, or formal RNG draws. Four protected source hashes and every CRNA file are pinned before/after; preregistration.json contains full hashes.",
           "\n## Definitions",json.dumps(RULES,ensure_ascii=False,indent=2),
           "\n## Instance-level evidence",json.dumps(summary["instance_evidence"],ensure_ascii=False,indent=2),
           "\nAll rates below are conditional on accepted Global moves outside H0. Counts/gains average trial -> state -> seed -> instance first. Coverage/exposure ratios are computed within each instance, then instances equally weighted. Pooled accepted-move totals are descriptive, never independent benchmark samples.",
           "\n|Instance|Global-H gain|Dynamic coverage|Structural0 coverage|Structural exposure|Narrow enrichment|D outside share|\n|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["instance_results"]:
        if row["policy"]=="Global-Sequential":
            fields=("Global_minus_H","dynamic_outside_gain_share","structural0_outside_gain_share","structural0_exposure","narrow0_enrichment","D_of_outside_gain")
            lines.append("|"+row["instance"]+"|"+"|".join(f"{row[key]:.6f}" if row[key] is not None else "NA" for key in fields)+"|")
    lines.extend(["\n## Q1 / Q2 / Q3", "Classification is an explanatory audit, not a measured Dynamic-H/expanded-H policy improvement. Review the measured instance table, drift, structural exposure and Global-win crossing moves together; overlap with eligible machines alone is not selective evidence.",
                  "\n## A. What is now established", "Only observed original accepted moves and re-diagnosed incumbents are established. Hpre membership is measured before the action, and never feeds search.",
                  "\n## B. What remains unresolved", "Whether a different proposal distribution, changing H, or structural relaxation improves a formal algorithm remains untested. Observational coverage is not causal proof.",
                  "\n## C. Recommended next experiment", {"DYNAMIC_H_SUPPORTED":"Test Dynamic-H Sequential only, without claiming integration efficacy.","STRUCTURAL_RELAXATION_SUPPORTED":"Test H + structural-boundary Sequential only; do not implement within this audit.","GLOBAL_CHAINING_SUPPORTED":"Test unrestricted Global Sequential only; no forced H-based continuation."}.get(summary["classification"],"Do not design a new mechanism yet; heterogeneous/unstable explanation does not justify choosing an H-based search boundary."),
                  "\n## D. What should NOT be implemented yet", "No LNS/ALNS/CP/MILP/destroy-repair, adaptive H, structural expanded H, global fallback or adaptive trigger. No formal source changes, rho/p_mu tuning, extra proposal budget, or selective deletion of instances."])
    (ROOT/"analysis/accepted_move_h_drift_interpretation.md").write_text("\n".join(lines)+"\n")


def figures(instances,moves,drifts,wins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=ROOT/"figures";out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in instances if r["instance"] in c.gate.PRIMARY and r["policy"]=="Global-Sequential"]
    fig,ax=plt.subplots(figsize=(11,5));bottom=[0.0]*len(rows)
    # These categories partition total Global gain, unlike overlapping coverage sets.
    for label,field in (("H0","inside_H0_gain_share"),("outside H0 / current H","C_gain_share"),("outside both H / structural current",None),("remote current","residual_gain_share")):
        values=[r[field] if field else r["combined_gain_share"]-r["C_gain_share"] for r in rows]
        ax.bar(range(len(rows)),values,bottom=bottom,label=label);bottom=[a+b for a,b in zip(bottom,values)]
    ax.set_xticks(range(len(rows)),[r["instance"].replace("WIPHFSP_","") for r in rows],rotation=45,ha="right")
    ax.set_ylabel("Within-instance accepted gain share");ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"global_gain_localization.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for policy in POLICIES:
        rr=[r for r in drifts if r["instance"] in c.gate.PRIMARY and r["policy"]==policy]
        indices=sorted({r["accepted_index"] for r in rr})
        ax.plot(indices,[mean([r["jaccard"] for r in rr if r["accepted_index"]==i]) for i in indices],marker="o",label=policy)
    ax.set_xlabel("Accepted index (conditional survivors)");ax.set_ylabel("Instance-mean H0/Ht Jaccard");ax.legend()
    fig.tight_layout();fig.savefig(out/"h_jaccard.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for field in ("dynamic_outside_gain_share","structural0_outside_gain_share","structural0_exposure"):
        vals=[mean([r[field] for r in rows if f"_{p}_" in r["instance"]]) for p in ("P2","P3")]
        ax.plot((0,1),vals,marker="o",label=field.replace("_outside_gain_share",""))
    ax.set_xticks((0,1),("P2","P3"));ax.set_ylabel("Instance-mean coverage / exposure");ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/"p2_p3_coverage.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));labels=("H0","new_Hpre","structural_Hpre","remote_Hpre")
    proportions=[]
    for name in c.gate.PRIMARY:
        rr=[r for r in wins if r["instance"]==name and r["outcome"]=="Global-win"]
        # Aggregate crossing counts through the same trial/state/seed hierarchy.
        for label in labels:
            all_ep=[{**r,"cross":int(r.get("explanation")==label),"global_win":int(r["outcome"]=="Global-win")} for r in wins if r["instance"]==name]
            st=group_mean(all_ep,("instance","seed","checkpoint"),("cross","global_win"))
            sd=group_mean(st,("instance","seed"),("cross","global_win"))
            a,b=mean([r["cross"] for r in sd]),mean([r["global_win"] for r in sd])
            proportions.append((label,a/b if b else None))
    ax.bar(labels,[mean([v for k,v in proportions if k==label]) for label in labels])
    ax.set_ylabel("Instance-mean share of Global-win crossing moves");ax.tick_params(axis="x",rotation=15)
    fig.tight_layout();fig.savefig(out/"global_win_crossing.png",dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("all","run","analyze"),default="all")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:parser.error("workers must be positive")
    fingerprint,payload=register()
    if args.mode in ("all","run"):
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context("spawn")) as pool:
            futures=[pool.submit(run_task,name,seed,fingerprint,payload) for name in c.gate.PRIMARY+c.gate.SECONDARY for seed in c.gate.SEEDS]
            for future in as_completed(futures):future.result()
        verify_sources(payload)
    if args.mode in ("all","analyze"):
        result=analyze(fingerprint,payload)
        print(json.dumps({k:result[k] for k in ("classification","formal_FE","replay_FE","states","counts")}),flush=True)


if __name__=="__main__":main()
