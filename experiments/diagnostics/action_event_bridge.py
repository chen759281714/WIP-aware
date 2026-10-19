"""Independent MS action -> residual event shock -> Lambda audit.

Run: PYTHONHASHSEED=0 python3 -m experiments.diagnostics.action_event_bridge
No formal search, operator, parameter, decoder or sensitivity source changes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from experiments import run_pre_ablation as r
from experiments import sis_active_lever as active
from experiments import sis_local_viability as local
from experiments import sis_role_direction as role
from experiments import sis_selection_aware as selection
from experiments.diagnostics import shortage_sensitivity as sensitivity
from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation

ROOT = r.PROJECT_ROOT / "experiments/results/action_event_bridge"
FAMILIES = ("ECT-AMR", "RandomMS", "FastestProcessingMS", "BasicMutation-MS")
RULES = {
    "families": FAMILIES, "time_tolerance": sensitivity.TOL,
    "coverage": "Validated Lambda path weighted mass: High >=.90, Medium >=.50 and <.90, Low <.50; zero weighted mass separate",
    "smooth": "Use prior invalid_path() labels unchanged. Local-only smooth fractions are secondary.",
    "primary_unit": "Unique (instance,seed,FE,child genotype); preserve overlapping family labels for family summaries",
    "gate_A": "High coverage actual-nonzero sign >.5 and Spearman >0 in both instances and >=4/6 groups; unavailable/constant signal is NOT TESTED",
    "gate_B": "Median Lambda-weighted top5 mass >=.80 for both all-MS and High subsets, in both instances and >=4/6 groups; no child benefit filtering",
    "gate_C": "Boundary nonzero sign and rank retain >=.80 of full-smooth oracle information; strictly exceed target-only and direct-g on sign OR within-target rank in both instances and >=4/6 groups",
    "boundary": "Target S; target C iff delta_p!=0; seed old immediate successor S; child new immediate successor S; at most one first directly affected factual release-regime boundary",
    "gate_D": "Audit frozen ECT capabilities before predictor. Unsupported release/regime propagation cannot be supplied by a new heuristic. Stop when a complete shock predictor cannot be established from allowed seed-only information.",
    "gate_E": "Only after D; both instance directions nonnegative and LambdaBridge beats ECT-only on within-target rank OR top1 improvement with >=4/6 group consistency",
    "future_target_sampling": "4 diagnostic-random targets/state with >=2 alternatives; enumerate all legal alternatives; no Lambda/ECT filtering",
    "locality": "All nonzero residuals counted, top1/3/5/10 raw and Lambda-weighted; location labels use structural adjacency only",
    "calibration": "Strong abs(prediction)>=1% S_seed, mild between tolerance and 1%, near-zero <=tolerance",
    "residual_identity": "Full residual oracle equals full-time direct algebraically; this equality does NOT establish predictive information",
    "no_outcome_roots": True, "formal_FE": 0,
}


def source_certificate():
    old = json.loads((sensitivity.ROOT / "preregistration.json").read_text())
    summary = json.loads((sensitivity.ROOT / "analysis/sensitivity_validation_summary.json").read_text())
    old_fingerprint = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    if old_fingerprint != summary["fingerprint"] or summary["stages"]["direct"] != "PASS" or summary["stages"]["graph"] != "PASS":
        raise AssertionError("Previous direct/graph certificate mismatch")
    if summary["propagation_only"]["cases"] != 504 or summary["propagation_only"]["nonzero_FD"] != 504:
        raise AssertionError("Must reproduce 504 propagation-only cases")
    for relative, expected in {**old["files"], **old["formal_source"]["checked_sources"],
                               **old["formal_source"]["capture_sha256"]}.items():
        if role.file_hash(r.PROJECT_ROOT / relative) != expected:
            raise AssertionError(f"Frozen input changed: {relative}")
    with (sensitivity.ROOT / "raw/perturbation_cases.csv").open() as f:
        fd = list(csv.DictReader(f))
    if len(fd) != 1920 or max(float(z["forward_reverse_error"]) for z in fd) > 1e-9:
        raise AssertionError("Forward/reverse certificate failed")
    preserved_prop = [z for z in fd if z["topology_preserved"] == "True" and z["subgroup"] == "propagation_only"]
    if len(preserved_prop) != 504:
        raise AssertionError("Raw and summary propagation-only certificate differ")
    caches = sorted((sensitivity.ROOT / "cache").glob("*/seed*/fe_*.json"))
    child_caches = sorted((sensitivity.ROOT / "cache/child_batches").glob("*/seed_*.json"))
    if len(caches) != 240 or len(child_caches) != 6:
        raise AssertionError("Incomplete historical sensitivity caches")
    pinned = {str(p.relative_to(r.PROJECT_ROOT)): role.file_hash(p) for p in
              [*caches, *child_caches, sensitivity.ROOT / "raw/perturbation_cases.csv",
               sensitivity.ROOT / "analysis/sensitivity_validation_summary.json", Path(__file__)]}
    payload = {"sensitivity_fingerprint": old_fingerprint, "sensitivity_source": old,
               "pinned": pinned, "rules": RULES}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = ROOT / "preregistration.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise AssertionError("Refuse post-hoc preregistration change")
    r.write_json_atomic(path, payload)
    return fingerprint, summary, fd


def residual(graph, shifts):
    incoming = defaultdict(list)
    for edge in graph["edges"]:
        if edge["alpha"]:
            incoming[edge["dst"]].append(edge)
    return {node: shifts[node] - sum(e["alpha"] * shifts[e["src"]] for e in incoming[node])
            for node in graph["order"]}


def shares(values):
    ordered = sorted(values, reverse=True)
    total = sum(ordered)
    return {f"top{k}_share": sum(ordered[:k])/total if total else None for k in (1, 3, 5, 10)}


def edge_diff(seed_graph, child_graph, machine_only=False):
    def signatures(graph):
        uid = {n["id"]: tuple(n["uid"]) for n in graph["nodes"]}
        return {(uid[e["src"]], uid[e["dst"]], e["kind"], e["alpha"])
                for e in graph["edges"] if not machine_only or "machine" in e["kind"]}
    before, after = signatures(seed_graph), signatures(child_graph)
    return {"deleted": sorted(before-after), "added": sorted(after-before)}


def neighbors(sequence, target):
    keys = [tuple(k) for k in sequence]
    if target not in keys:
        return None, None
    index = keys.index(target)
    return keys[index-1] if index else None, keys[index+1] if index+1 < len(keys) else None


def boundary_roots(seed_graph, child_graph, target, old_machine, new_machine, delta_p):
    """Pure topology/candidate rule: no event deltas or shortage outcomes."""
    nodes = {tuple(n["uid"]): n["id"] for n in seed_graph["nodes"]}
    child_uid = {n["id"]: tuple(n["uid"]) for n in child_graph["nodes"]}
    seed_uid = {n["id"]: tuple(n["uid"]) for n in seed_graph["nodes"]}
    _, old_successor = neighbors(seed_graph["machine_sequences"][old_machine], target)
    _, new_successor = neighbors(child_graph["machine_sequences"].get(new_machine, []), target)
    roots = defaultdict(list)
    roots[nodes[(*target, "start")]].append("target_start")
    if delta_p:
        roots[nodes[(*target, "complete")]].append("processing_time_change")
    if old_successor:
        roots[nodes[(*old_successor, "start")]].append("old_machine_removal_successor")
    if new_successor:
        roots[nodes[(*new_successor, "start")]].append("new_machine_insertion_successor")
    direct_operations = {target, *(k for k in (old_successor, new_successor) if k is not None)}
    start_uids = {(*k, "start") for k in direct_operations}
    seed_releases = {seed_uid[int(n)]: (d["type"], seed_uid.get(d["trigger"]))
                     for n, d in seed_graph["release_types"].items()}
    possibilities = []
    node_details = {n["id"]: n for n in child_graph["nodes"]}
    for n, definition in child_graph["release_types"].items():
        child_node = int(n)
        uid = child_uid[child_node]
        trigger = child_uid.get(definition["trigger"])
        relation = definition["type"], trigger
        if seed_releases.get(uid) == relation:
            continue
        seed_trigger = seed_releases.get(uid, (None, None))[1]
        if uid[:2] in direct_operations or trigger in start_uids or seed_trigger in start_uids:
            rec = node_details[child_node]
            possibilities.append((rec["time"], rec["logical_id"], uid))
    if possibilities:
        first = min(possibilities)[2]
        roots[nodes[first]].append("first_direct_release_regime_boundary")
    return {node: reasons for node, reasons in sorted(roots.items())}


def seed_schedule(graph, ind):
    times = {tuple(n["uid"]): n["time"] for n in graph["nodes"]}
    return [{"job": key[0], "op": key[1], "machine": ind.ms_map[key],
             "start": times[(*key, "start")], "end": times[(*key, "complete")],
             "release": times[(*key, "release")]} for key in ind.os_seq]


def child_for_context(state, engine, context, basic_map):
    ind = r.prefix_individual(state["genome"])
    family = context["row"]["family"]
    if family == "BasicMutation-MS":
        child = basic_map[context["row"]["child_id"]]
    else:
        child = active.construct_child(engine, ind, "AMR", context["candidate"])
    differences = [key for key in ind.ms_map if child.ms_map[key] != ind.ms_map[key]]
    if len(differences) != 1 or child.os_seq != ind.os_seq:
        raise AssertionError("Primary child must be exactly one MS reassignment")
    if selection.digest(engine.genotype_key(child)) != context["row"]["child_id"]:
        raise AssertionError("Historical genotype reconstruction differs")
    target = differences[0]
    return child, target, ind.ms_map[target], child.ms_map[target]


def shock_statistics(graph, e):
    nonzero = {n: value for n, value in e.items() if abs(value) > sensitivity.TOL}
    path_smooth = {n: not sensitivity.invalid_path(graph, n) for n in nonzero}
    local_smooth = {n: n not in graph["nonsmooth"] for n in nonzero}
    weighted = {n: abs(graph["lambda"][n]*value) for n, value in nonzero.items()}
    weighted_total = sum(weighted.values())
    smooth_mass = sum(weighted[n] for n in nonzero if path_smooth[n])
    coverage = smooth_mass/weighted_total if weighted_total > sensitivity.TOL else None
    bin_name = "zero_weighted" if coverage is None else "High" if coverage >= .9 else "Medium" if coverage >= .5 else "Low"
    return {
        "shock_nodes_total": len(nonzero), "shock_nodes_smooth": sum(local_smooth.values()),
        "shock_nodes_nonsmooth": sum(not v for v in local_smooth.values()),
        "shock_nodes_path_valid": sum(path_smooth.values()),
        "abs_shock_mass_smooth": sum(abs(nonzero[n]) for n in nonzero if local_smooth[n]),
        "abs_shock_mass_nonsmooth": sum(abs(nonzero[n]) for n in nonzero if not local_smooth[n]),
        "shock_L1_mass": sum(abs(v) for v in nonzero.values()),
        "weighted_shock_mass": weighted_total, "path_valid_weighted_coverage": coverage,
        "local_smooth_weighted_coverage": sum(weighted[n] for n in nonzero if local_smooth[n])/weighted_total if weighted_total else None,
        "coverage_bin": bin_name,
        **shares([abs(v) for v in nonzero.values()]),
        **{"weighted_"+k: v for k, v in shares(list(weighted.values())).items()},
    }


def origin_index(graph, child_graph, target, old, new):
    uid = {n["id"]:n["uid"] for n in graph["nodes"]}
    child_uid = {n["id"]:n["uid"] for n in child_graph["nodes"]}
    def signature(g, mapping):
        grouped=defaultdict(set)
        for e in g["edges"]:
            if "job" in e["kind"] or "wip" in e["kind"]:
                grouped[tuple(mapping[e["dst"]])].add((tuple(mapping[e["src"]]),e["kind"],e["alpha"]))
        return grouped
    before,after=signature(graph,uid),signature(child_graph,child_uid)
    def releases(g,mapping):
        return {tuple(mapping[int(n)]):(d["type"],tuple(mapping[d["trigger"]]) if d["trigger"] is not None else None)
                for n,d in g["release_types"].items()}
    seed_rel,child_rel=releases(graph,uid),releases(child_graph,child_uid)
    return {"uid":uid,
            "old_near":set(k for k in neighbors(graph["machine_sequences"][old],target) if k),
            "new_near":set(k for k in neighbors(child_graph["machine_sequences"][new],target) if k),
            "job_boundary":{key for key in set(before)|set(after) if before[key]!=after[key]},
            "release_boundary":{key for key in seed_rel if seed_rel[key]!=child_rel[key]}}


def origin_tags(node, target, index):
    uid = index["uid"][node]
    key, kind = tuple(uid[:2]), uid[2]
    tags = []
    if key == target:
        tags.append("target_operation")
    if key in index["old_near"]:
        tags.append("old_machine_vicinity")
    if key in index["new_near"]:
        tags.append("new_machine_vicinity")
    if kind == "release" and tuple(uid) in index["release_boundary"]:
        tags.append("blocking_unblocking_boundary")
    if kind == "start" and tuple(uid) in index["job_boundary"]:
        tags.append("job_wip_boundary")
    return tags or ["other"]


def validate_processing_residual(fd_rows):
    validations = []
    for name in local.INSTANCES:
        data = json.loads((r.INSTANCE_DIR / f"{name}.json").read_text())
        for seed in local.SEEDS:
            cases = [z for z in fd_rows if z["instance"] == name and int(z["seed"]) == seed and z["topology_preserved"] == "True"]
            selected = min(cases, key=lambda z: (int(z["fe_index"]), z["operation"]))
            cached = json.loads(sensitivity.state_path({"instance": name, "seed": seed, "fe_index": int(selected["fe_index"])}).read_text())
            state, graph = cached["state"], sensitivity.restore_graph(cached["graph"])
            ind = r.prefix_individual(state["genome"])
            key = tuple(json.loads(selected["operation"]))
            node = {tuple(k): ids for k, ids in graph["op_nodes"]}[key]["complete"]
            seed_times = {tuple(n["uid"]): n["time"] for n in graph["nodes"]}
            for step in (1, -1):
                decoded = sensitivity.normalized_decode(sensitivity.scaled_operations(data["operations"], key, ind.ms_map[key], step), data["buffers"], ind, True)
                perturbed_graph = sensitivity.graph_from_decode(decoded, data["operations"], data["buffers"])
                if graph["topology"] != perturbed_graph["topology"]:
                    raise AssertionError("Previously preserved processing case changed topology")
                new_times = {tuple(n["uid"]): n["time"] for n in perturbed_graph["nodes"]}
                delta = {n["id"]: new_times[tuple(n["uid"])]-seed_times[tuple(n["uid"])] for n in graph["nodes"]}
                e = residual(graph, delta)
                quantum = step/sensitivity.SCALE
                error = max(abs(value-(quantum if n == node else 0)) for n, value in e.items())
                predicted = -sum(graph["lambda"][n]*value for n, value in e.items())
                actual = state["S_seed"]-decoded["shortage"]
                if error > sensitivity.TOL or abs(predicted-actual) > sensitivity.TOL:
                    raise AssertionError("Pure processing residual sanity test failed")
                validations.append({**sensitivity.identity(state), "step": quantum,
                                    "root": node, "root_residual_error": error,
                                    "valuation_error": abs(predicted-actual), "passed": True})
    return validations


def audit_task(task, fingerprint):
    data = json.loads((r.INSTANCE_DIR / f"{task.instance}.json").read_text())
    engine = WIPGraphDualPopulation(data["operations"], data["buffers"], N=3, N_A=12, FE_max=6, seed=0)
    before = local.formal_state_signature(engine)
    batch = json.loads((sensitivity.ROOT / "cache/child_batches" / task.instance / f"seed_{task.seed}.json").read_text())
    contexts = [c for c in batch["contexts"] if c["row"]["family"] in FAMILIES]
    with (selection.OUTPUT_ROOT / "raw/proposal_trials.csv").open() as f:
        basic_rows = [z for z in csv.DictReader(f) if z["instance"] == task.instance and int(z["seed"]) == task.seed and z["policy"] == "BasicMutation" and z["move_type"] == "MS"]
    grouped = defaultdict(list)
    for context in contexts:
        grouped[context["row"]["fe_index"]].append(context)
    rows, shock_rows, alignments = [], [], []
    decodes = 0
    for fe, state_contexts in sorted(grouped.items()):
        cached = json.loads(sensitivity.state_path({"instance": task.instance, "seed": task.seed, "fe_index": fe}).read_text())
        state, graph = cached["state"], sensitivity.restore_graph(cached["graph"])
        ind = r.prefix_individual(state["genome"])
        basic_map = {}
        for b in basic_rows:
            if int(b["fe_index"]) != fe:
                continue
            child = selection.exact_basic(engine, ind, selection.rng_for(state, int(b["trial_id"]), "BasicMutation"))
            basic_map[b["child_genotype_id"]] = child
        physical = {}
        uid_by_node = {n["id"]: n["uid"] for n in graph["nodes"]}
        for context in state_contexts:
            label = context["row"]
            child, target, old, new = child_for_context(state, engine, context, basic_map)
            cid = label["child_id"]
            if cid not in physical:
                decoded = sensitivity.normalized_decode(data["operations"], data["buffers"], child)
                actual = state["S_seed"]-decoded["shortage"]
                if actual != label["actual"]:
                    raise AssertionError("Historical MS child objective differs")
                cg = sensitivity.graph_from_decode(decoded, data["operations"], data["buffers"])
                child_times = {tuple(n["uid"]): n["time"] for n in cg["nodes"]}
                changes = {n["id"]: child_times[tuple(n["uid"])]-n["time"] for n in graph["nodes"]}
                if any(abs(changes[n]-float(context["changes"][str(n)])) > 1e-9 for n in changes):
                    raise AssertionError("Historical MS child timing alignment differs")
                e = residual(graph, changes)
                reconstructed = sensitivity.forward(graph, e)
                reconstruction_error = max(abs(reconstructed[n]-changes[n]) for n in changes)
                full = -sum(graph["lambda"][n]*e[n] for n in e)
                direct = -sum(graph["g"].get(n, 0)*dt for n, dt in changes.items())
                if abs(full-direct) > 1e-7 or reconstruction_error > 1e-7:
                    raise AssertionError("Residual algebraic identity/reconstruction failed")
                ids = {tuple(k): v for k, v in graph["op_nodes"]}
                delta_p = data["operations"][target[0]][target[1]]["machines"][new]-data["operations"][target[0]][target[1]]["machines"][old]
                if abs(e[ids[target]["complete"]]-delta_p) > 1e-9:
                    raise AssertionError("Machine-dependent processing shock is not delta_p")
                smooth = -sum(graph["lambda"][n]*v for n, v in e.items() if not sensitivity.invalid_path(graph, n))
                record = {**sensitivity.identity(state), "child_id": cid, "target": list(target),
                          "old_machine": old, "new_machine": new, "delta_p": delta_p,
                          "S_seed": state["S_seed"], "actual": actual,
                          "smooth_prediction": smooth, "full_residual_prediction": full,
                          "fulltime_prediction": direct, **shock_statistics(graph, e),
                          "identity_error": abs(full-direct), "reconstruction_error": reconstruction_error,
                          "topology_group_previous": label["topology_group"]}
                index=origin_index(graph,cg,target,old,new)
                for n, value in e.items():
                    if abs(value) <= sensitivity.TOL:
                        continue
                    tags = origin_tags(n,target,index)
                    shock_rows.append({**sensitivity.identity(state), "child_id": cid, "node": n,
                        "event_uid": uid_by_node[n], "delta_t": changes[n], "shock": value,
                        "g": graph["g"].get(n, 0), "lambda": graph["lambda"][n],
                        "local_smooth": n not in graph["nonsmooth"], "path_valid": not sensitivity.invalid_path(graph, n),
                        "weighted_mass": abs(graph["lambda"][n]*value), "origin_tags": tags, "primary_origin": tags[0]})
                alignments.extend({**sensitivity.identity(state), "child_id": cid, "node": n["id"],
                    "event_uid": n["uid"], "seed_time": n["time"], "child_time": child_times[tuple(n["uid"])],
                    "delta_t": changes[n["id"]]} for n in graph["nodes"] if abs(changes[n["id"]]) > sensitivity.TOL)
                physical[cid] = {"record": record, "graph": cg, "e": e}
                r.write_json_atomic(ROOT / "cache/children" / task.instance / f"seed_{task.seed}" / f"fe_{fe}" / f"{cid}.json",
                                    {"fingerprint": fingerprint, "record": record, "child_graph": cg, "residual": e})
                decodes += 1
            rows.append({**physical[cid]["record"], "family": label["family"]})
        print(f"Bridge MS fidelity: {task.instance} seed={task.seed} FE={fe}", flush=True)
    if local.formal_state_signature(engine) != before:
        raise AssertionError("Bridge diagnostics changed formal state")
    result = {"rows": rows, "shocks": shock_rows, "alignments": alignments, "decodes": decodes,
              "fingerprint": fingerprint, "formal_FE": 0, "fidelity_passed": True}
    r.write_json_atomic(ROOT / "cache/batches" / task.instance / f"seed_{task.seed}.json", result)
    return result


def unique_rows(rows):
    seen, result = set(), []
    for row in sorted(rows, key=lambda z: (z["instance"], z["seed"], z["fe_index"], FAMILIES.index(z["family"]))):
        key = row["instance"], row["seed"], row["fe_index"], row["child_id"]
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def metrics(rows, pred):
    result = sensitivity.prediction_metrics(rows, pred)
    result["actual_improvement_rate"] = statistics.mean(z["actual"] > sensitivity.TOL for z in rows) if rows else None
    result["predictor"] = pred
    return result


def group_summary(rows, pred):
    result = [{"group": "overall", **metrics(rows, pred)}]
    for name in local.INSTANCES:
        subset = [z for z in rows if z["instance"] == name]
        result.append({"group": name, **metrics(subset, pred)})
        for seed in local.SEEDS:
            result.append({"group": f"{name}/seed{seed}", **metrics([z for z in subset if z["seed"] == seed], pred)})
    return result


def positive_information(groups):
    def good(g):
        return g["actual_nonzero_sign_accuracy"] is not None and g["actual_nonzero_sign_accuracy"] > .5 and g["spearman"] is not None and g["spearman"] > 0
    instances = [g for g in groups if g["group"] in local.INSTANCES]
    seeds = [g for g in groups if "/seed" in g["group"]]
    return len(instances) == 2 and all(good(g) for g in instances) and sum(good(g) for g in seeds) >= 4


def locality_summary(rows):
    fields = ["shock_nodes_total", "shock_L1_mass", "weighted_shock_mass",
              *[f"top{k}_share" for k in (1,3,5,10)], *[f"weighted_top{k}_share" for k in (1,3,5,10)]]
    result = []
    groups = [("overall", rows), *[(name, [z for z in rows if z["instance"] == name]) for name in local.INSTANCES],
              *[(f"{name}/seed{seed}", [z for z in rows if z["instance"] == name and z["seed"] == seed]) for name in local.INSTANCES for seed in local.SEEDS]]
    for label, subset in groups:
        for scope in ("all", "High"):
            values = subset if scope == "all" else [z for z in subset if z["coverage_bin"] == "High"]
            row = {"group": label, "scope": scope, "cases": len(values)}
            for field in fields:
                data = [z[field] for z in values if z[field] is not None]
                row[field+"_median"] = statistics.median(data) if data else None
                row[field+"_mean"] = statistics.mean(data) if data else None
            result.append(row)
    return result


def boundary_stage(rows, fingerprint):
    records, root_rows, diff_rows = [], [], []
    for row in rows:
        graph = sensitivity.restore_graph(json.loads(sensitivity.state_path(row).read_text())["graph"])
        cached = json.loads((ROOT / "cache/children" / row["instance"] / f"seed_{row['seed']}" / f"fe_{row['fe_index']}" / f"{row['child_id']}.json").read_text())
        if cached["fingerprint"] != fingerprint:
            raise AssertionError("Child graph cache identity changed")
        child_graph = sensitivity.restore_graph(cached["child_graph"])
        e = {int(n): v for n, v in cached["residual"].items()}
        target = tuple(row["target"])
        roots = boundary_roots(graph, child_graph, target, row["old_machine"], row["new_machine"], row["delta_p"])
        valid = [n for n in roots if not sensitivity.invalid_path(graph, n)]
        target_ids = [n["id"] for n in graph["nodes"] if tuple(n["uid"][:2]) == target and not sensitivity.invalid_path(graph, n["id"])]
        prediction = -sum(graph["lambda"][n]*e[n] for n in valid)
        target_prediction = -sum(graph["lambda"][n]*e[n] for n in target_ids)
        direct = -sum(graph["g"].get(n, 0)*e[n] for n in valid)
        mass = sum(abs(graph["lambda"][n]*value) for n, value in e.items() if not sensitivity.invalid_path(graph,n))
        coverage = sum(abs(graph["lambda"][n]*e[n]) for n in valid)/mass if mass else None
        record = {**row, "boundary_prediction": prediction, "target_only_prediction": target_prediction,
                  "direct_g_boundary_prediction": direct, "boundary_root_count": len(roots),
                  "boundary_valid_root_count": len(valid), "smooth_weighted_boundary_share": coverage}
        records.append(record)
        details = {n["id"]: n["uid"] for n in graph["nodes"]}
        root_rows.extend({**sensitivity.identity(row), "child_id": row["child_id"], "node": n,
                          "event_uid": details[n], "reasons": reasons, "shock": e[n],
                          "path_valid": n in valid, "lambda": graph["lambda"][n]} for n,reasons in roots.items())
        diff = edge_diff(graph, child_graph, True)
        diff_rows.extend({**sensitivity.identity(row), "child_id": row["child_id"], "change": mode,
                          "src_uid": edge[0], "dst_uid": edge[1], "kind": edge[2], "alpha": edge[3]}
                         for mode in ("deleted", "added") for edge in diff[mode])
    return records, root_rows, diff_rows


def within_target_ranks(rows, pred):
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["instance"],row["seed"],row["fe_index"],tuple(row["target"]))][row["new_machine"]] = row
    result = []
    for (name, seed, fe, target), group in sorted(groups.items()):
        values = list(group.values())
        if len(values) < 2:
            continue
        best = min(values,key=lambda z:(-z[pred],z["new_machine"]))
        result.append({"instance":name,"seed":seed,"fe_index":fe,"target":list(target),"predictor":pred,
                       "alternatives":len(values),"spearman":sensitivity.corr([z[pred] for z in values],[z["actual"] for z in values]),
                       "top1_improves":best["actual"]>sensitivity.TOL,"top1_actual":best["actual"],
                       "random_candidate_hit_rate":statistics.mean(z["actual"]>sensitivity.TOL for z in values)})
    return result


def ect_slot_metadata(schedule, target, machine, ready, duration):
    """Annotations of the frozen scalar slot, NOT a new insertion heuristic."""
    start = active.earliest_slot(schedule,target,machine,ready,duration)
    occupied = sorted((z for z in schedule if z["machine"]==machine and (z["job"],z["op"])!=target),key=lambda z:(z["start"],z["release"],z["job"],z["op"]))
    before = [z for z in occupied if z["release"]<=start]
    after = [z for z in occupied if z["start"]>=start+duration]
    key = lambda z:[z["job"],z["op"]] if z else None
    return {"predicted_start":start,"predicted_completion":start+duration,
            "seed_slot_predecessor":key(before[-1] if before else None),
            "seed_slot_successor":key(after[0] if after else None),
            "factual_child_neighbors_guaranteed":False}


def ect_capability_audit():
    # Read-only source audit. The current helper does not predict release under
    # counterfactual blocking or actual dynamic-priority machine insertion.
    return {
        "source":"experiments/sis_active_lever.py:readiness,earliest_slot,ect_options",
        "target_start":True,"target_completion":True,
        "seed_slot_neighbors_can_be_annotated":True,
        "factual_counterfactual_machine_neighbors":False,
        "target_release_under_blocking":False,
        "first_changed_unblocking_regime":False,
        "reason":"ECT scans frozen seed [start,release) occupancy. It returns earliest completion, not a counterfactual release or factual new decoder sequence. Finite-buffer release depends on future takes; deterministic priority competition and changing WIP readiness require counterfactual scheduling to resolve in general.",
        "decision":"NOT TESTED: no complete seed-only shock predictor invented. Stop before candidate scoring/decode stage.",
    }


def freeze_scores(records, directory):
    """Audit utility only; does not invent or compute any candidate score."""
    forbidden={"actual","B_actual","shortage_child","S_child","child_objective"}
    if any(forbidden & set(row) for row in records):
        raise ValueError("Outcome fields cannot enter predecode score table")
    path=directory/"candidate_scores_frozen.csv"
    local.save_csv(path,records)
    certificate={"path":str(path),"sha256":role.file_hash(path),
                 "scoring_timestamp_UTC":datetime.now(timezone.utc).isoformat(),
                 "child_objectives_loaded":False}
    r.write_json_atomic(directory/"score_freeze_certificate.json",certificate)
    return certificate


def save(name, rows, folder="analysis"):
    path=ROOT/folder/name
    if rows:
        local.save_csv(path,rows)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("status,reason\nNOT_TESTED,Prior gate or unsupported frozen predictor prevents this stage\n")


def finish(summary, records):
    summary["formal_FE"]=0
    summary["recommendation"]="GO" if all(summary["stages"].get(s)=="PASS" for s in ("A","B","C","D","E")) else "NO-GO"
    r.write_json_atomic(ROOT/"analysis/bridge_validation_summary.json",summary)
    lines=["# Discrete Action-to-Event Bridge Validation","",f"**{summary['recommendation']}**","",
           "## A. Fidelity",json.dumps(summary["fidelity"],indent=2),
           "Previous g/graph PASS, 504 propagation-only and forward/reverse certificates are hash-pinned. No old sensitivity formula or formal source was changed.",
           "## B. Residual formulation",json.dumps(summary["residual_validation"],indent=2),
           "e=(I-A)delta_t; reconstructing delta_t from all e and full Lambda residual=full-time g oracle are algebraic identities, not new evidence of prediction accuracy.",
           "## C. Full residual oracle",json.dumps(summary.get("A",{}),indent=2),
           "High/Medium/Low use validated downstream Lambda-path weighted coverage, not merely absence of a local tie. Zero weighted mass is separate; nonsmooth branch values are never ordinary derivatives.",
           "## D. Shock locality",json.dumps(summary.get("B",{}),indent=2),
           "No actual-shortage filtering in locality statistics. Shared genotypes across policy labels are deduplicated for pooled primary, retained for family summaries.",
           "## E. Topology boundary oracle",json.dumps(summary.get("C",{}),indent=2),
           "Boundary membership uses target, immediate old/new successor starts, known processing-time change and at most one first directly affected release regime. It never uses actual benefit or shock magnitude to select roots.",
           "## F. Lambda vs direct-g / target-only","See boundary_oracle_summary.csv and within_state_ranking.csv if reached; otherwise NOT TESTED.",
           "## G. Predecode prediction",json.dumps(summary.get("D",{}),indent=2),
           "## H. MS ranking","NOT TESTED unless all earlier gates permit a complete seed-only predictor. No best-of-child score leakage.",
           "## I. Instance and seed stability",json.dumps(summary.get("instance_groups",[]),indent=2),
           "## J. Failure localization",summary["failure"],
           "## K. Recommendation",summary["recommendation"],
           "No formal Full/SIS/P_S/default changes, formal FE=0, no pre-ablation or new search trajectory."]
    (ROOT/"analysis/bridge_validation_interpretation.md").write_text("\n".join(lines)+"\n")
    if records:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        folder=ROOT/"figures";folder.mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(6,5))
        for label,color in (("High","#177f86"),("Medium","#8c8c8c"),("Low","#b96b65")):
            values=[z for z in records if z["coverage_bin"]==label]
            ax.scatter([z["smooth_prediction"] for z in values],[z["actual"] for z in values],s=9,alpha=.45,label=label,color=color)
        ax.set(xlabel="Smooth residual oracle benefit",ylabel="Actual shortage benefit")
        ax.legend();fig.tight_layout();fig.savefig(folder/"residual_oracle.png",dpi=160);plt.close(fig)
        if summary["stages"]["B"]!="NOT TESTED":
            fig,ax=plt.subplots(figsize=(6,4))
            for name in local.INSTANCES:
                values=[z for z in records if z["instance"]==name and z["weighted_shock_mass"]>sensitivity.TOL]
                ax.plot([1,3,5,10],[statistics.median(z[f"weighted_top{k}_share"] for z in values) for k in (1,3,5,10)],marker="o",label=name)
            ax.set(xlabel="Top-k residual nodes",ylabel="Median Lambda-weighted mass coverage",ylim=(0,1.03));ax.legend()
            fig.tight_layout();fig.savefig(folder/"shock_locality.png",dpi=160);plt.close(fig)
    print(f"Bridge audit: {summary['recommendation']}; stages={summary['stages']}; failure={summary['failure']}",flush=True)
    return summary


def main():
    if os.environ.get("PYTHONHASHSEED")!="0":
        raise RuntimeError("PYTHONHASHSEED=0 required")
    fingerprint,previous,fd_rows=source_certificate()
    processing=validate_processing_residual(fd_rows)
    save("processing_residual_sanity.csv",processing)
    tasks=[r.Task(name,"full",seed) for name in local.INSTANCES for seed in local.SEEDS]
    batches=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context("spawn")) as executor:
        futures=[executor.submit(audit_task,t,fingerprint) for t in tasks]
        for future in as_completed(futures):
            batches.append(future.result())
    labeled=[z for batch in batches for z in batch["rows"]]
    rows=unique_rows(labeled)
    if Counter(z["family"] for z in labeled)!=Counter({"ECT-AMR":716,"RandomMS":716,"FastestProcessingMS":716,"BasicMutation-MS":944}):
        raise AssertionError("Frozen MS action coverage differs")
    save("ms_child_alignment.csv",[z for b in batches for z in b["alignments"]],"raw")
    save("residual_shocks.csv",[z for b in batches for z in b["shocks"]],"raw")
    save("ms_child_oracles.csv",labeled,"raw")
    for name in ("topology_edge_diff.csv","boundary_roots.csv","predecode_root_predictions.csv","candidate_scores_frozen.csv","candidate_actuals.csv"):
        save(name,[],"raw")
    for name in ("shock_locality_summary.csv","boundary_oracle_summary.csv","root_prediction_summary.csv","within_state_ranking.csv"):
        save(name,[])
    high=[z for z in rows if z["coverage_bin"]=="High"]
    groups=group_summary(high,"smooth_prediction")
    oracle=[]
    for coverage in ("High","Medium","Low","zero_weighted","all"):
        selected=rows if coverage=="all" else [z for z in rows if z["coverage_bin"]==coverage]
        oracle.extend({"coverage_bin":coverage,**g} for pred in ("smooth_prediction","full_residual_prediction","fulltime_prediction") for g in group_summary(selected,pred))
    save("residual_oracle_summary.csv",oracle)
    save("instance_seed_summary.csv",groups)
    save("action_family_summary.csv",[{"family":family,"coverage_bin":coverage,**metrics([z for z in labeled if z["family"]==family and (coverage=="all" or z["coverage_bin"]==coverage)],pred)}
         for family in FAMILIES for coverage in ("High","Medium","Low","zero_weighted","all") for pred in ("smooth_prediction","full_residual_prediction","fulltime_prediction")])
    save("calibration_bins.csv",[z for pred in ("smooth_prediction","full_residual_prediction","fulltime_prediction") for z in sensitivity.calibration_rows(
         [{**row,"topology_group":row["coverage_bin"]} for row in labeled],pred)])
    summary={"fingerprint":fingerprint,"stages":{s:"NOT TESTED" for s in ("A","B","C","D","E")},
             "fidelity":{"states":240,"prior_direct":"PASS","prior_graph":"PASS","prior_propagation_only":504,
             "new_trajectory_runs":0,"unique_MS_children":len(rows),"labeled_MS_children":len(labeled),
             "offline_child_decodes":sum(b["decodes"] for b in batches),"processing_sanity_decodes":len(processing),
             "source_changes":"New independent diagnostic module/tests only"},
             "residual_validation":{"processing_cases":len(processing),"passed":True,
             "max_processing_error":max(z["root_residual_error"] for z in processing),
             "max_identity_error":max(z["identity_error"] for z in rows),
             "max_reconstruction_error":max(z["reconstruction_error"] for z in rows)},
             "A":{"coverage":dict(Counter(z["coverage_bin"] for z in rows)),"primary_high_groups":groups},"instance_groups":groups}
    if not positive_information(groups):
        summary["stages"]["A"]="FAIL" if any(abs(z["actual"])>sensitivity.TOL for z in high) else "NOT TESTED"
        summary["failure"]="Layer A: High validated-smooth-shock residual oracle lacks reliable nonzero sign/rank information across both instances and six groups. No locality/boundary/predecode stage performed."
        return finish(summary,rows)
    summary["stages"]["A"]="PASS"
    locality=locality_summary(rows);save("shock_locality_summary.csv",locality)
    shocks=[z for batch in batches for z in batch["shocks"]]
    for field,filename in (("node_type","shock_node_type_summary.csv"),("primary_origin","shock_origin_summary.csv")):
        grouped=defaultdict(list)
        for row in shocks:
            key=row["event_uid"][2] if field=="node_type" else row[field]
            grouped[key].append(row)
        save(filename,[{field:key,"shock_nodes":len(values),
             "L1_mass":sum(abs(z["shock"]) for z in values),
             "Lambda_weighted_mass":sum(z["weighted_mass"] for z in values),
             "validated_path_weighted_mass":sum(z["weighted_mass"] for z in values if z["path_valid"])}
             for key,values in sorted(grouped.items())])
    inst=[z for z in locality if z["group"] in local.INSTANCES]
    seed_groups=[z for z in locality if "/seed" in z["group"]]
    sufficient=lambda z:z["weighted_top5_share_median"] is not None and z["weighted_top5_share_median"]>=.8
    locality_pass=all(sufficient(z) for z in inst) and all(sum(sufficient(z) for z in seed_groups if z["scope"]==scope)>=4 for scope in ("all","High"))
    summary["B"]={"groups":locality,"gate_passed":locality_pass}
    summary["stages"]["B"]="PASS" if locality_pass else "FAIL"
    if not locality_pass:
        summary["failure"]="Layer B: MS residual shocks are insufficiently concentrated in a small Lambda-weighted root set. No boundary or predecode predictor developed."
        return finish(summary,rows)
    boundary,root_rows,diff_rows=boundary_stage(rows,fingerprint)
    save("boundary_roots.csv",root_rows,"raw");save("topology_edge_diff.csv",diff_rows,"raw")
    primary=[z for z in boundary if z["coverage_bin"]=="High"]
    predictors=("smooth_prediction","boundary_prediction","target_only_prediction","direct_g_boundary_prediction")
    boundary_groups=[g for p in predictors for g in group_summary(primary,p)]
    save("boundary_oracle_summary.csv",boundary_groups)
    ranks=[z for p in predictors for z in within_target_ranks(primary,p)]
    save("within_state_ranking.csv",ranks)
    by={(g["group"],g["predictor"]):g for g in boundary_groups}
    def rank_average(name,pred):
        instance,_,seed_label=name.partition("/seed")
        vals=[z["spearman"] for z in ranks if z["instance"]==instance and
              (not seed_label or z["seed"]==int(seed_label)) and
              z["predictor"]==pred and z["spearman"] is not None]
        return statistics.mean(vals) if vals else None
    def better(group):
        full=by[(group,"smooth_prediction")];b=by[(group,"boundary_prediction")]
        retained=(b["actual_nonzero_sign_accuracy"] is not None and b["actual_nonzero_sign_accuracy"]>=.8*(full["actual_nonzero_sign_accuracy"] or 0) and
                  b["spearman"] is not None and b["spearman"]>=.8*(full["spearman"] or 0))
        rank_boundary=rank_average(group,"boundary_prediction")
        superior=all((b["actual_nonzero_sign_accuracy"] or 0)>(by[(group,p)]["actual_nonzero_sign_accuracy"] or 0) or
                     (rank_boundary is not None and rank_average(group,p) is not None and
                      rank_boundary>rank_average(group,p)) for p in ("target_only_prediction","direct_g_boundary_prediction"))
        return retained and superior
    boundary_pass=all(better(name) for name in local.INSTANCES) and sum(better(f"{name}/seed{seed}") for name in local.INSTANCES for seed in local.SEEDS)>=4
    summary["C"]={"groups":boundary_groups,"root_count_median":statistics.median(z["boundary_root_count"] for z in boundary),
                  "valid_boundary_weighted_share_median":statistics.median(z["smooth_weighted_boundary_share"] for z in primary if z["smooth_weighted_boundary_share"] is not None),
                  "within_target_ranks":{name:{p:rank_average(name,p) for p in predictors} for name in local.INSTANCES}}
    summary["stages"]["C"]="PASS" if boundary_pass else "FAIL"
    if not boundary_pass:
        summary["failure"]="Layer C: small topology-boundary roots fail to retain sufficient full residual information and/or fail to improve over BOTH target-only and direct-g baselines consistently. No predecode scoring."
        return finish(summary,boundary)
    audit=ect_capability_audit();summary["D"]=audit
    r.write_json_atomic(ROOT/"analysis/ect_slot_capability_audit.json",audit)
    summary["failure"]="Layer D: frozen ECT gives seed-occupancy completion slots but no generally determined counterfactual release/blocking regime or factual new-machine neighbors. A complete predecode residual vector cannot be asserted without unsupported scheduling assumptions. No predictor or candidate score was invented."
    return finish(summary,boundary)


if __name__=="__main__":
    main()
