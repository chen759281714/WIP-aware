"""DISVA: frozen diagnosis subsets, neutral restricted mutation, offline FE only.

Run: PYTHONHASHSEED=0 python3 -m experiments.diagnostics.diagnosis_subproblem_viability
Checkpoints are per-state and source-pinned. No formal trajectory is run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from experiments import run_pre_ablation as r
from experiments import sis_active_lever as active
from experiments import sis_local_viability as local
from experiments import sis_role_direction as role
from experiments import sis_selection_aware as selection
from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation

ROOT = r.PROJECT_ROOT / "experiments/results/diagnosis_subproblem_viability"
SET_REPS, PROPOSALS, SEARCH_TRIALS = 4, 16, 4
BUDGETS = (8, 16, 32)
KINDS = ("DIAG-CORE", "DIAG-SUPPORT-RANDOM", "GLOBAL-RANDOM", "OPPORTUNITY-MATCHED")
RULES = {
    "SET_REPS": SET_REPS, "PROPOSALS": PROPOSALS, "SEARCH_TRIALS": SEARCH_TRIALS,
    "budgets": BUDGETS, "formal_FE": 0, "rho": local.local_settings().rho,
    "core": "Exact frozen H; no new top-k/radius/direction/gradient",
    "random_sets": "Uniform without replacement; isolated SHA256-seeded RNG; opportunity strata=(op_idx,OS-actionable,legal alternatives), includes H in all-O pool, no relaxation",
    "stage": "Fixed-route stage = op_idx (zero based)",
    "stage1": "Core 16 proposals; each random subset has 4 sets x 16 proposals; all start from the same seed; no child-based action filtering",
    "stage2": "4 episodes per kind/state: core uses same H in all 4; random trial i uses subset replicate i, not 4x4 pseudo-replication; freeze D and never re-diagnose",
    "acceptance": "Strict shortage decrease only; Cmax is secondary; exhausted terminates, no global fallback",
    "paired_unit": "State: average random set/trial results first; dependent six trajectories, no inferential p-values",
    "gate1": "Strong core: core-global AND core-support positive on both EAG/best16, both instances, pooled W>L, >=4/6 trajectory means positive. Support-only: support-global satisfies same checks and core-support does not; otherwise insufficient stable evidence STOP",
    "gate2": "At B16 AND B32 core-global OR support-global has positive mean both instances, pooled W>L, >=4/6 positive trajectories; robustness requires all leave-one-state and leave-one-trajectory pooled means positive. Cmax distributions reported, no post-hoc threshold or acceptance constraint",
    "ties": "Exact numeric zero; no post-hoc tolerance on integer objective gains",
    "workers": 2,
}


def diagnostic_rng(state, stream, replicate=0):
    identity = (state["instance"], state["seed"], state["fe_index"], stream, replicate)
    seed = int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest(), "big")
    return random.Random(seed)


def atomic_space(engine, ind, allowed):
    """Same enumeration/order as formal basic_mutation, with only an allow filter."""
    positions = {key: i for i, key in enumerate(ind.os_seq)}
    windows = {}
    for key, pos in positions.items():
        if key not in allowed:
            continue
        lower = positions[(key[0], key[1]-1)] + 1 if key[1] else 0
        upper = (positions[(key[0], key[1]+1)] - 1
                 if key[1]+1 < len(engine.operations[key[0]]) else len(ind.os_seq)-1)
        if lower < pos or pos < upper:
            windows[key] = (lower, upper)
    ms = [key for key in engine.encoder.ms_index_order if key in allowed
          and len(engine.operations[key[0]][key[1]]["machines"]) >= 2]
    return positions, windows, ms


def restricted_basic_mutation(engine, ind, allowed, rng):
    positions, windows, ms = atomic_space(engine, ind, allowed)
    options = (["os"] if windows else []) + (["ms"] if ms else [])
    child = Individual(ind.os_seq[:], ind.ms_map.copy(), operation_order=ind.operation_order)
    if not options:
        return child, {"exhausted": True, "move_type": "none", "operation": None}
    kind = rng.choice(options)
    if kind == "ms":
        key = rng.choice(ms)
        machines = [m for m in engine.operations[key[0]][key[1]]["machines"]
                    if m != ind.ms_map[key]]
        machine = rng.choice(machines)
        child.ms_map[key] = machine
        return child, {"exhausted": False, "move_type": "MS", "operation": list(key),
                       "old_machine": ind.ms_map[key], "new_machine": machine}
    key = rng.choice(list(windows))
    lower, upper = windows[key]
    old = positions[key]
    target = rng.choice([pos for pos in range(lower, upper+1) if pos != old])
    child.os_seq = engine._insert(ind.os_seq, key, target)
    if child.os_seq is None:
        raise AssertionError("Formal atomic insertion unexpectedly unchanged")
    return child, {"exhausted": False, "move_type": "OS", "operation": list(key),
                   "old_position": old, "target_gap": target,
                   "realized_position": child.os_seq.index(key)}


def assert_restricted(seed, child, allowed, metadata):
    if metadata["exhausted"]:
        if child.os_seq != seed.os_seq or child.ms_map != seed.ms_map:
            raise AssertionError("Exhausted mutation changed genotype")
        return
    key = tuple(metadata["operation"])
    if key not in allowed:
        raise AssertionError("Selected operation outside frozen subset")
    changed_ms = {op for op in seed.ms_map if seed.ms_map[op] != child.ms_map[op]}
    if metadata["move_type"] == "MS":
        valid = changed_ms == {key} and child.os_seq == seed.os_seq
    else:
        valid = not changed_ms and ([op for op in seed.os_seq if op != key] ==
                                    [op for op in child.os_seq if op != key])
    if not valid:
        raise AssertionError("More than selected variable actively modified")
    # Multiple accepted insertions must also preserve the entire non-D projection.
    if ([op for op in seed.os_seq if op not in allowed] !=
            [op for op in child.os_seq if op not in allowed]):
        raise AssertionError("Non-relaxed priority projection changed")


def opportunity_features(engine, ind):
    _, windows, ms = atomic_space(engine, ind, set(ind.os_seq))
    return {op: (op[1], op in windows,
                 len(engine.operations[op[0]][op[1]]["machines"])-1)
            for op in engine.encoder.ms_index_order}


def matched_subset(core, candidates, features, rng):
    pools = defaultdict(list)
    for op in candidates:
        pools[features[op]].append(op)
    demands = Counter(features[op] for op in core)
    unmatched = sum(max(0, count-len(pools[stratum])) for stratum, count in demands.items())
    if unmatched:
        return None, unmatched
    chosen = []
    for stratum, count in sorted(demands.items()):
        chosen.extend(rng.sample(pools[stratum], count))
    return frozenset(chosen), 0


def construct_subsets(state, operations, features):
    """Reads seed phi/H only; never sees any child or decoded child objective."""
    core = frozenset(tuple(op) for op in state["H"])
    support = sorted(tuple(op) for op, value in state["phi"] if value > 0)
    if not core <= set(support) or len(core) > len(support):
        raise AssertionError("Invalid positive support/core")
    sets = [{"kind": KINDS[0], "replicate": 0, "D": core, "degenerate": False, "unmatched": 0}]
    for kind in KINDS[1:]:
        for rep in range(SET_REPS):
            rng = diagnostic_rng(state, "subset/"+kind, rep)
            unmatched = 0
            degenerate = kind == KINDS[1] and len(support) == len(core)
            if kind == KINDS[1]:
                chosen = core if degenerate else frozenset(rng.sample(support, len(core)))
            elif kind == KINDS[2]:
                chosen = frozenset(rng.sample(operations, len(core)))
            else:
                chosen, unmatched = matched_subset(core, operations, features, rng)
            sets.append({"kind": kind, "replicate": rep, "D": chosen,
                         "degenerate": degenerate, "unmatched": unmatched})
    return sets


def geometry(state, ind, engine, subsets, features):
    positive = {tuple(op) for op, value in state["phi"] if value > 0}
    phi = {tuple(op): value for op, value in state["phi"]}
    core = {tuple(op) for op in state["H"]}
    _, windows, _ = atomic_space(engine, ind, set(ind.os_seq))
    rows, members = [], []
    for subset in subsets:
        chosen = subset["D"]
        base = {k: state[k] for k in ("instance", "seed", "fe_index")}
        base.update({k: v for k, v in subset.items() if k != "D"})
        base.update(H_size=len(core), V_plus_size=len(positive), O_size=len(ind.os_seq),
                    H_ratio=len(core)/len(ind.os_seq), V_plus_ratio=len(positive)/len(ind.os_seq),
                    H_support_ratio=len(core)/len(positive) if positive else None,
                    H_mass=sum(phi[op] for op in core), S_seed=state["S_seed"])
        if chosen is None:
            rows.append({**base, "size": None, "strict_match": False})
            continue
        n = len(chosen)
        os_count = sum(features[op][1] for op in chosen)
        ms_count = sum(features[op][2] > 0 for op in chosen)
        both = sum(features[op][1] and features[op][2] > 0 for op in chosen)
        rows.append({**base, "size": n, "strict_match": True,
                     "positive_fraction": sum(op in positive for op in chosen)/n if n else 0,
                     "phi_mass": sum(phi.get(op, 0) for op in chosen),
                     "OS_fraction": os_count/n if n else 0, "MS_fraction": ms_count/n if n else 0,
                     "both_fraction": both/n if n else 0,
                     "no_action_fraction": (n-os_count-ms_count+both)/n if n else 0,
                     "mean_MS_alternatives": statistics.mean(features[op][2] for op in chosen) if n else 0,
                     "mean_OS_targets": statistics.mean(windows[op][1]-windows[op][0] if op in windows else 0 for op in chosen) if n else 0,
                     "stage_counts": dict(sorted(Counter(op[1] for op in chosen).items())),
                     "MS_alternative_counts": dict(sorted(Counter(features[op][2] for op in chosen).items())),
                     "overlap_H": len(chosen & core)/n if n else 0})
        for op in sorted(chosen):
            members.append({**base, "job": op[0], "op_idx": op[1], "phi": phi.get(op, 0),
                            "OS_actionable": features[op][1], "MS_alternatives": features[op][2]})
    return rows, members


def offline_decode(decoder, ind, full=False):
    cmax, schedule, trace, provenance = decoder.decode(ind.os_seq, ind.ms_map, return_provenance=True)
    stats = decoder.analyze(schedule, trace, cmax)
    ind.makespan, ind.shortage = cmax, stats["shortage"]["total_shortage_area"]
    if full:
        ind.schedule, ind.buffer_trace, ind.provenance, ind.stats = schedule, trace, provenance, stats
    return ind


def verify_state(engine, state):
    ind = offline_decode(engine.decoder, r.prefix_individual(state["genome"]), full=True)
    if r.prefix_genome(ind) != state["genome"] or ind.makespan != state["Cmax_seed"] or ind.shortage != state["S_seed"]:
        raise AssertionError("Historical genotype/objectives mismatch")
    saved = (copy.deepcopy(engine.diagnostics), engine.unanchored_shortage_intervals,
             engine.phi_mass_mismatch_count)
    try:
        diagnosis = engine.diagnose_shortage(ind)
        high = engine.high_influence_set(ind, diagnosis)
        engine.local_instance, engine.local_seed = state["instance"], state["seed"]
        rebuilt = local.capture_state(engine, ind, diagnosis, high)
    finally:
        engine.diagnostics, engine.unanchored_shortage_intervals, engine.phi_mass_mismatch_count = saved
    for field in ("genome", "S_seed", "Cmax_seed", "buffer_shortage", "phi", "H", "diagnosis"):
        if rebuilt is None or rebuilt[field] != state[field]:
            raise AssertionError("Historical diagnosis mismatch: "+field)
    phi = diagnosis.phi
    ordered = sorted(phi, key=lambda op: (-phi[op], op))
    size = len(high)
    if set(ordered[:size]) != high or sum(phi[op] for op in high)/ind.shortage < engine.rho:
        raise AssertionError("Frozen H coverage differs")
    if size > 1 and sum(phi[op] for op in ordered[:size-1])/ind.shortage >= engine.rho:
        raise AssertionError("H is not minimal")
    return ind


def hill_climb(engine, seed, allowed, rng, budget=32, decode=None):
    """Frozen D, no diagnosis, one neutral atomic proposal per offline evaluation."""
    allowed = frozenset(allowed)
    incumbent = seed.copy()
    decode = decode or (lambda child: offline_decode(engine.decoder, child))
    accepted = Counter()
    first = None
    unique = {selection.digest(engine.genotype_key(seed))}
    evaluations, attempts, exhausted = 0, 0, False
    checkpoints = []
    for step in range(1, budget+1):
        child, meta = restricted_basic_mutation(engine, incumbent, allowed, rng)
        attempts += 1
        assert_restricted(incumbent, child, allowed, meta)
        if meta["exhausted"]:
            exhausted = True
            break
        child = decode(child)
        evaluations += 1
        unique.add(selection.digest(engine.genotype_key(child)))
        if child.shortage < incumbent.shortage:
            incumbent = child
            accepted[meta["move_type"]] += 1
            if first is None:
                first = evaluations
        if step in BUDGETS:
            checkpoints.append(search_snapshot(seed, incumbent, step, evaluations, attempts,
                                               accepted, first, unique, exhausted))
    for checkpoint in BUDGETS:
        if checkpoint <= budget and not any(z["budget"] == checkpoint for z in checkpoints):
            checkpoints.append(search_snapshot(seed, incumbent, checkpoint, evaluations, attempts,
                                               accepted, first, unique, exhausted))
    return sorted(checkpoints, key=lambda z: z["budget"])


def search_snapshot(seed, incumbent, budget, evaluations, attempts, accepted, first, unique, exhausted):
    gain = seed.shortage-incumbent.shortage
    return {"budget": budget, "gain": gain, "gain_per_evaluation": gain/evaluations if evaluations else 0,
            "any_improvement": gain > 0, "accepted_moves": sum(accepted.values()),
            "OS_accepted": accepted["OS"], "MS_accepted": accepted["MS"],
            "time_to_first_evaluation": first, "best_shortage": incumbent.shortage,
            "Cmax_delta": incumbent.makespan-seed.makespan, "unique_genotypes": len(unique),
            "actual_offline_decodes": evaluations, "attempts": attempts, "exhausted": exhausted}


def preregister():
    source = json.loads((active.OUTPUT_ROOT/"source_manifest.json").read_text())
    files = {**source["checked_sources"], **source["capture_sha256"]}
    for path, expected in files.items():
        if role.file_hash(r.PROJECT_ROOT/path) != expected:
            raise AssertionError("Frozen source/capture changed: "+path)
    for name in local.INSTANCES:
        path = r.INSTANCE_DIR/f"{name}.json"
        files[str(path.relative_to(r.PROJECT_ROOT))] = role.file_hash(path)
    for path in (__file__, local.__file__, r.__file__):
        relative = str(Path(path).relative_to(r.PROJECT_ROOT))
        files[relative] = role.file_hash(Path(path))
    payload = {"rules": RULES, "files": files,
               "capture_fingerprint": source["original_capture_fingerprint"]}
    path = ROOT/"preregistration.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(payload)):
        raise AssertionError("Refuse changed preregistration; no post-hoc gates")
    r.write_json_atomic(path, payload)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def state_cache(task, state, phase):
    return ROOT/"cache"/phase/task.instance/f"seed_{task.seed}"/f"fe_{state['fe_index']}.json"


def prepare_task(task, fingerprint, payload):
    path = ROOT/"cache/verified"/task.instance/f"seed_{task.seed}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["fingerprint"] != fingerprint:
            raise AssertionError("Stale verification cache")
        return result
    data = json.loads((r.INSTANCE_DIR/f"{task.instance}.json").read_text())
    engine = WIPGraphDualPopulation(data["operations"], data["buffers"], **local.local_settings().algorithm_parameters(task.seed))
    before = local.formal_state_signature(engine)
    capture = local.load_capture(task, local.local_settings(), payload["capture_fingerprint"])
    states, rows, members = [], [], []
    for state in capture["states"]:
        ind = verify_state(engine, state)
        compact = {k: state[k] for k in ("instance", "seed", "fe_index", "genome", "S_seed", "Cmax_seed", "H", "phi")}
        features = opportunity_features(engine, ind)
        subsets = construct_subsets(compact, engine.encoder.ms_index_order, features)
        geo, membership = geometry(compact, ind, engine, subsets, features)
        states.append(compact); rows.extend(geo); members.extend(membership)
    if local.formal_state_signature(engine) != before:
        raise AssertionError("Seed verification changed formal state")
    result = {"fingerprint": fingerprint, "states": states, "geometry": rows, "members": members,
              "seed_decodes": len(states), "fidelity_passed": True}
    r.write_json_atomic(path, result)
    print(f"DISVA fidelity PASS: {task.instance} seed={task.seed}, {len(states)} states", flush=True)
    return result


def experiment_task(task, fingerprint, phase):
    verified = json.loads((ROOT/"cache/verified"/task.instance/f"seed_{task.seed}.json").read_text())
    if verified["fingerprint"] != fingerprint:
        raise AssertionError("Stale state cache")
    data = json.loads((r.INSTANCE_DIR/f"{task.instance}.json").read_text())
    engine = WIPGraphDualPopulation(data["operations"], data["buffers"], **local.local_settings().algorithm_parameters(task.seed))
    before = local.formal_state_signature(engine)
    rows = []
    for state in verified["states"]:
        path = state_cache(task, state, phase)
        if path.exists():
            record = json.loads(path.read_text())
            if record["fingerprint"] != fingerprint:
                raise AssertionError("Stale state experiment cache")
            rows.extend(record["rows"])
            continue
        seed = r.prefix_individual(state["genome"])
        seed.shortage, seed.makespan = state["S_seed"], state["Cmax_seed"]
        features = opportunity_features(engine, seed)
        subsets = construct_subsets(state, engine.encoder.ms_index_order, features)
        current = []
        for subset in subsets:
            if subset["D"] is None:
                continue
            if phase == "stage2" and subset["kind"] == KINDS[0]:
                trials = range(SEARCH_TRIALS)
            else:
                trials = [subset["replicate"]]
            for trial in trials:
                base = {k: state[k] for k in ("instance", "seed", "fe_index")}
                base.update(kind=subset["kind"], replicate=subset["replicate"], trial=trial,
                            subset_hash=selection.digest(sorted(subset["D"])), subset_size=len(subset["D"]),
                            S_seed=seed.shortage, Cmax_seed=seed.makespan)
                rng = diagnostic_rng(state, phase+"/"+subset["kind"], trial)
                if phase == "stage2":
                    current.extend({**base, **row} for row in hill_climb(engine, seed, subset["D"], rng))
                    continue
                for proposal in range(PROPOSALS):
                    child, metadata = restricted_basic_mutation(engine, seed, subset["D"], rng)
                    assert_restricted(seed, child, subset["D"], metadata)
                    if not metadata["exhausted"]:
                        offline_decode(engine.decoder, child)
                    else:
                        child.shortage, child.makespan = seed.shortage, seed.makespan
                    current.append({**base, **metadata, "proposal": proposal,
                                    "gain": seed.shortage-child.shortage, "S_child": child.shortage,
                                    "Cmax_delta": child.makespan-seed.makespan,
                                    "actual_offline_decodes": int(not metadata["exhausted"]),
                                    "child_genotype": selection.digest(engine.genotype_key(child))})
        r.write_json_atomic(path, {"fingerprint": fingerprint, "rows": current})
        rows.extend(current)
        print(f"DISVA {phase}: {task.instance} seed={task.seed} FE={state['fe_index']}", flush=True)
    if local.formal_state_signature(engine) != before:
        raise AssertionError("Offline experiment altered formal FE/RNG/populations/diagnostics")
    return rows


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def distribution(values):
    values = sorted(v for v in values if v is not None)
    return {"count": len(values), "mean": mean(values),
            "median": statistics.median(values) if values else None,
            "Q25": selection.quantile(values, .25) if values else None,
            "Q75": selection.quantile(values, .75) if values else None,
            "max": max(values) if values else None}


def state_metrics(rows, stage):
    episodes = defaultdict(list)
    for row in rows:
        key = (row["instance"], row["seed"], row["fe_index"], row["kind"],
               row["trial"], row.get("budget", 16))
        episodes[key].append(row)
    state_groups = defaultdict(list)
    for key, values in episodes.items():
        if stage == 1:
            gains = [v["gain"] for v in values]
            successful = [g for g in gains if g > 0]
            decoded = sum(v["actual_offline_decodes"] for v in values)
            result = {"EAG": sum(max(0, g) for g in gains)/decoded if decoded else 0,
                      "best16": max(gains), "improve_rate": mean(g > 0 for g in gains),
                      "neutral_rate": mean(g == 0 for g in gains), "worsen_rate": mean(g < 0 for g in gains),
                      "any_improvement": int(bool(successful)),
                      "successful_gain_mean": mean(successful),
                      "successful_gain_median": statistics.median(successful) if successful else None,
                      "Cmax_delta": mean(v["Cmax_delta"] for v in values),
                      "exhausted_rate": mean(v["exhausted"] for v in values), "offline_decodes": decoded}
        else:
            result = {k: values[0][k] for k in ("gain", "gain_per_evaluation", "any_improvement",
                      "accepted_moves", "OS_accepted", "MS_accepted", "time_to_first_evaluation",
                      "best_shortage", "Cmax_delta", "unique_genotypes", "actual_offline_decodes", "exhausted")}
        state_groups[key[:4]+(key[-1],)].append(result)
    out = []
    for key, values in sorted(state_groups.items()):
        result = {"instance": key[0], "seed": key[1], "fe_index": key[2], "kind": key[3],
                  "budget": key[4], "replicates": len(values)}
        for field in values[0]:
            result[field] = mean(z[field] for z in values if z[field] is not None)
        out.append(result)
    return out


COMPARISONS = ((KINDS[0], KINDS[2]), (KINDS[1], KINDS[2]),
               (KINDS[0], KINDS[1]), (KINDS[0], KINDS[3]), (KINDS[1], KINDS[3]))


def paired_rows(states, fields):
    index = {(z["instance"], z["seed"], z["fe_index"], z["kind"], z["budget"]): z for z in states}
    pairs = []
    keys = sorted({(z["instance"], z["seed"], z["fe_index"], z["budget"]) for z in states})
    for instance, seed, fe, budget in keys:
        for left, right in COMPARISONS:
            a = index.get((instance, seed, fe, left, budget))
            b = index.get((instance, seed, fe, right, budget))
            if a is None or b is None:
                continue
            for field in fields:
                if a[field] is None or b[field] is None:
                    continue
                pairs.append({"instance": instance, "seed": seed, "fe_index": fe,
                              "budget": budget, "comparison": left+" vs "+right,
                              "metric": field, "left": a[field], "right": b[field],
                              "difference": a[field]-b[field]})
    return pairs


def group_labels(row):
    return ("overall", row["instance"], f"{row['instance']}/seed{row['seed']}")


def summarize_pairs(pairs):
    groups = defaultdict(list)
    for row in pairs:
        for group in group_labels(row):
            groups[(group, row["comparison"], row["metric"], row["budget"])].append(row)
    out = []
    for (group, comparison, metric, budget), values in sorted(groups.items()):
        diffs = [z["difference"] for z in values]
        trajectory_means = [mean(z["difference"] for z in values if z["instance"] == name and z["seed"] == seed)
                            for name in local.INSTANCES for seed in local.SEEDS]
        trajectory_means = [v for v in trajectory_means if v is not None]
        total = sum(diffs)
        out.append({"group": group, "comparison": comparison, "metric": metric, "budget": budget,
                    "states": len(values), "wins": sum(v > 0 for v in diffs),
                    "ties": sum(v == 0 for v in diffs), "losses": sum(v < 0 for v in diffs),
                    "mean_difference": mean(diffs), "median_difference": statistics.median(diffs),
                    "min_leave_one_state_mean": min((total-v)/(len(diffs)-1) for v in diffs) if len(diffs)>1 else None,
                    "min_leave_one_trajectory_mean": min(mean(v for i,v in enumerate(trajectory_means) if i!=j)
                         for j in range(len(trajectory_means))) if len(trajectory_means)>1 else None})
    return out


def comparison_pass(summary, comparison, metric, budget, robust=False):
    index = {z["group"]: z for z in summary if z["comparison"] == comparison
             and z["metric"] == metric and z["budget"] == budget}
    if "overall" not in index or any(name not in index for name in local.INSTANCES):
        return False
    overall = index["overall"]
    positive = all(index[name]["mean_difference"] > 0 for name in local.INSTANCES)
    groups = sum(z["mean_difference"] > 0 for key, z in index.items() if "/seed" in key)
    ok = positive and overall["wins"] > overall["losses"] and groups >= 4
    if robust:
        ok = ok and (overall["min_leave_one_state_mean"] or 0) > 0 and (overall["min_leave_one_trajectory_mean"] or 0) > 0
    return ok


def stage1_decision(summary):
    passes = lambda left, right: all(comparison_pass(summary, left+" vs "+right, metric, 16)
                                     for metric in ("EAG", "best16"))
    if passes(KINDS[0], KINDS[2]) and passes(KINDS[0], KINDS[1]):
        return "A: strong core evidence"
    if passes(KINDS[1], KINDS[2]) and not passes(KINDS[0], KINDS[1]):
        return "B: support evidence without stable core increment"
    return "STOP: insufficient cross-instance diagnosis-set enrichment"


def save(name, rows, folder="analysis"):
    path = ROOT/folder/name
    if rows:
        local.save_csv(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("status,reason\nNOT_TESTED,Stage 1 gate did not permit restricted search\n")


def grouped_metrics(states, fields):
    groups = defaultdict(list)
    for row in states:
        for group in group_labels(row):
            groups[(group, row["kind"], row["budget"])].append(row)
    return [{"group": group, "kind": kind, "budget": budget, "metric": field,
             **distribution(z[field] for z in values)}
            for (group, kind, budget), values in sorted(groups.items()) for field in fields]


def figures(one_step, search):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = ROOT/"figures"; folder.mkdir(parents=True, exist_ok=True)
    colors = ("#187f83", "#6d79b5", "#a76653")
    for metric in ("EAG", "best16"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, instance in zip(axes, local.INSTANCES):
            for i, kind in enumerate(KINDS[:3]):
                vals = [z[metric] for z in one_step if z["instance"] == instance and z["kind"] == kind]
                ax.bar(i, mean(vals), color=colors[i])
            ax.set(title=instance, ylabel=metric, xticks=range(3), xticklabels=("Core", "Support", "Global"))
        fig.tight_layout(); fig.savefig(folder/f"one_step_{metric}.png", dpi=160); plt.close(fig)
    if search:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, instance in zip(axes, local.INSTANCES):
            for kind, color in zip(KINDS[:3], colors):
                ax.plot(BUDGETS, [mean(z["gain"] for z in search if z["instance"]==instance and z["kind"]==kind and z["budget"]==b) for b in BUDGETS], marker="o", color=color, label=kind)
            ax.set(title=instance, xlabel="Offline decode budget", ylabel="Mean state gain"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(folder/"restricted_search_gain.png", dpi=160); plt.close(fig)
        pairs = paired_rows(search, ("gain",))
        fig, ax = plt.subplots(figsize=(7, 4))
        for kind, color in zip(KINDS[:2], colors):
            vals = [z["difference"] for z in pairs if z["budget"]==32 and z["comparison"]==kind+" vs "+KINDS[2]]
            ax.hist(vals, bins=25, alpha=.5, color=color, label=kind)
        ax.set(xlabel="State-paired gain difference vs global (B32)", ylabel="States"); ax.legend()
        fig.tight_layout(); fig.savefig(folder/"state_paired_gain.png", dpi=160); plt.close(fig)


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Historical diagnosis reproduction requires PYTHONHASHSEED=0")
    fingerprint, payload = preregister()
    tasks = [r.Task(name, "full", seed) for name in local.INSTANCES for seed in local.SEEDS]
    verified = []
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(prepare_task, t, fingerprint, payload) for t in tasks]):
            verified.append(future.result())
    geometry_rows = [z for result in verified for z in result["geometry"]]
    save("state_subset_geometry.csv", geometry_rows, "raw")
    save("subset_membership.csv", [z for result in verified for z in result["members"]], "raw")
    for name in ("restricted_search_runs.csv",):
        save(name, [], "raw")
    for name in ("restricted_search_summary.csv", "restricted_search_state_pairs.csv"):
        save(name, [])
    cores = [z for z in geometry_rows if z["kind"] == KINDS[0]]
    geometry_summary = []
    for field in ("H_size", "V_plus_size", "H_ratio", "V_plus_ratio", "H_support_ratio", "H_mass"):
        for group in ("overall", *local.INSTANCES):
            geometry_summary.append({"group": group, "metric": field, **distribution(z[field] for z in cores if group=="overall" or z["instance"]==group)})
    save("subset_geometry_summary.csv", geometry_summary)
    actionability = []
    geometry_states = defaultdict(list)
    for row in geometry_rows:
        if row["strict_match"]:
            geometry_states[(row["instance"], row["seed"], row["fe_index"], row["kind"])].append(row)
    averaged_geometry = []
    stages = sorted({int(stage) for z in geometry_rows if z["strict_match"] for stage in z["stage_counts"]})
    geometry_fields = ("OS_fraction", "MS_fraction", "both_fraction", "no_action_fraction",
                       "mean_MS_alternatives", "mean_OS_targets", "positive_fraction", "overlap_H")
    for (instance, seed, fe, kind), values in sorted(geometry_states.items()):
        item = {"instance": instance, "seed": seed, "fe_index": fe, "kind": kind, "budget": 0}
        item.update({field: mean(z[field] for z in values) for field in geometry_fields})
        for stage in stages:
            item[f"stage_{stage}_fraction"] = mean(z["stage_counts"].get(stage, z["stage_counts"].get(str(stage), 0))/z["size"] if z["size"] else 0 for z in values)
        averaged_geometry.append(item)
    for kind in KINDS:
        selected = [z for z in averaged_geometry if z["kind"] == kind]
        for group in ("overall", *local.INSTANCES):
            for field in geometry_fields:
                actionability.append({"group": group, "kind": kind, "metric": field,
                                      **distribution(z[field] for z in selected if group=="overall" or z["instance"]==group)})
    save("actionability_summary.csv", actionability)
    save("stage_composition_summary.csv", grouped_metrics(averaged_geometry, tuple(f"stage_{i}_fraction" for i in stages)))
    one_step = []
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(experiment_task, t, fingerprint, "stage1") for t in tasks]):
            one_step.extend(future.result())
    save("one_step_proposals.csv", one_step, "raw")
    one_states = state_metrics(one_step, 1)
    one_pairs = paired_rows(one_states, ("EAG", "best16", "improve_rate", "any_improvement", "Cmax_delta"))
    one_summary = summarize_pairs(one_pairs)
    save("one_step_state_pairs.csv", one_pairs)
    save("one_step_enrichment_summary.csv", one_summary)
    save("one_step_state_metrics.csv", one_states)
    save("one_step_metric_distributions.csv", grouped_metrics(one_states, ("EAG", "best16", "improve_rate", "neutral_rate", "worsen_rate", "any_improvement", "successful_gain_mean", "successful_gain_median", "Cmax_delta", "exhausted_rate")))
    decomposed = []
    for move in ("OS", "MS"):
        states = state_metrics([z for z in one_step if z["move_type"] == move], 1)
        decomposed.extend({"move_type": move, **z} for z in grouped_metrics(states, ("EAG", "improve_rate", "Cmax_delta")))
    save("one_step_OS_MS_secondary.csv", decomposed)
    decision1 = stage1_decision(one_summary)
    summary = {"fingerprint": fingerprint, "formal_FE": 0,
               "fidelity": {"states": len(cores), "seed_decodes": sum(z["seed_decodes"] for z in verified),
                            "genotype_objective_intervals_phi_H_dependencies": "PASS", "formal_state_isolation": "PASS",
                            "new_trajectories": 0, "formal_sources_changed": False},
               "geometry": geometry_summary, "actionability": actionability,
               "Stage1": {"decision": decision1, "paired_groups": one_summary,
                          "offline_decodes": sum(z["actual_offline_decodes"] for z in one_step)},
               "Stage2": {"status": "NOT TESTED", "offline_decodes": 0},
               "statistical_limit": "Descriptive state-paired diagnostics; states nested within six trajectories; no significance claims",
               "recommendation": "NO-GO"}
    search_states = []
    if not decision1.startswith("STOP"):
        search = []
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
            for future in as_completed([pool.submit(experiment_task, t, fingerprint, "stage2") for t in tasks]):
                search.extend(future.result())
        save("restricted_search_runs.csv", search, "raw")
        search_states = state_metrics(search, 2)
        search_pairs = paired_rows(search_states, ("gain", "gain_per_evaluation", "any_improvement", "Cmax_delta"))
        search_summary = summarize_pairs(search_pairs)
        save("restricted_search_state_pairs.csv", search_pairs)
        save("restricted_search_summary.csv", search_summary)
        save("restricted_search_state_metrics.csv", search_states)
        save("restricted_search_metric_distributions.csv", grouped_metrics(search_states, ("gain", "gain_per_evaluation", "any_improvement", "accepted_moves", "time_to_first_evaluation", "best_shortage", "Cmax_delta", "unique_genotypes", "OS_accepted", "MS_accepted", "exhausted")))
        winners = [kind for kind in KINDS[:2] if all(comparison_pass(search_summary, kind+" vs "+KINDS[2], "gain", b, robust=True) for b in (16,32))]
        summary["Stage2"] = {"status": "PASS" if winners else "FAIL", "winning_subsets": winners,
                             "paired_groups": search_summary,
                             "offline_decodes": sum(z["actual_offline_decodes"] for z in search if z["budget"]==32),
                             "Cmax_side_effect": grouped_metrics(search_states, ("Cmax_delta",))}
        summary["recommendation"] = "GO" if winners else "NO-GO"
        summary["failure_localization"] = "None" if winners else "subset enriched but sequential reoptimization fails / cross-instance instability"
    else:
        summary["failure_localization"] = "diagnosis support not stably enriched and/or phi core lacks cross-instance superiority; see state-paired comparisons, not a generic diagnosis failure"
    save("instance_seed_summary.csv", [{"stage": 1, **z} for z in one_summary if "/seed" in z["group"]] +
         ([{"stage": 2, **z} for z in summary["Stage2"]["paired_groups"] if "/seed" in z["group"]] if search_states else []))
    for path, expected in payload["files"].items():
        if role.file_hash(r.PROJECT_ROOT/path) != expected:
            raise AssertionError("Pinned source/input changed during run: "+path)
    summary["source_hashes_rechecked"] = True
    r.write_json_atomic(ROOT/"analysis/subproblem_viability_summary.json", summary)
    report = ["# Diagnosis-Induced Subproblem Viability Audit", "", "## A. Fidelity", json.dumps(summary["fidelity"], indent=2),
              "## B. Diagnosis-set geometry", json.dumps(geometry_summary, indent=2),
              "Actionability and stage counts: actionability_summary.csv and state_subset_geometry.csv. Matching includes H in the all-O pool, and records overlap; no artificial difference is forced.",
              "## C. One-step enrichment", decision1, "See one_step_enrichment_summary.csv: replicate-average first, then state-paired W/T/L; no independent-proposal significance.",
              "## D. Is phi ranking useful?", "Core superiority requires BOTH EAG and best16 across instances/trajectories. Support-only gate never establishes H necessity. Consult core-support paired rows.",
              "## E. Restricted reoptimization", summary["Stage2"]["status"], "Budget 8/16/32 comes from a single frozen-subset episode; no re-diagnosis or Cmax acceptance. If NOT TESTED, prior gate stopped all episodes.",
              "## F. Cross-instance stability", "P2/P3 and all six trajectory groups are separately reported in instance_seed_summary.csv and both comparison summaries.",
              "## G. Makespan side effect", "Secondary distribution only; Cmax_delta>0 is worse. No post-hoc numerical disaster threshold; a statistical gate alone does not certify an acceptable tradeoff.",
              "## H. Failure localization", summary["failure_localization"],
              "## I. Final decision", summary["recommendation"],
              "GO, if reached, means diagnostic subproblem-identification evidence, not formal algorithm performance or authorization for CP/MILP. No gradient/directional claim.",
              "Full/SIS/P_M/MGS/formal P_S/encoder/decoder/defaults unchanged. formal FE=0. No pre-ablation rerun."]
    (ROOT/"analysis/subproblem_viability_interpretation.md").write_text("\n".join(report)+"\n")
    figures(one_states, search_states)
    print(f"DISVA {summary['recommendation']}: Stage1={decision1}; Stage2={summary['Stage2']['status']}", flush=True)


if __name__ == "__main__":
    main()
