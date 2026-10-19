"""CRNA: snapshot-local 2x2 audit, not an LNS implementation.

PYTHONHASHSEED=0 python3 -m experiments.diagnostics.coupled_reoptimization_audit
Completed replay pairs and completed state experiments are resumable. The frozen
Core-PS implementation is imported, never edited. No diagnostic formal FE.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import random
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from experiments import core_ps_variants as variants
from experiments import run_core_ps_integration_gate as gate
from experiments import run_pre_ablation as r
from experiments import sis_local_viability as local
from experiments import sis_selection_aware as selection
from experiments.diagnostics import diagnosis_subproblem_viability as disva
from src.algorithms.wip_graph_dual_population import Individual
from tests.performance_trajectory_regression import population_state, signature

ROOT = r.PROJECT_ROOT / "experiments/results/coupled_reoptimization_audit"
CHECKPOINTS = tuple(range(2000, 10000, 1000))
POLICIES = ("H-Independent", "H-Sequential", "Global-Independent", "Global-Sequential")
BUDGET, TRIALS = 8, 8
CONTRASTS = ("Delta_H_chain", "Delta_seq_region", "Delta_ind_region",
             "Delta_global_chain", "Interaction")
RULES = {
    "primary": gate.PRIMARY, "secondary": gate.SECONDARY, "seeds": gate.SEEDS,
    "checkpoints": CHECKPOINTS, "policies": POLICIES, "budget": BUDGET,
    "trials": TRIALS, "rho": .8, "formal_FE": 0,
    "capture": "First specialized branch with pre-child FE>=checkpoint (child FE>checkpoint), including fallback branches; no outcome selection",
    "replay": "Source-pinned original Core-PS, gate observer ON both controls; CRNA capture default OFF. Full-length capture OFF/ON pair; compare original events/diagnostics/archive, compare paired final populations/RNG/all counters. Original gate did not store final populations/RNG; historical equivalence for those is indirect, not a nonexistent direct comparison.",
    "atomic": "H: imported exact DISVA restricted_basic_mutation; global: original engine.basic_mutation with isolated RNG; no resampling, weights, direction, repair, guides or child selection",
    "acceptance": "Strict shortage decrease only; independent always proposes from seed, earliest strict best retained including seed index0; sequential proposes from incumbent; no tie acceptance",
    "freeze": "phi/H computed once from captured seed; no episode diagnosis or H refresh; no global fallback on exhaustion",
    "RNG": "SHA256(instance,formal_seed,checkpoint,trial_id,CRNA-v1), identical starting seed for all four policies; no forced alignment after divergence",
    "unit": "8 trials -> state; 8 states -> seed; 3 seeds -> instance; 10 unseen instances primary, two seen secondary only",
    "GO": "C1 and C2 instance means>0 and W>L; all leave-one-instance/seed/checkpoint primary means>0; real multi-accepted episodes contribute positive gains; no size/profile group with a negative C1 or C2 mean. Conservative sign-based group gate fixed before outcomes; zero does not count as negative.",
    "NO_GO": "C1 not stable (mean<=0 or W<=L); if C3 positive, preserve H-region evidence",
    "MIXED": "Positive stable C1 but C2/robustness/group/multi-move conditions fail",
    "tolerance": "Exact zero on integer-objective episode gains and rational means; no tuned statistical threshold",
    "H_changed": "H operation whose MS assignment OR relative OS relation to any incomparable operation differs from seed; displacement alone is not sufficient",
    "accepted_independent": "Record-breaking strictly lower-shortage proposals, NOT a chain; final independent genotype always <=1 active atomic change",
    "archive": "Secondary only: no snapshot point strictly Pareto-dominates final; equal point is nondominated, not necessarily a new archive entry",
    "runtime": "Offline decode calls counted including duplicate proposals; replay FE separately, no midrun state resume",
}


def write_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        pickle.dump(value, stream, protocol=4)
    temporary.replace(path)


def read_pickle(path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def register():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 required")
    original = json.loads((gate.ROOT / "preregistration.json").read_text())
    files = dict(original["files"])
    for path in (Path(__file__), r.PROJECT_ROOT / "experiments/tests/test_coupled_reoptimization_audit.py"):
        files[str(path.relative_to(r.PROJECT_ROOT))] = disva.role.file_hash(path)
    artifacts = {}
    for name in gate.PRIMARY + gate.SECONDARY:
        for seed in gate.SEEDS:
            path = gate.run_path(r.Task(name, "core_ps", seed))
            artifacts[str(path.relative_to(r.PROJECT_ROOT))] = disva.role.file_hash(path)
    payload = {"rules": RULES, "files": files, "artifacts": artifacts,
               "gate_fingerprint": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
               "environment": {"python": sys.version, "executable": sys.executable,
                               "platform": platform.platform(), "hash_seed": os.environ["PYTHONHASHSEED"]}}
    payload = json.loads(json.dumps(payload))
    check_sources(payload)
    path = ROOT / "preregistration.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise AssertionError("Preregistration changed; never mix sources or tune outcomes")
    r.write_json_atomic(path, payload)
    return signature(payload), payload


def check_sources(payload):
    for path, expected in {**payload["files"], **payload["artifacts"]}.items():
        if disva.role.file_hash(r.PROJECT_ROOT / path) != expected:
            raise AssertionError("Pinned source/artifact changed: " + path)


def compact_population(population):
    return [{"genome": r.prefix_genome(ind), "Cmax": ind.makespan, "shortage": ind.shortage}
            for ind in population]


class CaptureCorePS(variants.CorePSGateAlgorithm):
    """Read-only capture, default OFF; reuse the diagnosis already computed by Core."""

    def __init__(self, *args, capture=False, checkpoints=CHECKPOINTS, **kwargs):
        self.crna_capture = capture
        self.crna_checkpoints = tuple(checkpoints)
        self.crna_states = []
        self._crna_active = False
        self._crna_diagnosis = None
        self._crna_high = frozenset()
        super().__init__(*args, **kwargs)

    def diagnose_shortage(self, ind):
        result = super().diagnose_shortage(ind)
        if self._crna_active:
            self._crna_diagnosis = result
        return result

    def high_influence_set(self, ind, diagnosis):
        result = super().high_influence_set(ind, diagnosis)
        if self._crna_active:
            self._crna_high = frozenset(result)
        return result

    def _core_child(self, x):
        captured = {s["checkpoint"] for s in self.crna_states}
        due = [cp for cp in self.crna_checkpoints if cp <= self.n_evaluations and cp not in captured]
        if not self.crna_capture or not due:
            return super()._core_child(x)
        self._crna_active = True
        self._crna_diagnosis, self._crna_high = None, frozenset()
        try:
            result = super()._core_child(x)
        finally:
            self._crna_active = False
        snapshot = {"actual_capture_FE": self.n_evaluations, "child_FE": self.n_evaluations + 1,
                    "parent": x, "diagnosis": self._crna_diagnosis, "H": self._crna_high,
                    "archive": compact_population(self.A), "P_S": compact_population(self.P_S),
                    "P_M": compact_population(self.P_M),
                    "P_S_identifier": signature(population_state(self.P_S)),
                    "formal_RNG_identifier": signature(self.rng.getstate()),
                    "fallback_reason": result[1]}
        for cp in due:
            self.crna_states.append(copy.deepcopy({"checkpoint": cp, **snapshot}))
        return result


def final_signature(engine):
    return signature((population_state(engine.P_M), population_state(engine.P_S),
                      population_state(engine.A), engine.rng.getstate(), engine.diagnostics,
                      engine._first_source_by_genotype, engine.round, engine.n_evaluations,
                      engine.unanchored_shortage_intervals, engine.phi_mass_mismatch_count,
                      engine.machine_priority_move_spans))


def replay_path(instance, seed):
    return ROOT / "cache/replays" / instance / f"seed_{seed}.pkl.gz"


def replay_task(instance, seed, fingerprint, payload):
    path = replay_path(instance, seed)
    if path.exists():
        cached = read_pickle(path)
        if cached["fingerprint"] != fingerprint or not cached["fidelity"]["passed"]:
            raise AssertionError("Stale replay")
        return cached["fidelity"]
    check_sources(payload)
    old = gate.load_run(r.Task(instance, "core_ps", seed), payload["gate_fingerprint"])
    if old is None:
        raise AssertionError("Missing original Core-PS run")
    data = json.loads((r.INSTANCE_DIR / f"{instance}.json").read_text())
    signatures, states = [], None
    for enabled in (False, True):
        engine = CaptureCorePS(data["operations"], data["buffers"], core_ps_enabled=True,
                               gate_observer=True, capture=enabled,
                               **gate.settings().algorithm_parameters(seed))
        def progress(e):
            r.write_json_atomic(ROOT / "progress/replay" / instance / f"seed_{seed}.json",
                                {"FE": e.n_evaluations, "capture": enabled, "fingerprint": fingerprint})
        engine.run(diagnostic_callback=progress)
        archive = engine.gate_archive_records()
        diagnostics = {k: engine.diagnostics[k] for k in old["diagnostics"]}
        if (json.loads(json.dumps(engine.gate_events)) != old["events"] or
                json.loads(json.dumps(archive)) != old["final_archive"] or
                json.loads(json.dumps(diagnostics)) != old["diagnostics"] or
                engine.n_evaluations != old["actual_FE"]):
            raise AssertionError(f"Historical replay mismatch: {instance} seed={seed} capture={enabled}")
        signatures.append(final_signature(engine))
        if enabled:
            states = engine.crna_states
        print(f"CRNA replay {'capture' if enabled else 'control'}: {instance} seed={seed} FE={engine.n_evaluations}", flush=True)
    if signatures[0] != signatures[1]:
        raise AssertionError("Capture observer altered final state/RNG/counters")
    found = {s["checkpoint"] for s in states}
    fidelity = {"instance": instance, "seed": seed, "passed": True, "state_count": len(states),
                "missing": sorted(set(CHECKPOINTS) - found), "replay_FE": 20000,
                "historical_events_diagnostics_archive_equal": True,
                "paired_final_population_RNG_counters_equal": True,
                "final_signature": signatures[0], "historical_final_RNG_saved": False}
    for state in states:
        state.update(instance=instance, seed=seed)
    check_sources(payload)
    write_pickle(path, {"fingerprint": fingerprint, "fidelity": fidelity, "states": states})
    return fidelity


def reconstruct(engine, state):
    """One seed decode; exact schedule/provenance/diagnosis/H equality, no formal FE."""
    before = copy.deepcopy((engine.diagnostics, engine.unanchored_shortage_intervals,
                            engine.phi_mass_mismatch_count))
    captured = state["parent"]
    rebuilt = disva.offline_decode(engine.decoder, r.prefix_individual(r.prefix_genome(captured)), full=True)
    for field in ("makespan", "shortage", "schedule", "buffer_trace", "stats", "provenance"):
        if getattr(captured, field) != getattr(rebuilt, field):
            raise AssertionError("Captured seed reconstruction mismatch: " + field)
    try:
        if captured.shortage:
            diagnosis = engine.diagnose_shortage(rebuilt)
            high = engine.high_influence_set(rebuilt, diagnosis)
            if diagnosis != state["diagnosis"] or high != state["H"]:
                raise AssertionError("Captured phi/H reconstruction mismatch")
            if diagnosis.unanchored_intervals or not math.isclose(sum(diagnosis.phi.values()), rebuilt.shortage, abs_tol=1e-9, rel_tol=0):
                raise AssertionError("Seed phi mass/anchor mismatch")
            ordered = sorted(diagnosis.phi, key=lambda op: (-diagnosis.phi[op], op))
            n = len(high)
            if (not n or set(ordered[:n]) != high or
                    sum(diagnosis.phi[op] for op in ordered[:n])/rebuilt.shortage < engine.rho or
                    sum(diagnosis.phi[op] for op in ordered[:n-1])/rebuilt.shortage >= engine.rho):
                raise AssertionError("H coverage/minimality mismatch")
        elif state["H"] or state["diagnosis"] is not None:
            raise AssertionError("Unexpected zero-shortage diagnosis")
    finally:
        engine.diagnostics, engine.unanchored_shortage_intervals, engine.phi_mass_mismatch_count = before
    return rebuilt


def trial_rng(state, trial):
    key = (state["instance"], state["seed"], state["checkpoint"], trial, "CRNA-v1")
    return random.Random(int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest(), "big"))


def global_proposal(engine, parent, rng):
    # Detect exhaustion without RNG draws, then call the exact formal operator.
    _, windows, ms = disva.atomic_space(engine, parent, set(parent.os_seq))
    if not windows and not ms:
        return None, {"exhausted": True, "move_type": "none", "operation": None}
    class ObservedChoices:
        def __init__(self):
            self.values = []

        def choice(self, options):
            value = rng.choice(options)
            self.values.append(value)
            return value

    observed = ObservedChoices()
    child = selection.exact_basic(engine, parent, observed)
    kind, op, target = observed.values
    if kind == "ms":
        meta = {"move_type": "MS", "operation": list(op), "old_machine": parent.ms_map[op], "new_machine": child.ms_map[op]}
    else:
        meta = {"move_type": "OS", "operation": list(op), "target_gap": target,
                "old_position": parent.os_seq.index(op), "realized_position": child.os_seq.index(op)}
    return child, {"exhausted": False, **meta}


def distances(seed, child, high):
    """Explicit incomparable-pair enumeration, independent of engine caches."""
    a, b = {op:i for i,op in enumerate(seed.os_seq)}, {op:i for i,op in enumerate(child.os_seq)}
    ops = seed.os_seq
    changed = {op for op in high if seed.ms_map[op] != child.ms_map[op]}
    inversions, denominator = 0, 0
    for i, op in enumerate(ops):
        for other in ops[i+1:]:
            if op[0] == other[0]:
                continue
            denominator += 1
            if (a[op] < a[other]) != (b[op] < b[other]):
                inversions += 1
                changed.update(x for x in (op,other) if x in high)
    d_os = inversions/denominator if denominator else 0.0
    d_ms = sum(seed.ms_map[op] != child.ms_map[op] for op in ops)/len(ops) if ops else 0.0
    return {"d_OS": d_os, "d_MS": d_ms, "d_total": .5*(d_os+d_ms), "changed_H_operations": len(changed)}


def episode(engine, state, policy, trial, decode=None, propose=None):
    """Frozen H, no diagnosis, one proposal/decode per used slot; no formal methods."""
    if policy not in POLICIES:
        raise ValueError(policy)
    seed = state["parent"]
    high = frozenset(state["H"])
    restricted, sequential = policy.startswith("H-"), policy.endswith("Sequential")
    allowed = high if restricted else frozenset(seed.os_seq)
    rng = trial_rng(state, trial)
    best, accepted, first, used, exhausted = seed, 0, None, 0, None
    os_accepted = ms_accepted = 0
    steps = []
    base = {k: state[k] for k in ("instance", "seed", "checkpoint", "actual_capture_FE")}
    base.update(policy=policy, trial=trial)
    for step in range(1, BUDGET+1):
        parent = best if sequential else seed
        parent_hash = selection.digest(engine.genotype_key(parent))
        if propose is not None:
            child, meta = propose(parent, allowed, rng)
        elif restricted:
            child, meta = disva.restricted_basic_mutation(engine, parent, allowed, rng)
        else:
            child, meta = global_proposal(engine, parent, rng)
        if meta["exhausted"]:
            exhausted = step
            break
        if restricted:
            disva.assert_restricted(seed, child, high, meta) if not sequential else disva.assert_restricted(parent, child, high, meta)
            if ([op for op in seed.os_seq if op not in high] != [op for op in child.os_seq if op not in high] or
                    any(child.ms_map[op] != seed.ms_map[op] for op in seed.ms_map if op not in high)):
                raise AssertionError("Frozen non-H projection changed")
        child = decode(child) if decode is not None else disva.offline_decode(engine.decoder, child)
        used += 1
        improvement = child.shortage < best.shortage
        if improvement:
            best = child
            accepted += 1
            os_accepted += meta["move_type"] == "OS"
            ms_accepted += meta["move_type"] == "MS"
            if first is None:
                first = step
        steps.append({**base, "step": step, "proposal_parent_hash": parent_hash,
                      "proposal_hash": selection.digest(engine.genotype_key(child)),
                      "proposal_genome": r.prefix_genome(child), "S_proposal": child.shortage,
                      "Cmax_proposal": child.makespan, "accepted": improvement,
                      "incumbent_hash": selection.digest(engine.genotype_key(best)),
                      "gain": seed.shortage-best.shortage, **meta})
    gain = seed.shortage-best.shortage
    dominates = best.shortage <= seed.shortage and best.makespan <= seed.makespan and (best.shortage < seed.shortage or best.makespan < seed.makespan)
    archive_nd = not any(p["shortage"] <= best.shortage and p["Cmax"] <= best.makespan and
                         (p["shortage"] < best.shortage or p["Cmax"] < best.makespan) for p in state["archive"])
    summary = {**base, "S_seed": seed.shortage, "Cmax_seed": seed.makespan,
               "S_final": best.shortage, "Cmax_final": best.makespan, "FinalGain": gain,
               "any_improvement": gain>0, "budget_used": used, "exhausted_step": exhausted,
               "gain_per_actual_decode": gain/used if used else 0.0,
               "accepted_moves": accepted, "OS_accepted": os_accepted, "MS_accepted": ms_accepted,
               "first_improvement_step": first, "Cmax_delta": best.makespan-seed.makespan,
               "dominates_seed": dominates, "snapshot_archive_nondominated": archive_nd,
               **distances(seed, best, high)}
    for step in range(1, BUDGET+1):
        summary[f"gain_step_{step}"] = steps[min(step,len(steps))-1]["gain"] if steps else 0
    final = {**base, "genome": r.prefix_genome(best), "shortage": best.shortage, "Cmax": best.makespan}
    return summary, steps, final


def state_task(state, fingerprint, engine):
    path = ROOT / "cache/episodes" / state["instance"] / f"seed_{state['seed']}" / f"checkpoint_{state['checkpoint']}.pkl.gz"
    if path.exists():
        value = read_pickle(path)
        if value["fingerprint"] != fingerprint:
            raise AssertionError("Stale episode cache")
        return value
    before = local.formal_state_signature(engine)
    rebuilt = reconstruct(engine, state)
    state = {**state, "parent": rebuilt}
    high, diag = state["H"], state["diagnosis"]
    total = sum(diag.phi.values()) if diag else 0
    mass = sum(diag.phi[op] for op in high) if diag else 0
    geometry = {k:state[k] for k in ("instance","seed","checkpoint","actual_capture_FE","child_FE","P_S_identifier","fallback_reason")}
    _, windows, ms = disva.atomic_space(engine, rebuilt, high)
    geometry.update(S_seed=rebuilt.shortage,Cmax_seed=rebuilt.makespan,O_size=len(rebuilt.os_seq),
                    H_size=len(high),H_ratio=len(high)/len(rebuilt.os_seq),H_mass=mass,
                    phi_mass=total,H_coverage=mass/rebuilt.shortage if rebuilt.shortage else None,
                    H_OS_actionable=len(windows),H_MS_actionable=len(ms),
                    H=[list(op) for op in sorted(high)], phi=[[list(op),v] for op,v in sorted(diag.phi.items())] if diag else [])
    episodes, steps, finals = [], [], []
    for trial in range(TRIALS):
        for policy in POLICIES:
            summary, records, final = episode(engine, state, policy, trial)
            episodes.append(summary); steps.extend(records); finals.append(final)
    if local.formal_state_signature(engine) != before:
        raise AssertionError("Offline experiments altered formal FE/RNG/populations/archive/counters")
    value = {"fingerprint":fingerprint,"seed_decodes":1,"child_decodes":sum(e["budget_used"] for e in episodes),
             "geometry":geometry,"episodes":episodes,"steps":steps,"finals":finals,"fidelity_passed":True}
    write_pickle(path,value)
    return value


def offline_task(instance, seed, fingerprint):
    captured = read_pickle(replay_path(instance,seed))
    if captured["fingerprint"] != fingerprint:
        raise AssertionError("Stale capture source")
    data = json.loads((r.INSTANCE_DIR / f"{instance}.json").read_text())
    engine = variants.CorePSGateAlgorithm(data["operations"],data["buffers"],core_ps_enabled=True,
                                         **gate.settings().algorithm_parameters(seed))
    counts = []
    for state in captured["states"]:
        result = state_task(state,fingerprint,engine)
        counts.append(result["child_decodes"])
    print(f"CRNA offline complete: {instance} seed={seed} states={len(counts)} child_decodes={sum(counts)}",flush=True)
    return sum(counts)


def csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    serial = [{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()} for row in rows]
    r.write_csv(path, fields, serial)


def avg(rows, field):
    return statistics.mean(row[field] for row in rows) if rows else 0.0


def grouped(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys,key)), **{field:avg(values,field) for field in fields}, "n_children_units":len(values)}
            for key,values in sorted(groups.items())]


def compare(rows, field):
    values = [row[field] for row in rows]
    return {"mean":statistics.mean(values) if values else 0, "median":statistics.median(values) if values else 0,
            "W":sum(x>0 for x in values), "T":sum(x==0 for x in values), "L":sum(x<0 for x in values),"n":len(values)}


def state_contrasts(state_policy):
    groups = defaultdict(dict)
    for row in state_policy:
        groups[(row["instance"],row["seed"],row["checkpoint"])][row["policy"]] = row["FinalGain"]
    rows = []
    for key,values in sorted(groups.items()):
        hi,hs,gi,gs = (values[p] for p in POLICIES)
        rows.append({**dict(zip(("instance","seed","checkpoint"),key)),
                     **{p:values[p] for p in POLICIES}, "Delta_H_chain":hs-hi,
                     "Delta_seq_region":hs-gs,"Delta_ind_region":hi-gi,
                     "Delta_global_chain":gs-gi,"Interaction":(hs-hi)-(gs-gi)})
    return rows


def classification(instances, seed_rows, state_rows, episodes):
    primary = [row for row in instances if row["instance"] in gate.PRIMARY]
    c1,c2 = (compare(primary,field) for field in CONTRASTS[:2])
    stable1 = c1["mean"]>0 and c1["W"]>c1["L"]
    stable2 = c2["mean"]>0 and c2["W"]>c2["L"]
    robust = []
    for name in gate.PRIMARY:
        subset=[row for row in primary if row["instance"]!=name]
        robust.append({"omitted_type":"instance","omitted":name,**{f:avg(subset,f) for f in CONTRASTS}})
    for kind,levels in (("seed",gate.SEEDS),("checkpoint",CHECKPOINTS)):
        for level in levels:
            kept=[row for row in state_rows if row["instance"] in gate.PRIMARY and row[kind]!=level]
            by_seed=grouped(kept,("instance","seed"),CONTRASTS)
            by_instance=grouped(by_seed,("instance",),CONTRASTS)
            robust.append({"omitted_type":kind,"omitted":level,**{f:avg(by_instance,f) for f in CONTRASTS}})
    strata=[]
    for kind,index in (("size",1),("profile",2)):
        for level in sorted({row["instance"].split("_")[index] for row in primary}):
            subset=[row for row in primary if row["instance"].split("_")[index]==level]
            strata.append({"kind":kind,"level":level,"n_instances":len(subset),**{f:avg(subset,f) for f in CONTRASTS}})
    multi=[e for e in episodes if e["instance"] in gate.PRIMARY and e["policy"]=="H-Sequential" and e["accepted_moves"]>=2]
    checks={"C1_stable":stable1,"C2_stable":stable2,
            "all_leave_one_directions_positive":all(row[f]>0 for row in robust for f in CONTRASTS[:2]),
            "no_negative_size_profile_means":all(row[f]>=0 for row in strata for f in CONTRASTS[:2]),
            "actual_multi_move_gain":bool(multi) and sum(e["FinalGain"] for e in multi)>0}
    label="NO-GO" if not stable1 else "GO" if all(checks.values()) else "MIXED"
    return label,checks,robust,strata


def analyze(fingerprint,payload):
    check_sources(payload)
    replays,results,manifest=[],[],[]
    for name in gate.PRIMARY+gate.SECONDARY:
        for seed in gate.SEEDS:
            replay=read_pickle(replay_path(name,seed))
            if replay["fingerprint"]!=fingerprint:
                raise AssertionError("Stale replay during analysis")
            replays.append(replay["fidelity"])
            found={s["checkpoint"]:s for s in replay["states"]}
            for cp in CHECKPOINTS:
                row={"instance":name,"seed":seed,"checkpoint":cp,"missing":cp not in found}
                if cp in found:
                    state=found[cp]
                    row.update(actual_capture_FE=state["actual_capture_FE"],child_FE=state["child_FE"],
                               capture_path=str(replay_path(name,seed).relative_to(ROOT)))
                    path=ROOT/"cache/episodes"/name/f"seed_{seed}"/f"checkpoint_{cp}.pkl.gz"
                    result=read_pickle(path)
                    if result["fingerprint"]!=fingerprint or not result["fidelity_passed"]:
                        raise AssertionError("Invalid episode cache")
                    results.append(result)
                manifest.append(row)
    episodes=[e for result in results for e in result["episodes"]]
    steps=[e for result in results for e in result["steps"]]
    finals=[e for result in results for e in result["finals"]]
    geometry=[result["geometry"] for result in results]
    fields=("FinalGain","any_improvement","Cmax_delta","budget_used","gain_per_actual_decode",
            "accepted_moves","OS_accepted","MS_accepted","d_OS","d_MS","d_total","changed_H_operations",
            "dominates_seed","snapshot_archive_nondominated",*[f"gain_step_{i}" for i in range(1,9)])
    state_policy=grouped(episodes,("instance","seed","checkpoint","policy"),fields)
    seed_policy=grouped(state_policy,("instance","seed","policy"),fields)
    instance_policy=grouped(seed_policy,("instance","policy"),fields)
    contrasts=state_contrasts(state_policy)
    seed_contrasts=grouped(contrasts,("instance","seed"),(*POLICIES,*CONTRASTS))
    instance_contrasts=grouped(seed_contrasts,("instance",),(*POLICIES,*CONTRASTS))
    decision,checks,robust,strata=classification(instance_contrasts,seed_contrasts,contrasts,episodes)
    policy_summary=[]
    for scope,names in (("primary_unseen",gate.PRIMARY),("secondary_seen",gate.SECONDARY)):
        for policy in POLICIES:
            rows=[row for row in instance_policy if row["instance"] in names and row["policy"]==policy]
            policy_summary.append({"set":scope,"policy":policy,"n_instances":len(rows),**{f:avg(rows,f) for f in fields}})
    accepted=[]
    for scope,names in (("primary_unseen",gate.PRIMARY),("secondary_seen",gate.SECONDARY)):
        for policy in POLICIES:
            rows=[e for e in episodes if e["instance"] in names and e["policy"]==policy]
            improved=[e for e in rows if e["FinalGain"]>0]
            multi=[e for e in rows if e["accepted_moves"]>=2]
            accepted.append({"set":scope,"policy":policy,"episodes":len(rows),
                             "count_distribution":dict(sorted(Counter(e["accepted_moves"] for e in rows).items())),
                             "P_accepted_ge2":len(multi)/len(rows) if rows else 0,
                             "P_accepted_ge3":sum(e["accepted_moves"]>=3 for e in rows)/len(rows) if rows else 0,
                             "among_improved_ge2":len(multi)/len(improved) if improved else 0,
                             "gain_fraction_ge2":sum(e["FinalGain"] for e in multi)/sum(e["FinalGain"] for e in rows) if improved else 0,
                             "warning":"Independent counts are record updates, not accepted chains" if "Independent" in policy else "Restricted sequential local search, not LNS"})
    contribution=[]
    indexed={(e["instance"],e["seed"],e["checkpoint"],e["trial"],e["policy"]):e for e in episodes}
    for bucket in (0,1,2):
        rows=[e for e in episodes if e["instance"] in gate.PRIMARY and e["policy"]=="H-Sequential" and min(e["accepted_moves"],2)==bucket]
        differences=[e["FinalGain"]-indexed[(e["instance"],e["seed"],e["checkpoint"],e["trial"],"H-Independent")]["FinalGain"] for e in rows]
        contribution.append({"H_Sequential_accepted_bucket":"2+" if bucket==2 else str(bucket),"episodes":len(rows),
                             "paired_C1_sum":sum(differences),"paired_C1_mean":statistics.mean(differences) if differences else 0})
    fidelity={"fingerprint":fingerprint,"passed":all(x["passed"] for x in replays),"replays":replays,
              "replay_FE":sum(x["replay_FE"] for x in replays),"formal_FE":0,
              "seed_reconstruction_decodes":sum(x["seed_decodes"] for x in results),
              "offline_child_decodes":sum(x["child_decodes"] for x in results),
              "historical_limitation":"Original final P_M/P_S/RNG were not saved. Exact historical equality is checked for all recorded child events/counters/archive; complete final states and RNG are compared between new full-length OFF/ON replays, not directly against missing historical data."}
    summary={"fingerprint":fingerprint,"classification":decision,"checks":checks,"formal_FE":0,
             "state_count":len(results),"missing_states":sum(row["missing"] for row in manifest),
             "episode_count":len(episodes),"offline_child_decodes":fidelity["offline_child_decodes"],
             "seed_reconstruction_decodes":fidelity["seed_reconstruction_decodes"],"replay_FE":fidelity["replay_FE"],
             "exhausted_episodes":sum(e["exhausted_step"] is not None for e in episodes),
             "primary_contrasts":{f:compare([x for x in instance_contrasts if x["instance"] in gate.PRIMARY],f) for f in CONTRASTS},
             "policy_summary":policy_summary,"accepted_move_summary":accepted,"multi_move_contribution":contribution,
             "size_profile":strata,"source_hashes_rechecked":True,"sources":payload["files"],
             "interpretation_scope":"H-Sequential is restricted sequential local search, not LNS. Positive evidence supports only proceeding to a genuine relaxed-set/destroy-repair prototype; no such mechanism is implemented here."}
    raw,analysis=ROOT/"raw",ROOT/"analysis"
    for name,rows in (("captured_states_manifest",manifest),("state_geometry",geometry),("episode_summary",episodes),
                      ("step_records",steps),("final_solutions",finals)):
        csv(raw/f"{name}.csv",rows)
    r.write_json_atomic(raw/"replay_fidelity.json",fidelity)
    for name,rows in (("policy_summary",policy_summary),("state_pairwise_contrasts",contrasts),
                      ("instance_seed_summary",seed_contrasts),("instance_mean_summary",instance_contrasts),
                      ("size_profile_summary",strata),("accepted_move_summary",accepted),
                      ("multi_move_contribution",contribution),("leave_one_instance_out",robust)):
        csv(analysis/f"{name}.csv",rows)
    curves=[{"set":row["set"],"policy":row["policy"],"step":i,"mean_gain":row[f"gain_step_{i}"]}
            for row in policy_summary for i in range(1,9)]
    csv(analysis/"step_gain_curves.csv",curves)
    csv(analysis/"structural_change_summary.csv",[{k:row[k] for k in ("set","policy","d_OS","d_MS","d_total","changed_H_operations")} for row in policy_summary])
    csv(analysis/"archive_side_effect_summary.csv",[{k:row[k] for k in ("set","policy","Cmax_delta","dominates_seed","snapshot_archive_nondominated")} for row in policy_summary])
    r.write_json_atomic(analysis/"coupled_reoptimization_summary.json",summary)
    report(summary,geometry,instance_contrasts,robust,fidelity)
    figures(instance_policy,contrasts,curves,episodes)
    return summary


def report(summary,geometry,instances,robust,fidelity):
    lines=["# CRNA：耦合重优化必要性审计", "",f"## A. Fidelity\n\n结论：**{summary['classification']}**。诊断 formal FE=0；状态 {summary['state_count']}，缺失 {summary['missing_states']}；离线子代解码 {summary['offline_child_decodes']}，seed 重建 {summary['seed_reconstruction_decodes']}；重放 FE={summary['replay_FE']}（单独计账）。",
           "捕获器默认关闭，无 RNG/额外正式评价。原记录中逐子代事件、目标、诊断计数及档案精确一致；完整最终种群、计数和 RNG 在配对 OFF/ON 重放中一致。原历史运行没有保存完整最终种群/RNG，因此不能声称直接与不存在的历史字段比较。详见 raw/replay_fidelity.json 和 preregistration.json 源码哈希。",
           "## B. State characteristics", "\n|组|状态数|H均值|H比例均值|seed shortage均值|\n|---|---:|---:|---:|---:|"]
    for names,label in ((gate.PRIMARY,"unseen"),(gate.SECONDARY,"seen")):
        rows=[g for g in geometry if g["instance"] in names]
        lines.append(f"|{label}|{len(rows)}|{avg(rows,'H_size'):.3f}|{avg(rows,'H_ratio'):.4f}|{avg(rows,'S_seed'):.3f}|")
    lines.extend(["\n## C. Four-policy result","\n主分析：trial→state→seed→instance 逐层平均，10 个 unseen 实例等权。", "\n|策略|FinalGain|P(improve)|Cmax delta|\n|---|---:|---:|---:|"])
    for row in summary["policy_summary"]:
        if row["set"]=="primary_unseen":
            lines.append(f"|{row['policy']}|{row['FinalGain']:.4f}|{row['any_improvement']:.4f}|{row['Cmax_delta']:.4f}|")
    for title,field in (("D. Coupling increment","Delta_H_chain"),("E. Diagnosis value under sequential search","Delta_seq_region"),("F. Factorial interaction","Interaction")):
        row=summary["primary_contrasts"][field]
        lines.append(f"\n## {title}\n\n{field}: mean={row['mean']:.6f}, median={row['median']:.6f}, W/T/L={row['W']}/{row['T']}/{row['L']}。正值有利于 H-Sequential；Interaction 只是描述性支持，不是独立硬门槛或因果证据。")
    lines.append("\nC3/C4："+json.dumps({k:summary["primary_contrasts"][k] for k in CONTRASTS[2:4]},ensure_ascii=False))
    lines.append("\n## G. Multi-move evidence\n\nIndependent 的 accepted_moves 只是从同一 seed 抽样时的纪录刷新次数，绝不能当作多动作链。Sequential 才是连续接受。以下为 episode 描述，不将 trials 当成独立实例。")
    for row in summary["accepted_move_summary"]:
        if row["set"]=="primary_unseen" and row["policy"]=="H-Sequential":
            lines.append("\n"+json.dumps(row,ensure_ascii=False))
    lines.append("\n按 H-Sequential 接受 0/1/2+ 次分组的同 trial C1 收益分解（描述性选择后分组，非因果）："+json.dumps(summary["multi_move_contribution"],ensure_ascii=False))
    lines.append("\n## H. Size / profile\n\n|组|C1|C2|C3|C4|Interaction|\n|---|---:|---:|---:|---:|---:|")
    for row in summary["size_profile"]:
        lines.append("|"+row["level"]+"|"+"|".join(f"{row[f]:.5f}" for f in CONTRASTS)+"|")
    lines.append("\n这里检验局部连续搜索增量，不能据此声称已修复 Core-PS 正式算法在 S1 的 Pareto 表现。S8 unseen 仅含 P1/P4，规模/profile 非全因子平衡。")
    lines.append("\n## I. Seen secondary\n\n|实例|C1|C2|C3|C4|Interaction|\n|---|---:|---:|---:|---:|---:|")
    for row in instances:
        if row["instance"] in gate.SECONDARY:
            lines.append("|"+row["instance"]+"|"+"|".join(f"{row[f]:.5f}" for f in CONTRASTS)+"|")
    lines.append("\nS8_P3_B2 的结果仅为机制一致性检查，不能挽救主分析，也不能直接等同于正式集成成功。")
    lines.append("\n## J. Makespan / archive side effect\n\nCmax delta 正数表示恶化。最终解相对快照档案非支配不等于正式新增档案贡献；所有副作用均未参与 proposal、acceptance 或 H 选择。详见 archive_side_effect_summary.csv 和逐 episode 数据。")
    lines.append("\n## K. Failure localization\n\n预注册检查："+json.dumps(summary["checks"],ensure_ascii=False)+f"。exhausted episodes={summary['exhausted_episodes']}，未进行 global fallback。逐实例、逐 seed、逐 checkpoint 留一结果见 leave_one_instance_out.csv。")
    for field in CONTRASTS[:2]:
        lines.append(f"\n{field} 最小留一均值={min(row[field] for row in robust):.6f}。")
    lines.append("\nC1 不稳定：不支持增加联合重优化复杂度，并不推翻 H；C1 正但 C2 失败：连续搜索有用但 H 不是更优区域；分组反转：说明规模/profile 边界，不能立即设计自适应触发。")
    lines.append(f"\n## L. Final classification\n\n**{summary['classification']}**。")
    if summary["classification"]=="GO":
        lines.append("当前证据支持进入真正的诊断引导大邻域搜索原型设计；下一阶段由 H 定义松弛变量集合，并正式实现破坏/松弛 + 修复/重优化。本轮 H-Sequential 仅是 restricted sequential local search，未实现 LNS，也未证明任意多决策联合最优性。")
    else:
        lines.append("当前结果尚不能无条件进入 H-based LNS：应按 C1/C2、分组、稳健性和多动作证据定位限制，不扩大 H 或调预算挽救结论。")
    lines.append("\n本轮未实现 LNS/CP/MILP/ALNS，未恢复 SIS/Lambda 方向路线；P_M/MGS、Core-PS、archive、encoder/decoder、rho=.8、p_mu=.1 均未修改。formal FE=0，重放开销单独报告。")
    (ROOT/"analysis/coupled_reoptimization_interpretation.md").write_text("\n".join(lines)+"\n")


def figures(instance_policy,contrasts,curves,episodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=ROOT/"figures";out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,5))
    for name in gate.PRIMARY:
        by={row["policy"]:row["FinalGain"] for row in instance_policy if row["instance"]==name}
        ax.plot(range(4),[by[p] for p in POLICIES],marker="o",label=name.replace("WIPHFSP_",""))
    ax.set_xticks(range(4),POLICIES,rotation=12);ax.set_ylabel("Instance-mean FinalGain");ax.legend(fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(out/"instance_paired_gains.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    rows=[row for row in contrasts if row["instance"] in gate.PRIMARY]
    ax.scatter([row["H-Independent"] for row in rows],[row["H-Sequential"] for row in rows],s=15,alpha=.6)
    upper=max(max(row[p] for row in rows) for p in POLICIES[:2]);ax.plot([0,upper],[0,upper],color="black",lw=.7)
    ax.set_xlabel("H-Independent state-mean gain");ax.set_ylabel("H-Sequential state-mean gain")
    fig.tight_layout();fig.savefig(out/"state_paired_gains.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for policy in ("H-Sequential","Global-Sequential"):
        rows=[row for row in curves if row["set"]=="primary_unseen" and row["policy"]==policy]
        ax.plot([row["step"] for row in rows],[row["mean_gain"] for row in rows],marker="o",label=policy)
    ax.set_xlabel("Decode slot (maximum 8)");ax.set_ylabel("Instance-mean cumulative gain");ax.legend()
    fig.tight_layout();fig.savefig(out/"sequential_step_gains.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for policy in ("H-Sequential","Global-Sequential"):
        rows=[e for e in episodes if e["instance"] in gate.PRIMARY and e["policy"]==policy]
        ax.scatter([e["accepted_moves"] for e in rows],[e["FinalGain"] for e in rows],s=7,alpha=.2,label=policy)
    ax.set_xlabel("Accepted moves");ax.set_ylabel("FinalGain");ax.legend()
    fig.tight_layout();fig.savefig(out/"accepted_moves_gain.png",dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("all","capture","offline","analyze"),default="all")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:
        parser.error("workers must be positive")
    fingerprint,payload=register()
    tasks=[(name,seed) for name in gate.PRIMARY+gate.SECONDARY for seed in gate.SEEDS]
    for stage in (("capture","offline") if args.mode=="all" else (args.mode,)):
        if stage not in ("capture","offline"):
            continue
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context("spawn")) as pool:
            futures=[pool.submit(replay_task,name,seed,fingerprint,payload) if stage=="capture" else
                     pool.submit(offline_task,name,seed,fingerprint) for name,seed in tasks]
            for future in as_completed(futures):
                future.result()
        check_sources(payload)
    if args.mode in ("all","analyze"):
        result=analyze(fingerprint,payload)
        print(json.dumps({k:result[k] for k in ("classification","state_count","offline_child_decodes","replay_FE","formal_FE")}),flush=True)


if __name__=="__main__":
    main()
