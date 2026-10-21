"""L16 parameter tuning. No experiment, import of Full, or plot runs on import.

Run and offline plot modes are intentionally independent. Resume is task-level.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import re
import statistics as st
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

# ============================================================
# USER CONFIGURATION
# ============================================================
MODE = "run"  # "run" or "plot"
ROUND_NAME = "round1"
FE_MAX = 40000
SEEDS = [0, 1, 2, 3, 4]
MAX_WORKERS = 5
INSTANCE_NAMES = [
    "WIPHFSP_S1_P1_B1", "WIPHFSP_S1_P2_B2", "WIPHFSP_S1_P3_B3", "WIPHFSP_S1_P4_B1",
    "WIPHFSP_S4_P1_B2", "WIPHFSP_S4_P2_B3", "WIPHFSP_S4_P3_B1", "WIPHFSP_S4_P4_B2",
    "WIPHFSP_S8_P1_B3", "WIPHFSP_S8_P2_B1", "WIPHFSP_S8_P3_B2", "WIPHFSP_S8_P4_B3",
]
FACTOR_LEVELS = {
    "N": [20, 40, 60, 80],
    "p_mut": [0.05, 0.15, 0.25, 0.35],
    "rho": [0.50, 0.65, 0.80, 0.95],
    "T_coop": [5, 15, 25, 35],
}
ARCHIVE_RATIO = 2
HV_REFERENCE_POINT = (1.1, 1.1)
HV_NORMALIZATION_FACTOR = 1.21
FAIR_ARCHIVE_LIMIT = 40
SKIP_COMPLETED = True
INSTANCE_DIR = "data/final_benchmark/instances"
RESULTS_DIR = "experiments/results/taguchi_tuning"
# ============================================================
# END USER CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FACTORS = ("N", "p_mut", "rho", "T_coop")
LETTERS = ("A", "B", "C", "D")
CONFIG_NAMES = {
    "MODE", "ROUND_NAME", "FE_MAX", "SEEDS", "MAX_WORKERS", "INSTANCE_NAMES",
    "FACTOR_LEVELS", "ARCHIVE_RATIO", "HV_REFERENCE_POINT", "HV_NORMALIZATION_FACTOR",
    "FAIR_ARCHIVE_LIMIT", "SKIP_COMPLETED", "INSTANCE_DIR", "RESULTS_DIR",
}
SOURCES = ("src/algorithms/wip_graph_dual_population.py",
           "src/solution/encoder.py", "src/solution/decoder.py")
SCHEMA = "hlns-taguchi-l16-v1"
EPS = 1e-12
METRICS = ("raw_hv", "nhv", "nhv40", "gd", "igd", "best_makespan",
           "best_shortage", "archive_size", "runtime_seconds")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    # Switching MODE/workers must not invalidate a scientifically identical round.
    # Semantic configuration is pinned separately in the experiment fingerprint.
    tree = ast.parse(Path(__file__).read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in CONFIG_NAMES
                                             for t in node.targets))]
    return {"sources": {p: file_sha(ROOT/p) for p in SOURCES},
            "runner_logic_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
            "python": platform.python_version(), "python_hash_seed": "0"}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def l16_levels():
    """Four columns of OA(16,4,4,2) over GF(4); symbols encoded as 0..3."""
    times_two = (0, 2, 3, 1)  # GF(4), modulus x^2+x+1.
    rows = [(a+1, b+1, (a ^ b)+1, (a ^ times_two[b])+1)
            for a in range(4) for b in range(4)]
    validate_orthogonality(rows)
    return rows


def validate_orthogonality(rows):
    if len(rows) != 16 or any(len(r) != 4 for r in rows) or len(set(rows)) != 16:
        raise ValueError("L16 must have sixteen distinct four-factor rows")
    for j in range(4):
        if Counter(r[j] for r in rows) != Counter({v: 4 for v in range(1, 5)}):
            raise ValueError("Each factor level must occur four times")
    for j, k in itertools.combinations(range(4), 2):
        if Counter((r[j], r[k]) for r in rows) != Counter(itertools.product(range(1, 5), repeat=2)):
            raise ValueError("Factor pairs are not orthogonal")


def orthogonal_array(levels=None, ratio=None):
    levels = FACTOR_LEVELS if levels is None else levels
    ratio = ARCHIVE_RATIO if ratio is None else ratio
    result = []
    for idx, row in enumerate(l16_levels(), 1):
        values = {factor: levels[factor][v-1] for factor, v in zip(FACTORS, row)}
        result.append({"run_id": f"{idx:02d}", **{letter+"_level": v for letter, v in zip(LETTERS, row)},
                       **values, "N_A": ratio*values["N"]})
    if len({tuple(r[k] for k in FACTORS) for r in result}) != len(result):
        raise ValueError("Duplicate actual parameter combinations")
    return result


def dimensions(name):
    match = re.fullmatch(r"WIPHFSP_(S\d+)_(P\d+)_(B\d+)", name)
    if not match:
        raise ValueError(f"Invalid benchmark ID: {name}")
    return dict(zip(("size", "profile", "buffer"), match.groups()))


def pre_ablation_instances():
    tree = ast.parse((ROOT/"experiments/run_pre_ablation.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SELECTED_INSTANCES"
                                               for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Cannot read pre-ablation instance configuration")


def validate_configuration():
    if MODE not in ("run", "plot"):
        raise ValueError("MODE must be run or plot")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", ROUND_NAME):
        raise ValueError("ROUND_NAME must be a single safe directory name")
    if type(FE_MAX) is not int or FE_MAX < 1 or type(MAX_WORKERS) is not int or MAX_WORKERS < 1:
        raise ValueError("FE_MAX and MAX_WORKERS must be positive integers")
    if ARCHIVE_RATIO != 2 or type(ARCHIVE_RATIO) is not int:
        raise ValueError("This design requires N_A=2N")
    if type(FAIR_ARCHIVE_LIMIT) is not int or FAIR_ARCHIVE_LIMIT < 2:
        raise ValueError("Fair archive cap must protect both extremes (>=2)")
    if tuple(HV_REFERENCE_POINT) != (1.1, 1.1) or not math.isclose(HV_NORMALIZATION_FACTOR, 1.21):
        raise ValueError("Expected reference=(1.1,1.1), NHV divisor=1.21")
    if type(SKIP_COMPLETED) is not bool:
        raise ValueError("SKIP_COMPLETED must be boolean")
    for values, label in ((SEEDS, "seeds"), (INSTANCE_NAMES, "instances")):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Nonempty unique {label} required")
    if any(type(seed) is not int for seed in SEEDS):
        raise ValueError("Integer seeds required")
    if set(FACTOR_LEVELS) != set(FACTORS):
        raise ValueError("Exactly four specified factors required")
    for factor in FACTORS:
        values = FACTOR_LEVELS[factor]
        if len(values) != 4 or len(set(values)) != 4:
            raise ValueError(f"{factor} must have four distinct levels")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("Finite numeric levels required")
        if values != sorted(values):
            raise ValueError("Levels must be ascending for interpretable boundary trends")
        if factor in ("N", "T_coop") and any(type(v) is not int or v < 1 for v in values):
            raise ValueError("N and T_coop levels must be positive integers")
    if min(FACTOR_LEVELS["p_mut"]) < 0 or max(FACTOR_LEVELS["p_mut"]) > 1:
        raise ValueError("p_mut must be in [0,1]")
    if min(FACTOR_LEVELS["rho"]) <= 0 or max(FACTOR_LEVELS["rho"]) > 1:
        raise ValueError("rho must be in (0,1]")
    if FE_MAX < 2*max(FACTOR_LEVELS["N"]):
        raise ValueError("FE budget must cover 2N initialization")
    for value in (INSTANCE_DIR, RESULTS_DIR):
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("Use project-relative, non-traversing directories")
    results = (ROOT/RESULTS_DIR).resolve()
    protected = (ROOT/"experiments/results/taguchi_tuning").resolve()
    if results != protected and protected not in results.parents:
        raise ValueError("Results must stay inside experiments/results/taguchi_tuning")
    missing = []
    for name in INSTANCE_NAMES:
        dimensions(name)
        if not (ROOT/INSTANCE_DIR/(name+".json")).is_file():
            missing.append(name)
    if missing:
        raise FileNotFoundError("Missing instances: " + ", ".join(missing))
    return orthogonal_array()


def experiment_config(array):
    return {"schema": SCHEMA, "round": ROUND_NAME, "FE_max": FE_MAX,
            "seeds": list(SEEDS), "instances": list(INSTANCE_NAMES),
            "factor_levels": FACTOR_LEVELS, "archive_ratio": ARCHIVE_RATIO,
            "fair_archive_limit": FAIR_ARCHIVE_LIMIT, "hv_reference_point": list(HV_REFERENCE_POINT),
            "hv_normalization_factor": HV_NORMALIZATION_FACTOR, "orthogonal_array": array,
            "instance_dir": INSTANCE_DIR,
            "instance_hashes": {name: file_sha(ROOT/INSTANCE_DIR/(name+".json")) for name in INSTANCE_NAMES},
            "source_identity": source_identity(),
            "matches_pre_ablation_instances": INSTANCE_NAMES == pre_ablation_instances()}


def build_tasks(config):
    return [{"run_id": row["run_id"], "instance": name, "seed": seed,
             "N": row["N"], "N_A": row["N_A"], "p_mut": row["p_mut"],
             "rho": row["rho"], "T_coop": row["T_coop"], "FE_max": config["FE_max"],
             "algorithm_variant": "HLNS_FULL", "round": config["round"]}
            for row in config["orthogonal_array"] for name in config["instances"] for seed in config["seeds"]]


def raw_path(root, task):
    return root/"raw"/("run_"+task["run_id"])/task["instance"]/f'seed_{task["seed"]}.json'


def task_source(config, task):
    return {**config["source_identity"], "instance_sha256": config["instance_hashes"][task["instance"]]}


def nondominated(points):
    front, best_y = [], math.inf
    for x, y in sorted(set(map(tuple, points))):
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def validate_record(record, task, config):
    if any(record.get(k) != v for k, v in task.items()):
        raise ValueError("Task identity or parameter mismatch")
    if record.get("experiment_fingerprint") != digest(config) or record.get("source_identity") != task_source(config, task):
        raise ValueError("Round/configuration/source fingerprint mismatch")
    if record.get("actual_parameters") != {k: task[k] for k in ("N", "N_A", "p_mut", "rho", "T_coop", "FE_max", "seed")}:
        raise ValueError("Actual constructor parameters differ")
    if record.get("actual_FE") != task["FE_max"]:
        raise ValueError("Incomplete FE budget")
    if record.get("population_sizes") != {"P_M": task["N"], "P_S": task["N"]}:
        raise ValueError("Population sizes differ")
    if record.get("mechanisms") != ["MGS", "HLNS", True, False]:
        raise ValueError("Not the frozen Full algorithm")
    points = record.get("final_archive_objectives")
    if not isinstance(points, list) or not points or len(points) > task["N_A"]:
        raise ValueError("Empty or oversized archive")
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in point):
            raise ValueError("Invalid objective vector")
    if record.get("final_archive_size") != len(points):
        raise ValueError("Archive size mismatch")
    if nondominated(points) != sorted(set(map(tuple, points))):
        raise ValueError("Archive contains dominated objective vectors")
    seconds = record.get("runtime_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid runtime")


def scan_records(root, tasks, config):
    expected = {raw_path(root, t).resolve(): t for t in tasks}
    extras = sorted(str(p.relative_to(root)) for p in (root/"raw").rglob("*.json")
                    if p.resolve() not in expected)
    if extras:
        raise ValueError("Unexpected/duplicate raw files (resolve before analysis): " + ", ".join(extras))
    good, errors = {}, []
    for task in tasks:
        path = raw_path(root, task)
        try:
            record = json.loads(path.read_text())
            validate_record(record, task, config)
            good[(task["run_id"], task["instance"], task["seed"])] = record
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"task": task, "error": str(exc)})
    return good, errors


def run_task(task, config, output):
    """Only this function imports/constructs/runs the optimizer."""
    if source_identity() != config["source_identity"]:
        raise ValueError("Sources/runtime changed; restart with compatible configuration")
    instance = ROOT/config["instance_dir"]/(task["instance"]+".json")
    if file_sha(instance) != config["instance_hashes"][task["instance"]]:
        raise ValueError("Instance changed")
    from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation
    data = json.loads(instance.read_text())
    params = {k: task[k] for k in ("N", "N_A", "p_mut", "rho", "T_coop", "FE_max", "seed")}
    started = time.perf_counter()
    algorithm = WIPGraphDualPopulation(data["operations"], data["buffers"], variant="HLNS_FULL", **params)
    if algorithm.algorithm_variant != "HLNS_FULL":
        raise ValueError("Wrong algorithm identity")
    algorithm.run()
    record = {**task, "actual_FE": algorithm.n_evaluations,
              "actual_parameters": {**{k: getattr(algorithm, k) for k in params if k != "seed"}, "seed": task["seed"]},
              "population_sizes": {"P_M": len(algorithm.P_M), "P_S": len(algorithm.P_S)},
              "mechanisms": [algorithm.pm_specialized_kind, algorithm.ps_specialized_kind,
                             algorithm.pm_archive_guidance_enabled, algorithm.ps_archive_guidance_enabled],
              "runtime_seconds": time.perf_counter()-started,
              "final_archive_size": len(algorithm.A),
              "final_archive_objectives": [[float(x.makespan), float(x.shortage)] for x in algorithm.A],
              "source_identity": task_source(config, task), "experiment_fingerprint": digest(config),
              "runner_file_sha256": file_sha(__file__), "completed_utc": utc_now()}
    if source_identity() != config["source_identity"] or file_sha(instance) != config["instance_hashes"][task["instance"]]:
        raise ValueError("Sources changed during optimization; refusing mixed result")
    validate_record(record, task, config)
    atomic_json(raw_path(Path(output), task), record)
    return {"run_id": task["run_id"], "instance": task["instance"], "seed": task["seed"], "status": "successful"}


def safe_run_task(task, config, output):
    """Serialize errors as plain text, never as project exception objects."""
    try:
        return run_task(task, config, output)
    except Exception as exc:
        return {"run_id": task["run_id"], "instance": task["instance"], "seed": task["seed"],
                "status": "failed", "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "error": traceback.format_exc()}


@contextmanager
def round_lock(root):
    path = root/".session.lock"
    try:
        with path.open("x") as handle:
            json.dump({"pid": os.getpid(), "host": platform.node(), "started": utc_now()}, handle)
    except FileExistsError:
        raise RuntimeError(f"Round is locked: {path}. After an interrupted run, verify the old process has exited before removing this lock.")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def run_mode(root, tasks, config):
    total = len(tasks)
    print(f"Taguchi Design: L16(4^4)\nParameter Combinations: {len(config['orthogonal_array'])}\n"
          f"Instances: {len(config['instances'])}\nSeeds: {len(config['seeds'])}\nTotal Tasks: {total}\n"
          f"FE per Task: {config['FE_max']}\nTotal Planned FE: {total*config['FE_max']}\nWorkers: {MAX_WORKERS}", flush=True)
    good, _ = scan_records(root, tasks, config)
    pending, statuses = [], []
    for task in tasks:
        key = (task["run_id"], task["instance"], task["seed"])
        if key in good and SKIP_COMPLETED:
            statuses.append({"run_id": key[0], "instance": key[1], "seed": key[2], "status": "skipped"})
        else:
            path = raw_path(root, task)
            if path.exists():
                preserved = root/"replaced_raw"/str(time.time_ns())/path.relative_to(root/"raw")
                preserved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, preserved)
            pending.append(task)
    started = time.perf_counter()
    def progress():
        counts = Counter(r["status"] for r in statuses)
        state = {"expected": total, "processed": len(statuses), "successful": counts["successful"],
                 "failed": counts["failed"], "skipped": counts["skipped"],
                 "elapsed_seconds": time.perf_counter()-started, "tasks": statuses,
                 "experiment_fingerprint": digest(config), "updated_utc": utc_now()}
        atomic_json(root/"manifest.json", state)
        print(f"Completed {len(statuses)}/{total} | Successful {counts['successful']} | Failed {counts['failed']} | "
              f"Skipped {counts['skipped']} | Elapsed {state['elapsed_seconds']:.1f}s", flush=True)
    progress()
    pool_error = None
    if pending:
        futures = {}
        try:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=get_context("spawn")) as pool:
                try:
                    for task in pending:
                        futures[pool.submit(safe_run_task, task, config, str(root))] = task
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            result = future.result()
                        except BrokenProcessPool:
                            raise
                        except Exception:
                            result = {"run_id": task["run_id"], "instance": task["instance"], "seed": task["seed"],
                                      "status": "failed", "error": traceback.format_exc()}
                        statuses.append(result)
                        if result["status"] == "failed":
                            print(f"Task failed: run={task['run_id']} instance={task['instance']} seed={task['seed']}\n"
                                  f"{result['error']}", file=sys.stderr, flush=True)
                        progress()
                except BrokenProcessPool:
                    for future in futures:
                        future.cancel()
                    raise
        except BrokenProcessPool:
            pool_error = traceback.format_exc()
            print("Process pool broken; stopping collection. Unfinished tasks are not counted as individual algorithm failures.\n"
                  + pool_error, file=sys.stderr, flush=True)
    valid, invalid = scan_records(root, tasks, config)
    manifest = json.loads((root/"manifest.json").read_text())
    manifest.update(valid_completed=len(valid), complete=not invalid, invalid_or_missing=invalid,
                    actual_FE_completed=sum(r["actual_FE"] for r in valid.values()),
                    pool_error=pool_error)
    atomic_json(root/"manifest.json", manifest)
    if pool_error:
        raise RuntimeError("Process pool broken. Original error saved in manifest.json; valid completed raw results are preserved.")
    if invalid:
        raise RuntimeError(f"{len(invalid)} tasks incomplete. See manifest.json and rerun run mode to resume.")
    print("All tasks valid. Run mode does not analyze or plot.", flush=True)


def normalize(point, bounds):
    return tuple(0.0 if hi == lo else (v-lo)/(hi-lo) for v, (lo, hi) in zip(point, bounds))


def compute_hv(points, reference=(1.1, 1.1)):
    previous, area = reference[1], 0.0
    eligible = [p for p in points if p[0] <= reference[0] and p[1] <= reference[1]]
    for x, y in nondominated(eligible):
        area += max(0.0, reference[0]-x)*max(0.0, previous-y)
        previous = min(previous, y)
    return area


def compute_gd(points, reference):
    if not points or not reference:
        raise ValueError("GD requires nonempty objective sets")
    # Requested convention: sqrt(sum(d^2))/|A|, NOT RMS distance.
    distances = [min(math.hypot(x-rx, y-ry) for rx, ry in reference) for x, y in points]
    return math.sqrt(math.fsum(d*d for d in distances))/len(points)


def compute_igd(points, reference):
    if not points or not reference:
        raise ValueError("IGD requires nonempty objective sets")
    return st.mean(min(math.hypot(x-rx, y-ry) for x, y in points) for rx, ry in reference)


def fair_truncate(points, limit):
    """Deterministic iterative crowding truncation of unique normalized ND points."""
    if limit < 2:
        raise ValueError("Need room for both extremes")
    front = nondominated(points)
    while len(front) > limit:
        protected = {min(front, key=lambda p: (p[0], p[1])), min(front, key=lambda p: (p[1], p[0]))}
        crowd = {p: 0.0 for p in front}
        for j in (0, 1):
            ordered = sorted(front, key=lambda p: (p[j], p[1-j]))
            crowd[ordered[0]] = crowd[ordered[-1]] = math.inf
            span = ordered[-1][j]-ordered[0][j]
            if span:
                for k in range(1, len(ordered)-1):
                    crowd[ordered[k]] += (ordered[k+1][j]-ordered[k-1][j])/span
        victim = min((p for p in front if p not in protected), key=lambda p: (crowd[p], p[0], p[1]))
        front.remove(victim)
    return front


def summarize(rows, fields):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in fields)].append(r)
    result = []
    for keys, group in sorted(groups.items()):
        summary = dict(zip(fields, keys))
        summary["n_observations"] = len(group)
        for metric in METRICS:
            values = [r[metric] for r in group]
            summary["mean_"+metric] = st.mean(values)
            summary["std_"+metric] = st.stdev(values) if len(values) > 1 else 0.0
        result.append(summary)
    return result


def ranks(rows, field):
    # Competition ranks preserve ties; stable run_id orders presentation only.
    return {r["run_id"]: 1+sum(other[field] > r[field] for other in rows) for r in rows}


def main_effects(combinations, array, grouped):
    lookup = {r["run_id"]: r for r in combinations}
    effects = []
    sizes = sorted({r["size"] for r in grouped["size"]})
    for factor, letter in zip(FACTORS, LETTERS):
        for level, value in enumerate(FACTOR_LEVELS[factor], 1):
            ids = [r["run_id"] for r in array if r[letter+"_level"] == level]
            row = {"factor": factor, "level_index": level, "level_value": value, "n_combinations": len(ids)}
            for metric in METRICS:
                vals = [lookup[i]["mean_"+metric] for i in ids]
                row["mean_"+metric] = st.mean(vals)
                row["std_combination_mean_"+metric] = st.stdev(vals)
            row["mean_runtime"] = row["mean_runtime_seconds"]
            for size in sizes:
                sub = [r for r in grouped["size"] if r["size"] == size and r["run_id"] in ids]
                for metric in ("nhv", "nhv40", "gd", "igd"):
                    row["mean_"+metric+"_"+size] = st.mean(r["mean_"+metric] for r in sub)
            effects.append(row)
    return effects


def plot_mode(root, tasks, config):
    """Only reads saved objectives; no optimizer, decoder or diagnosis calls."""
    records, errors = scan_records(root, tasks, config)
    if errors:
        for error in errors:
            print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        raise RuntimeError(f"Offline analysis refused: {len(errors)}/{len(tasks)} missing/invalid tasks; no tasks launched")
    bounds_json, references, normalized_refs = {}, {}, {}
    for name in config["instances"]:
        pool = [p for r in records.values() if r["instance"] == name for p in r["final_archive_objectives"]]
        bounds = [(min(p[j] for p in pool), max(p[j] for p in pool)) for j in (0, 1)]
        bounds_json[name] = {"makespan": bounds[0], "shortage": bounds[1],
                             "degenerate_objectives": [k for k, (lo, hi) in zip(("makespan", "shortage"), bounds) if lo == hi]}
        reference = nondominated(pool)
        references[name] = {"raw": reference, "normalized": [normalize(p, bounds) for p in reference]}
        normalized_refs[name] = (bounds, references[name]["normalized"])
    run_rows = []
    for task in tasks:
        r = records[task["run_id"], task["instance"], task["seed"]]
        bounds, reference = normalized_refs[task["instance"]]
        front = nondominated([normalize(p, bounds) for p in r["final_archive_objectives"]])
        fair = fair_truncate(front, config["fair_archive_limit"])
        hv = compute_hv(front, config["hv_reference_point"])
        row = {**task, **dimensions(task["instance"]), "raw_hv": hv,
               "nhv": hv/config["hv_normalization_factor"],
               "nhv40": compute_hv(fair, config["hv_reference_point"])/config["hv_normalization_factor"],
               "gd": compute_gd(front, reference), "igd": compute_igd(front, reference),
               "best_makespan": min(p[0] for p in r["final_archive_objectives"]),
               "best_shortage": min(p[1] for p in r["final_archive_objectives"]),
               "archive_size": r["final_archive_size"], "unique_objective_count": len(front),
               "fair_archive_size": len(fair), "runtime_seconds": r["runtime_seconds"], "actual_FE": r["actual_FE"]}
        run_rows.append(row)
    # Hierarchy: seeds -> instance -> equal-weight instance means. Never point-weighted.
    instance_rows = summarize(run_rows, ("run_id", "instance", "size", "profile", "buffer"))
    params = {r["run_id"]: {k: r[k] for k in (*FACTORS, "N_A")} for r in config["orthogonal_array"]}
    for row in instance_rows:
        row.update(params[row["run_id"]])
        row["n_seeds"] = row["n_observations"]
    instance_observations = [{**{k: r[k] for k in ("run_id", "instance", "size", "profile", "buffer")},
                              **{k: r["mean_"+k] for k in METRICS}} for r in instance_rows]
    combinations = summarize(instance_observations, ("run_id",))
    grouped = {field: summarize(instance_observations, ("run_id", field)) for field in ("size", "profile", "buffer")}
    full_ranks, fair_ranks = ranks(combinations, "mean_nhv"), ranks(combinations, "mean_nhv40")
    for row in combinations:
        rid = row["run_id"]
        row.update(params[rid])
        row["mean_runtime"] = row["mean_runtime_seconds"]
        row["rank_by_mean_nhv"] = full_ranks[rid]
        row["rank_by_mean_nhv40"] = fair_ranks[rid]
        for sub in grouped["size"]:
            if sub["run_id"] == rid:
                for metric in ("nhv", "nhv40", "gd", "igd"):
                    row["mean_"+metric+"_"+sub["size"]] = sub["mean_"+metric]
    effects = main_effects(combinations, config["orthogonal_array"], grouped)
    factor_ranking, recommended = [], {}
    for factor in FACTORS:
        group = [r for r in effects if r["factor"] == factor]
        best = max(group, key=lambda r: (r["mean_nhv"], -r["level_index"]))
        recommended[factor] = best["level_value"]
        factor_ranking.append({"factor": factor, "best_level": best["level_index"],
                               "best_value": best["level_value"], "delta_nhv": max(r["mean_nhv"] for r in group)-min(r["mean_nhv"] for r in group)})
    tested = [r["run_id"] for r in config["orthogonal_array"] if all(r[k] == recommended[k] for k in FACTORS)]
    recommended.update(N_A=2*recommended["N"], FE_max=config["FE_max"],
                       recommendation_basis="总体实例等权 Mean NHV_full 的四因素主效应最大值；并列取较低水平索引",
                       is_tested_orthogonal_row=bool(tested), tested_run_id=tested[0] if tested else None,
                       confirmation_required=not bool(tested), experiment_fingerprint=digest(config),
                       warning="主效应不充分辨识交互作用；未运行组合没有实测响应；独立确认推荐组合仍然有价值。")
    out = root/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/"normalization_bounds.json", bounds_json)
    atomic_json(out/"reference_fronts.json", references)
    for name, data in (("run_metrics", run_rows), ("instance_metrics", instance_rows),
                       ("taguchi_combination_summary", sorted(combinations, key=lambda r: (r["rank_by_mean_nhv"], r["run_id"]))),
                       ("taguchi_main_effects", effects), ("taguchi_factor_ranking", sorted(factor_ranking, key=lambda r: -r["delta_nhv"]))):
        save_csv(out/(name+".csv"), data)
    for field in grouped:
        save_csv(out/({"size": "size_group_summary", "profile": "profile_group_summary", "buffer": "buffer_group_summary"}[field]+".csv"), grouped[field])
    atomic_json(out/"recommended_parameters.json", recommended)
    draw_figures(root/"figures", effects, sorted({r["size"] for r in grouped["size"]}), config["fair_archive_limit"])
    write_report(out/"taguchi_report.md", config, combinations, effects, factor_ranking, grouped, recommended)
    print(f"Offline analysis complete: {out}")


def draw_figures(directory, effects, sizes, fair_limit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    for metric, label in (("nhv", "Mean NHV"), ("nhv40", f"Mean NHV (cap {fair_limit})"), ("gd", "Mean GD"), ("igd", "Mean IGD")):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
        for ax, factor in zip(axes.flat, FACTORS):
            group = [r for r in effects if r["factor"] == factor]
            xs, ys = [r["level_value"] for r in group], [r["mean_"+metric] for r in group]
            best = (max if metric.startswith("nhv") else min)(range(len(ys)), key=lambda i: ys[i])
            ax.plot(xs, ys, marker="o", linewidth=1.3)
            ax.scatter([xs[best]], [ys[best]], facecolors="none", edgecolors="black", s=90, zorder=3)
            ax.set_xticks(xs, [str(x) for x in xs]); ax.set_xlabel(factor); ax.set_ylabel(label)
            ax.grid(alpha=.2)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(directory/f"main_effects_{metric}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for ax, factor in zip(axes.flat, FACTORS):
        group = [r for r in effects if r["factor"] == factor]
        xs = [r["level_value"] for r in group]
        for size in sizes:
            ax.plot(xs, [r["mean_nhv_"+size] for r in group], marker="o", linewidth=1.3, label=size)
        ax.set_xticks(xs, [str(x) for x in xs]); ax.set_xlabel(factor); ax.set_ylabel("Mean NHV")
        ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(directory/f"main_effects_nhv_by_size.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(path, config, combinations, effects, ranking, grouped, recommended):
    count = len(config["orthogonal_array"])*len(config["instances"])*len(config["seeds"])
    lines = ["# Full 算法田口参数校正报告", "", "## 实验配置",
             f"轮次：{config['round']}；L16(4⁴)，四因素四水平，{len(config['orthogonal_array'])}组参数，"
             f"{len(config['instances'])}个实例，每例{len(config['seeds'])}个种子，共{count}次运行，每次{config['FE_max']} FE。",
             "N表示每个种群规模；N_A=2N。因素N校正的是种群及关联档案容量，二者作用未独立分离。",
             f"实例集合与当前预消融配置完全一致：{config['matches_pre_ablation_instances']}。",
             "采用固定FE比较，不宣称该预算已经通过充分收敛验证。",
             "", "## 参数组合结果", "|组合|N / N_A|p_mut|rho|T_coop|Mean NHV|NHV cap|GD|IGD|耗时秒|",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(combinations, key=lambda r: (r["rank_by_mean_nhv"], r["run_id"])):
        lines.append(f"|{r['run_id']}|{r['N']} / {r['N_A']}|{r['p_mut']}|{r['rho']}|{r['T_coop']}|"
                     f"{r['mean_nhv']:.6f}|{r['mean_nhv40']:.6f}|{r['mean_gd']:.6f}|{r['mean_igd']:.6f}|{r['mean_runtime']:.2f}|")
    lines += ["", "## 四因素主效应与后续范围建议"]
    for factor in FACTORS:
        group = [r for r in effects if r["factor"] == factor]
        best = max(group, key=lambda r: (r["mean_nhv"], -r["level_index"]))
        delta = max(r["mean_nhv"] for r in group)-min(r["mean_nhv"] for r in group)
        values = "、".join(f"{r['level_value']}→{r['mean_nhv']:.6f}" for r in group)
        tied = [r["level_value"] for r in group if r["mean_nhv"] == best["mean_nhv"]]
        position = "下边界" if best["level_index"] == 1 else "上边界" if best["level_index"] == 4 else "内部"
        suggestion = ("可考虑向较小水平扩展，但需遵守参数合法域。" if position == "下边界" else
                      "可考虑向较大水平扩展，但需遵守参数合法域。" if position == "上边界" else "可考虑围绕内部较优区间缩小范围、提高分辨率。")
        if delta <= EPS:
            suggestion = "各水平主效应数值基本一致，当前响应下不能确定明确方向。"
        lines.append(f"- {factor}：{values}；最大值对应{best['level_value']}（{position}），极差{delta:.6f}；并列最佳={tied}。{suggestion}")
    lines += ["主效应极差仅表示当前测试范围内响应幅度，不是统计显著性或独立因果贡献。",
              "", "## 分规模、瓶颈与缓冲配置"]
    for field, title in (("size", "规模"), ("profile", "负荷结构"), ("buffer", "缓冲等级")):
        for value in sorted({r[field] for r in grouped[field]}):
            subset = {r["run_id"]: r for r in grouped[field] if r[field] == value}
            choices = {}
            for factor, letter in zip(FACTORS, LETTERS):
                level_means = [(level, st.mean(subset[r["run_id"]]["mean_nhv"] for r in config["orthogonal_array"]
                                               if r[letter+"_level"] == level)) for level in range(1, 5)]
                best_level = max(level_means, key=lambda x: (x[1], -x[0]))[0]
                choices[factor] = config["factor_levels"][factor][best_level-1]
            mismatches = [factor for factor in FACTORS if choices[factor] != recommended[factor]]
            lines.append(f"- {title}{value}：描述性最佳水平{choices}；与总体建议不同的因素：{mismatches}。")
    lines.append("各分组完整指标保留在分组CSV。分组响应差异不自动转化为自适应参数；正式选择仍以总体实例等权Mean NHV为准。")
    best_row = max(combinations, key=lambda r: (r["mean_nhv"], -int(r["run_id"])))
    for field in ("size", "profile", "buffer"):
        for group in sorted({r[field] for r in grouped[field]}):
            sub = [r for r in grouped[field] if r[field] == group]
            selected = next(r for r in sub if r["run_id"] == best_row["run_id"])
            winner = max(sub, key=lambda r: (r["mean_nhv"], -int(r["run_id"])))
            lines.append(f"- 总体实测最佳行{best_row['run_id']}在{group}：NHV={selected['mean_nhv']:.6f}；组内最佳行{winner['run_id']}，"
                         f"NHV差距={winner['mean_nhv']-selected['mean_nhv']:.6f}（描述性差距，未作显著性判定）。")
    changed = [r["run_id"] for r in combinations if r["rank_by_mean_nhv"] != r["rank_by_mean_nhv40"]]
    full_winners = [r["run_id"] for r in combinations if r["rank_by_mean_nhv"] == 1]
    fair_winners = [r["run_id"] for r in combinations if r["rank_by_mean_nhv40"] == 1]
    lines += ["", "## 档案点数公平性辅助检查",
              f"离线统一上限={config['fair_archive_limit']}；完整档案最佳行={full_winners}，统一上限最佳行={fair_winners}；排名发生变化的行={changed}。",
              "截断在同一归一化目标空间内，对唯一非支配目标点迭代拥挤度删除，保护两极端点，并采用固定字典序破除并列。",
              "该辅助检查不消除档案容量对AC及整个搜索轨迹的影响；主响应仍是完整档案NHV。",
              "", "## 推荐参数", "```json", json.dumps(recommended, ensure_ascii=False, indent=2), "```",
              "推荐组合的四个水平来自主效应最大值，未必是16行之一。若不是实测行，需要独立确认，不能编造实测响应。",
              "若恰为实测行，其测量也仅代表当前实例/种子；不构成独立确认试验。本脚本不自动开展确认或第二轮实验。",
              "", "## 统计与解释边界",
              "实例内种子等权，然后实例等权；组合std为实例均值之间的样本标准差，实例std为种子之间的样本标准差；",
              "主效应std_combination_mean为四个组合均值的样本标准差。图中不添加未经解释的误差棒。",
              "GD采用sqrt(sum(d²))/|A|，IGD采用参考点最近距离算术平均；距离在统一归一化空间计算。",
              "raw_hv指归一化空间HV除以1.21之前的面积，不是原始物理单位目标空间的HV。重复目标点按集合去重。",
              "经验参考前沿不是数学最优前沿；退化目标维度归一化为0，记录在normalization_bounds.json。",
              "最佳makespan与最佳shortage可能来自不同解；不能解释为同一解同时达到两个极端。",
              "L16主要估计主效应，不能充分辨识全部交互作用。候选间细小差异不等于显著差异，没有统计证据时不作强敏感性结论。",
              "不同轮次参考尺度独立，不能直接拼接不同轮次曲线或以其NHV绝对数值判断哪一轮更优。"]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "plot"), default=MODE)
    args = parser.parse_args()
    # Pin hash randomization before spawning, independently of completion order.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())]+sys.argv[1:])
    array = validate_configuration()
    config = experiment_config(array)
    root = ROOT/RESULTS_DIR/ROUND_NAME
    tasks = build_tasks(config)
    config_path = root/"config.json"
    if config_path.exists():
        stored = json.loads(config_path.read_text())
        if stored.get("experiment_fingerprint") != digest(config) or stored.get("experiment") != config:
            raise ValueError("Incompatible round configuration/source. Change ROUND_NAME or explicitly resolve old data; never mix rounds.")
    elif args.mode == "plot":
        raise FileNotFoundError(f"No completed experiment configuration: {config_path}")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Nonempty round lacks config.json; refusing to overwrite unknown data")
    root.mkdir(parents=True, exist_ok=True)
    with round_lock(root):
        if not config_path.exists():
            atomic_json(config_path, {"experiment": config, "experiment_fingerprint": digest(config),
                                      "created_utc": utc_now(), "initial_workers": MAX_WORKERS,
                                      "runner_file_sha256": file_sha(__file__)})
            save_csv(root/"orthogonal_array.csv", array)
        else:
            with (root/"orthogonal_array.csv").open(encoding="utf-8-sig", newline="") as handle:
                saved_array = list(csv.DictReader(handle))
            if saved_array != [{k: str(v) for k, v in r.items()} for r in array]:
                raise ValueError("Saved orthogonal table is missing or inconsistent")
        if args.mode == "run":
            run_mode(root, tasks, config)
        else:
            plot_mode(root, tasks, config)


if __name__ == "__main__":
    main()
