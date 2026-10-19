"""H-LNS pre-ablation: run, analyze, or strictly complete run then analyze."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation
from src.algorithms.wip_graph_dual_population_wo_mgs import WIPGraphDualPopulationWithoutMGS
from src.algorithms.wip_graph_dual_population_wo_hlns import WIPGraphDualPopulationWithoutHLNS
from src.algorithms.wip_graph_dual_population_wo_ac import WIPGraphDualPopulationWithoutAC


# USER CONFIGURATION
MODE = "all"  # "run", "analyze", "all"
ALLOWED_MODES = {"run", "analyze", "all"}
VARIANT_REGISTRY = {
    "full": WIPGraphDualPopulation,
    "wo_mgs": WIPGraphDualPopulationWithoutMGS,
    "wo_hlns": WIPGraphDualPopulationWithoutHLNS,
    "wo_ac": WIPGraphDualPopulationWithoutAC,
}
EXPECTED_ALGORITHM_IDS = MappingProxyType({
    "full": "HLNS_FULL", "wo_mgs": "HLNS_WO_MGS",
    "wo_hlns": "HLNS_WO_HLNS", "wo_ac": "HLNS_WO_AC",
})
EXPECTED_MECHANISMS = MappingProxyType({
    "full": ("MGS", "HLNS", True, False),
    "wo_mgs": ("BasicMutation", "HLNS", True, False),
    "wo_hlns": ("MGS", "BasicMutation", True, False),
    "wo_ac": ("MGS", "HLNS", False, False),
})
SELECTED_VARIANTS = ["full", "wo_mgs", "wo_hlns", "wo_ac"]
SELECTED_INSTANCES = [
    "WIPHFSP_S1_P1_B1", "WIPHFSP_S1_P2_B2", "WIPHFSP_S1_P3_B3", "WIPHFSP_S1_P4_B1",
    "WIPHFSP_S4_P1_B2", "WIPHFSP_S4_P2_B3", "WIPHFSP_S4_P3_B1", "WIPHFSP_S4_P4_B2",
    "WIPHFSP_S8_P1_B3", "WIPHFSP_S8_P2_B1", "WIPHFSP_S8_P3_B2", "WIPHFSP_S8_P4_B3",
]
SEEDS = [0, 1, 2, 3, 4]
FE_MAX = 10000
N = 50
N_A = 100
P_MU = 0.1
RHO = 0.8
T_COOP = 10
USE_MULTIPROCESSING = True
N_WORKERS = 60
INSTANCE_DIR = PROJECT_ROOT / "data/final_benchmark/instances"
OUTPUT_ROOT = PROJECT_ROOT / "experiments/results/pre_ablation_hlns"
OVERWRITE_EXISTING = False
HV_REFERENCE_POINT = (1.1, 1.1)
EPS = 1e-12
TOL = 1e-12
PYTHON_HASH_SEED = "0"
RESULT_SCHEMA = "hlns-pre-ablation-v1"
SOURCE_FILES = (
    "src/algorithms/wip_graph_dual_population.py",
    "src/algorithms/wip_graph_dual_population_wo_mgs.py",
    "src/algorithms/wip_graph_dual_population_wo_hlns.py",
    "src/algorithms/wip_graph_dual_population_wo_ac.py",
    "src/solution/encoder.py", "src/solution/decoder.py",
)
IMPORTED_SOURCE_HASHES = MappingProxyType({
    name: hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES
})
IDENTITY_FIELDS = ("pm_specialized_kind", "ps_specialized_kind",
                   "pm_archive_guidance_enabled", "ps_archive_guidance_enabled")


@dataclass(frozen=True)
class Task:
    instance: str
    variant: str
    seed: int


@dataclass(frozen=True)
class Settings:
    instance_dir: Path
    output_root: Path
    fe_max: int
    n: int
    n_a: int
    p_mu: float
    rho: float
    t_coop: int

    def algorithm_parameters(self, seed: int) -> dict:
        return {"N": self.n, "N_A": self.n_a, "p_mut": self.p_mu,
                "rho": self.rho, "T_coop": self.t_coop,
                "FE_max": self.fe_max, "seed": seed}


def current_settings() -> Settings:
    return Settings(INSTANCE_DIR, OUTPUT_ROOT, FE_MAX, N, N_A, P_MU, RHO, T_COOP)


def validate_output_root(settings: Settings) -> None:
    root = settings.output_root.resolve()
    project = PROJECT_ROOT.resolve()
    allowed = (project / "experiments/results/pre_ablation_hlns").resolve()
    historical = (project / "experiments/results/pre_ablation").resolve()
    if root == historical or historical in root.parents or root in historical.parents:
        raise ValueError("Historical pre_ablation output must never be overwritten")
    if (root == project or project in root.parents) and root != allowed and allowed not in root.parents:
        raise ValueError("Repository output must be inside pre_ablation_hlns; use an external temporary directory for tests")


def validate_configuration(settings: Settings) -> None:
    if MODE not in ALLOWED_MODES:
        raise ValueError(f"Invalid MODE: {MODE}; allowed: run, analyze, all")
    for name, values in (("variants", SELECTED_VARIANTS),
                         ("instances", SELECTED_INSTANCES), ("seeds", SEEDS)):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be nonempty and unique")
    if any(v not in VARIANT_REGISTRY for v in SELECTED_VARIANTS) or "full" not in SELECTED_VARIANTS:
        raise ValueError("Select current variants and include full as the paired baseline")
    if any(type(seed) is not int for seed in SEEDS):
        raise ValueError("SEEDS must contain integers")
    if any(type(v) is not int for v in (settings.n, settings.n_a, settings.fe_max, settings.t_coop)):
        raise ValueError("Population sizes, FE and cooperation interval must be integers")
    if settings.n < 1 or settings.n_a < 2 or settings.fe_max < 2*settings.n or settings.t_coop < 1:
        raise ValueError("Require N>=1, N_A>=2, FE_MAX>=2*N, T_COOP>=1")
    if not (0 <= settings.p_mu <= 1 and 0 < settings.rho <= 1):
        raise ValueError("Invalid p_mut or rho")
    if type(N_WORKERS) is not int or N_WORKERS < 1:
        raise ValueError("N_WORKERS must be a positive integer")
    if tuple(HV_REFERENCE_POINT) != (1.1, 1.1) or EPS <= 0 or TOL < 0:
        raise ValueError("Expected HV reference (1.1,1.1), EPS>0, TOL>=0")
    validate_output_root(settings)
    for name in SELECTED_INSTANCES:
        instance_dimensions(name)
        if not (settings.instance_dir/f"{name}.json").is_file():
            raise FileNotFoundError(f"Missing benchmark instance: {name}")


def build_tasks() -> List[Task]:
    return [Task(instance, variant, seed) for instance in SELECTED_INSTANCES
            for variant in SELECTED_VARIANTS for seed in SEEDS]


def raw_path(task: Task, settings: Settings) -> Path:
    instance_dimensions(task.instance)
    if task.variant not in VARIANT_REGISTRY or type(task.seed) is not int:
        raise ValueError(f"Invalid task: {task}")
    path = settings.output_root/"raw"/task.instance/task.variant/f"seed_{task.seed}.json"
    if settings.output_root.resolve() not in path.resolve().parents:
        raise ValueError(f"Task path escapes output root: {task}")
    return path


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def task_source(task: Task, settings: Settings) -> dict:
    """Stable across MODE/worker-count edits; pins all comparison implementations."""
    sources = {name: hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest()
               for name in SOURCE_FILES}
    if sources != IMPORTED_SOURCE_HASHES:
        raise ValueError("Algorithm/encoder/decoder source changed since import; restart this runner")
    return {"schema": RESULT_SCHEMA, "instance": task.instance, "variant": task.variant,
            "algorithm_variant": EXPECTED_ALGORITHM_IDS[task.variant], "seed": task.seed,
            "algorithm_parameters": settings.algorithm_parameters(task.seed),
            "python_hash_seed": PYTHON_HASH_SEED,
            "sources": sources,
            "instance_sha256": hashlib.sha256(
                (settings.instance_dir/f"{task.instance}.json").read_bytes()).hexdigest()}


def source_fingerprint(source: dict) -> str:
    return hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_identity(metadata: dict, task: Task) -> None:
    if metadata.get("algorithm_variant") != EXPECTED_ALGORITHM_IDS[task.variant]:
        raise ValueError(f"Formal algorithm identity mismatch: {task}")
    for field, expected in zip(IDENTITY_FIELDS, EXPECTED_MECHANISMS[task.variant]):
        value = metadata.get(field)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Mechanism identity mismatch: {task}: {field}")


def validated_points(archive: Sequence[dict]) -> List[Tuple[float, float]]:
    if not isinstance(archive, list) or not archive:
        raise ValueError("final_archive must be a nonempty list")
    points = []
    for solution in archive:
        x, y = float(solution["makespan"]), float(solution["shortage"])
        if not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0:
            raise ValueError("Nonfinite or negative objective in final_archive")
        points.append((x, y))
    return points


def validate_raw_record(record: dict, task: Task, settings: Settings) -> None:
    if type(record["seed"]) is not int or (record["instance"], record["variant"], record["seed"]) != (task.instance, task.variant, task.seed):
        raise ValueError(f"Task identity mismatch: {task}")
    if record["algorithm_variant"] != EXPECTED_ALGORITHM_IDS[task.variant]:
        raise ValueError(f"Algorithm identity mismatch: {task}")
    for key in ("FE_max", "actual_FE"):
        if type(record[key]) is not int or record[key] != settings.fe_max:
            raise ValueError(f"FE mismatch: {task}: {key}")
    if record["algorithm_parameters"] != settings.algorithm_parameters(task.seed):
        raise ValueError(f"Algorithm parameter mismatch: {task}")
    constructor = settings.algorithm_parameters(task.seed)
    if task.variant == "full":
        constructor["variant"] = EXPECTED_ALGORITHM_IDS["full"]
    if record["constructor_parameters"] != constructor:
        raise ValueError(f"Constructor parameter mismatch: {task}")
    if record["population_sizes"] != {"P_M": settings.n, "P_S": settings.n}:
        raise ValueError(f"Population size mismatch: {task}")
    expected_source = task_source(task, settings)
    if (record["source"] != expected_source or
            source_fingerprint(record["source"]) != source_fingerprint(expected_source) or
            record["fingerprint"] != source_fingerprint(expected_source)):
        raise ValueError(f"Source fingerprint mismatch: {task}")
    validate_identity(record["diagnostics"], task)
    points = validated_points(record["final_archive"])
    if type(record["final_archive_size"]) is not int or record["final_archive_size"] != len(points):
        raise ValueError(f"Archive size mismatch: {task}")
    if len(points) > settings.n_a or set(points) != set(nondominated_filter(points)):
        raise ValueError(f"Invalid nondominated archive/capacity: {task}")
    if record["best_makespan"] != min(x for x, _ in points) or record["best_shortage"] != min(y for _, y in points):
        raise ValueError(f"Archive extreme mismatch: {task}")
    runtime = record["runtime_seconds"]
    if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime < 0:
        raise ValueError(f"Invalid runtime: {task}")
    if not isinstance(record["timestamp_utc"], str) or not record["timestamp_utc"]:
        raise ValueError(f"Missing timestamp: {task}")
    generated = 0
    for group in ("pm_moves", "ps_moves"):
        expected_kinds = ({"machine_priority", "machine_reassign", "basic_mutation", "fallback_basic_mutation"}
                          if group == "pm_moves" else {"basic_mutation", "hlns_joint"})
        moves = record["diagnostics"][group]
        if set(moves) != expected_kinds:
            raise ValueError(f"Unexpected move statistics: {task}: {group}")
        for counts in moves.values():
            if any(type(counts[k]) is not int or counts[k] < 0 for k in ("generated", "accepted")):
                raise ValueError(f"Invalid move counts: {task}")
            if counts["accepted"] > counts["generated"]:
                raise ValueError(f"Accepted exceeds generated: {task}")
            generated += counts["generated"]
    if generated != settings.fe_max - 2*settings.n:
        raise ValueError(f"Offspring FE mismatch: {task}")
    if set(record["diagnostics"]["cooperation"]) != {"pm_attempts", "pm_usable", "pm_fallback_self"}:
        raise ValueError(f"Unexpected cooperation statistics: {task}")
    cooperation = record["diagnostics"]["cooperation"]
    if any(type(v) is not int or v < 0 for v in cooperation.values()):
        raise ValueError(f"Invalid cooperation counts: {task}")
    if cooperation["pm_attempts"] != cooperation["pm_usable"] + cooperation["pm_fallback_self"]:
        raise ValueError(f"Inconsistent cooperation counts: {task}")
    diagnostics = record["diagnostics"]
    if task.variant == "wo_ac" and cooperation["pm_attempts"]:
        raise ValueError(f"Disabled cooperation was used: {task}")
    if task.variant == "wo_hlns" and (diagnostics["ps_moves"]["hlns_joint"]["generated"] or
                                       diagnostics["hlns"]["hlns_calls"] or diagnostics["shortage_diagnosis"]["calls"]):
        raise ValueError(f"Disabled H-LNS was used: {task}")
    if task.variant == "wo_mgs" and any(diagnostics["pm_moves"][k]["generated"]
                                      for k in ("machine_priority", "machine_reassign")):
        raise ValueError(f"Disabled MGS was used: {task}")


def load_valid_raw(path: Path, task: Task, settings: Settings) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    validate_raw_record(record, task, settings)
    return record


def run_single_task(task: Task, settings: Settings, overwrite: Optional[bool] = None) -> dict:
    start = time.perf_counter()
    try:
        validate_output_root(settings)
        if os.environ.get("PYTHONHASHSEED") != PYTHON_HASH_SEED:
            raise RuntimeError(f"Set PYTHONHASHSEED={PYTHON_HASH_SEED} before running tasks")
        path = raw_path(task, settings)
        overwrite = OVERWRITE_EXISTING if overwrite is None else overwrite
        if path.exists() and not overwrite:
            load_valid_raw(path, task, settings)
            return {"task": task.__dict__, "status": "skipped"}
        source = task_source(task, settings)
        instance = json.loads((settings.instance_dir/f"{task.instance}.json").read_text())
        parameters = settings.algorithm_parameters(task.seed)
        constructor = dict(parameters)
        if task.variant == "full":
            constructor["variant"] = EXPECTED_ALGORITHM_IDS["full"]
        cls = VARIANT_REGISTRY[task.variant]
        algorithm = cls(instance["operations"], instance["buffers"], **constructor)
        if type(algorithm) is not cls:
            raise ValueError(f"Wrong algorithm class: {task}")
        validate_identity({k: getattr(algorithm, k) for k in ("algorithm_variant", *IDENTITY_FIELDS)}, task)
        validate_identity(algorithm.diagnostics, task)
        archive = algorithm.run()
        points = [(float(ind.makespan), float(ind.shortage)) for ind in archive]
        if not points:
            raise ValueError(f"Empty final archive: {task}")
        diagnostics = {key: algorithm.diagnostics[key] for key in
                       ("algorithm_variant", *IDENTITY_FIELDS, "pm_moves", "ps_moves", "cooperation",
                        "shortage_diagnosis", "source_generated")}
        diagnostics["hlns"] = algorithm.hlns_statistics
        record = {
            "instance": task.instance, "variant": task.variant, "seed": task.seed,
            "algorithm_variant": algorithm.algorithm_variant,
            "FE_max": settings.fe_max, "actual_FE": algorithm.n_evaluations,
            "algorithm_parameters": parameters, "constructor_parameters": constructor,
            "population_sizes": {"P_M": len(algorithm.P_M), "P_S": len(algorithm.P_S)},
            "runtime_seconds": time.perf_counter()-start,
            "final_archive": [{"makespan": x, "shortage": y} for x, y in points],
            "final_archive_size": len(points), "best_makespan": min(x for x, _ in points),
            "best_shortage": min(y for _, y in points), "diagnostics": diagnostics,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": source, "fingerprint": source_fingerprint(source),
        }
        validate_raw_record(record, task, settings)
        write_json_atomic(path, record)
        return {"task": task.__dict__, "status": "completed", "runtime_seconds": record["runtime_seconds"]}
    except Exception as exc:
        return {"task": task.__dict__, "status": "failed", "exception": repr(exc)}


def inspect_raw(tasks: Sequence[Task], settings: Settings) -> Tuple[Dict[Task, dict], List[dict]]:
    found, missing = {}, []
    for task in tasks:
        try:
            found[task] = load_valid_raw(raw_path(task, settings), task, settings)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            missing.append({**task.__dict__, "reason": str(exc)})
    return found, missing


def run_experiments(settings: Optional[Settings] = None) -> dict:
    settings = settings or current_settings()
    validate_configuration(settings)
    if os.environ.get("PYTHONHASHSEED") != PYTHON_HASH_SEED:
        raise RuntimeError(f"Set PYTHONHASHSEED={PYTHON_HASH_SEED} before run_experiments()")
    tasks, start = build_tasks(), time.perf_counter()
    pending, skipped, completed, failed = [], [], [], []
    for task in tasks:
        path = raw_path(task, settings)
        if path.exists() and not OVERWRITE_EXISTING:
            try:
                load_valid_raw(path, task, settings)
                skipped.append(task)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failed.append({"task": task.__dict__, "status": "failed",
                               "exception": f"Existing result is invalid; inspect/remove it or explicitly enable overwrite: {exc}"})
        else:
            pending.append(task)

    def manifest(missing=None):
        payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "expected": len(tasks), "completed": len(completed), "skipped": len(skipped),
                   "failed": len(failed), "missing": None if missing is None else len(missing),
                   "complete": missing is not None and not failed and not missing,
                   "tasks": [t.__dict__ for t in tasks], "failed_tasks": failed,
                   "missing_tasks": [] if missing is None else missing,
                   "total_runtime_seconds": time.perf_counter()-start}
        write_json_atomic(settings.output_root/"run_manifest.json", payload)
        return payload

    if failed:
        _, missing = inspect_raw(tasks, settings)
        manifest(missing)
        raise RuntimeError(f"Invalid existing results; no tasks launched: {failed}")
    manifest()
    print(f"H-LNS pre-ablation: {len(tasks)} expected, {len(skipped)} skipped, {len(pending)} pending", flush=True)

    def report(result):
        if result["status"] == "skipped":
            skipped.append(Task(**result["task"]))
        else:
            (completed if result["status"] == "completed" else failed).append(result)
        print(f"{result['status']}: {result['task']} {result.get('exception', '')}", flush=True)
        manifest()

    if USE_MULTIPROCESSING and pending:
        try:
            with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=get_context("spawn")) as executor:
                futures = {executor.submit(run_single_task, task, settings, OVERWRITE_EXISTING): task for task in pending}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"task": futures[future].__dict__, "status": "failed", "exception": repr(exc)}
                    report(result)
        except Exception as exc:
            reported = {Task(**entry["task"]) for entry in completed + failed} | set(skipped)
            for task in pending:
                if task not in reported:
                    report({"task": task.__dict__, "status": "failed",
                            "exception": f"Executor failed: {exc!r}"})
    else:
        for task in pending:
            report(run_single_task(task, settings, OVERWRITE_EXISTING))
    _, missing = inspect_raw(tasks, settings)
    payload = manifest(missing)
    if not payload["complete"]:
        raise RuntimeError(f"Run incomplete: failed={failed}, missing/invalid={missing}")
    return payload


def nondominated_filter(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Unique 2D minimization front in increasing first-objective order."""
    front, best_y = [], math.inf
    for x, y in sorted(set(points)):
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def pooled_bounds(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    return (min(x for x, _ in points), max(x for x, _ in points),
            min(y for _, y in points), max(y for _, y in points))


def normalize_point(point: Tuple[float, float], bounds: Tuple[float, float, float, float]
                    ) -> Tuple[float, float]:
    x_min, x_max, y_min, y_max = bounds
    x = 0.0 if x_max - x_min < EPS else (point[0] - x_min) / (x_max - x_min)
    y = 0.0 if y_max - y_min < EPS else (point[1] - y_min) / (y_max - y_min)
    if not (-EPS <= x <= 1 + EPS and -EPS <= y <= 1 + EPS):
        raise ValueError("Normalized objective outside pooled [0,1] bounds")
    return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))


def compute_hv_2d(points: Sequence[Tuple[float, float]],
                  ref_point: Tuple[float, float] = HV_REFERENCE_POINT) -> float:
    """Exact area dominated by normalized 2D minimization points."""
    eligible = [(x, y) for x, y in points if x <= ref_point[0] and y <= ref_point[1]]
    hv, previous_y = 0.0, ref_point[1]
    for x, y in nondominated_filter(eligible):
        hv += max(0.0, ref_point[0] - x) * max(0.0, previous_y - y)
        previous_y = min(previous_y, y)
    return hv


def compute_igd(archive: Sequence[Tuple[float, float]],
                reference: Sequence[Tuple[float, float]]) -> float:
    if not archive or not reference:
        raise ValueError("IGD needs a nonempty archive and reference front")
    return statistics.mean(min(math.hypot(rx - ax, ry - ay) for ax, ay in archive)
                           for rx, ry in reference)


def metric_sanity_checks() -> None:
    if nondominated_filter([(1, 3), (2, 2), (3, 1), (2, 3), (1, 3)]) != [
            (1, 3), (2, 2), (3, 1)]:
        raise AssertionError("Nondominated filtering failed")
    if not math.isclose(compute_hv_2d([(0.2, 0.8), (0.8, 0.2)]), 0.45,
                        abs_tol=EPS):
        raise AssertionError("2D minimization HV sanity check failed")
    if not math.isclose(compute_igd([(0, 0)], [(0, 0), (1, 1)]),
                        math.sqrt(2) / 2, abs_tol=EPS):
        raise AssertionError("IGD sanity check failed")


def instance_dimensions(name: str) -> dict:
    match = re.fullmatch(r"WIPHFSP_(S\d+)_(P\d+)_(B\d+)", name)
    if not match:
        raise ValueError(f"Unexpected benchmark instance name: {name}")
    return dict(zip(("size_group", "profile", "buffer_level"), match.groups()))


def summary_stats(values: Sequence[float]) -> dict:
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values)}


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


RUN_FIELDS = ("instance", "size_group", "profile", "buffer_level", "variant", "seed",
              "nhv", "raw_hv", "igd", "best_makespan", "best_shortage",
              "archive_size", "runtime_seconds", "actual_FE")
DELTA_METRICS = ("nhv", "igd", "makespan", "shortage")
DELTA_FIELDS = ("instance", "size_group", "profile", "buffer_level", "variant", "seed",
                "delta_nhv", "delta_igd", "delta_makespan", "delta_shortage")
GROUP_FIELDS = ("scope", "group", "variant", "metric", "n_pairs", "mean_delta",
                "median_delta", "std_delta", "wins", "ties", "losses")


def build_run_metrics(records: Dict[Task, dict]) -> List[dict]:
    by_instance = defaultdict(list)
    for task, record in records.items():
        by_instance[task.instance].extend(validated_points(record["final_archive"]))
    references = {}
    for instance, pooled in by_instance.items():
        bounds = pooled_bounds(pooled)
        raw_front = nondominated_filter(pooled)
        if raw_front != nondominated_filter(raw_front):
            raise AssertionError("Empirical reference front is dominated")
        references[instance] = (bounds, [normalize_point(point, bounds)
                                        for point in raw_front])
    rows = []
    for task in sorted(records, key=lambda t: (t.instance, t.variant, t.seed)):
        record = records[task]
        bounds, reference = references[task.instance]
        raw_points = nondominated_filter(validated_points(record["final_archive"]))
        normalized = [normalize_point(point, bounds) for point in raw_points]
        raw_hv = compute_hv_2d(normalized, HV_REFERENCE_POINT)
        nhv = raw_hv / (HV_REFERENCE_POINT[0] * HV_REFERENCE_POINT[1])
        if not -EPS <= nhv <= 1 + EPS:
            raise ValueError("NHV outside [0,1]")
        rows.append({**instance_dimensions(task.instance),
                     "instance": task.instance, "variant": task.variant, "seed": task.seed,
                     "nhv": nhv, "raw_hv": raw_hv,
                     "igd": compute_igd(normalized, reference),
                     "best_makespan": record["best_makespan"],
                     "best_shortage": record["best_shortage"],
                     "archive_size": record["final_archive_size"],
                     "runtime_seconds": record["runtime_seconds"],
                     "actual_FE": record["actual_FE"]})
    return rows


def build_instance_summary(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["instance"], row["variant"])].append(row)
    result = []
    for (instance, variant), group in sorted(grouped.items()):
        entry = {"instance": instance, **instance_dimensions(instance),
                 "variant": variant, "n_seeds": len(group)}
        for metric in ("nhv", "igd", "best_makespan", "best_shortage"):
            for name, value in summary_stats([row[metric] for row in group]).items():
                entry[f"{metric}_{name}"] = value
        entry["runtime_mean"] = statistics.mean(row["runtime_seconds"] for row in group)
        entry["archive_size_mean"] = statistics.mean(row["archive_size"] for row in group)
        result.append(entry)
    return result


def build_pairwise_deltas(rows: Sequence[dict]) -> List[dict]:
    by_key = {(row["instance"], row["variant"], row["seed"]): row for row in rows}
    paired = []
    for row in rows:
        if row["variant"] == "full":
            continue
        full = by_key.get((row["instance"], "full", row["seed"]))
        if full is None:
            continue
        paired.append({key: row[key] for key in DELTA_FIELDS[:6]} | {
            "delta_nhv": full["nhv"] - row["nhv"],
            "delta_igd": row["igd"] - full["igd"],
            "delta_makespan": row["best_makespan"] - full["best_makespan"],
            "delta_shortage": row["best_shortage"] - full["best_shortage"],
        })
    return paired


def build_group_summary(paired: Sequence[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in paired:
        for scope, group in (("overall", "all"),
                             ("size_group", row["size_group"]),
                             ("profile", row["profile"]),
                             ("buffer_level", row["buffer_level"])):
            for metric in DELTA_METRICS:
                groups[(scope, group, row["variant"], metric)].append(row[f"delta_{metric}"])
    result = []
    for (scope, group, variant, metric), values in sorted(groups.items()):
        stats = summary_stats(values)
        result.append({"scope": scope, "group": group, "variant": variant,
                       "metric": metric, "n_pairs": len(values),
                       "mean_delta": stats["mean"], "median_delta": stats["median"],
                       "std_delta": stats["std"],
                       "wins": sum(value > TOL for value in values),
                       "ties": sum(abs(value) <= TOL for value in values),
                       "losses": sum(value < -TOL for value in values)})
    return result


def wilcoxon_signed_rank(values: Sequence[float]) -> Tuple[float, float, int]:
    """Exact two-sided sign-rank test; zero differences are excluded."""
    nonzero = sorted((abs(value), value > 0) for value in values if abs(value) > TOL)
    if not nonzero:
        return (0.0, 1.0, 0)
    ranks_twice, index = [], 0
    while index < len(nonzero):
        end = index + 1
        while end < len(nonzero) and nonzero[end][0] == nonzero[index][0]:
            end += 1
        ranks_twice.extend((index + 1 + end) for _ in range(index, end))
        index = end
    positive_twice = sum(rank for rank, (_, positive) in zip(ranks_twice, nonzero)
                         if positive)
    total_twice = sum(ranks_twice)
    counts = {0: 1}
    for rank in ranks_twice:
        updated = counts.copy()
        for total, count in counts.items():
            updated[total + rank] = updated.get(total + rank, 0) + count
        counts = updated
    lower = sum(count for total, count in counts.items() if total <= positive_twice)
    upper = sum(count for total, count in counts.items() if total >= positive_twice)
    return (min(positive_twice, total_twice - positive_twice) / 2,
            min(1.0, 2 * min(lower, upper) / (2 ** len(nonzero))), len(nonzero))


def build_wilcoxon(paired: Sequence[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in paired:
        for metric in DELTA_METRICS:
            groups[(row["variant"], metric)].append(row[f"delta_{metric}"])
    result = []
    for (variant, metric), values in sorted(groups.items()):
        statistic, p_value, n_nonzero = wilcoxon_signed_rank(values)
        result.append({"variant": variant, "metric": metric, "n_pairs": len(values),
                       "n_nonzero": n_nonzero, "statistic": statistic,
                       "p_value": p_value})
    return result


def mechanism_summary(group_rows: Sequence[dict]) -> str:
    focus = {
        "wo_mgs": (("nhv", "igd", "makespan"), ("overall", "size_group", "profile")),
        "wo_hlns": (("nhv", "igd", "shortage"), ("overall", "buffer_level")),
        "wo_ac": (("nhv", "igd"), ("overall", "size_group")),
    }
    lines = ["H-LNS pre-ablation summary (positive delta = Full better).",
             "Descriptive only; no automatic mechanism verdict.",
             "Wilcoxon pools instance-seed pairs; seeds within one instance are not independent benchmark instances.",
             "Do not infer cross-instance universal significance from pooled p-values alone.", ""]
    for variant in SELECTED_VARIANTS:
        if variant not in focus:
            continue
        metrics, scopes = focus[variant]
        lines.append(variant)
        for row in group_rows:
            if row["variant"] == variant and row["metric"] in metrics and row["scope"] in scopes:
                lines.append(f"  {row['scope']}={row['group']} {row['metric']}: "
                             f"n={row['n_pairs']} mean_delta={row['mean_delta']:.6g} "
                             f"W/T/L={row['wins']}/{row['ties']}/{row['losses']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_analysis(settings: Optional[Settings] = None) -> dict:
    settings = settings or current_settings()
    validate_configuration(settings)
    metric_sanity_checks()
    records, missing = inspect_raw(build_tasks(), settings)
    if missing:
        print(f"Missing or invalid raw results ({len(missing)}):", flush=True)
        for item in missing:
            print(f"  {item['instance']} {item['variant']} seed={item['seed']}: "
                  f"{item['reason']}", flush=True)
        raise RuntimeError("Analysis stopped: raw results are incomplete or incompatible")
    if not records:
        raise RuntimeError("No valid raw results to analyze")
    run_rows = build_run_metrics(records)
    instance_rows = build_instance_summary(run_rows)
    paired = build_pairwise_deltas(run_rows)
    grouped = build_group_summary(paired)
    wtl = [row for row in grouped if row["scope"] == "overall"]
    wilcoxon = build_wilcoxon(paired)
    output = settings.output_root / "analysis"
    if settings.output_root.resolve() not in output.resolve().parents:
        raise ValueError("Analysis path escapes output root")
    summary_fields = ("instance", "size_group", "profile", "buffer_level", "variant", "n_seeds")
    summary_fields += tuple(f"{metric}_{stat}" for metric in
                            ("nhv", "igd", "best_makespan", "best_shortage")
                            for stat in ("mean", "std", "median"))
    summary_fields += ("runtime_mean", "archive_size_mean")
    write_csv(output / "pre_ablation_run_metrics.csv", RUN_FIELDS, run_rows)
    write_csv(output / "pre_ablation_instance_summary.csv", summary_fields, instance_rows)
    write_csv(output / "pre_ablation_pairwise_deltas.csv", DELTA_FIELDS, paired)
    write_csv(output / "pre_ablation_wtl_summary.csv", GROUP_FIELDS, wtl)
    write_csv(output / "pre_ablation_group_summary.csv", GROUP_FIELDS, grouped)
    write_csv(output / "pre_ablation_wilcoxon.csv",
              ("variant", "metric", "n_pairs", "n_nonzero", "statistic", "p_value"),
              wilcoxon)
    (output / "pre_ablation_mechanism_summary.txt").write_text(
        mechanism_summary(grouped), encoding="utf-8")
    print(f"Analysis: {len(run_rows)} runs, {len(instance_rows)} instance summaries, "
          f"{len(paired)} paired comparisons; output={output}", flush=True)
    return {"runs": len(run_rows), "instances": len(instance_rows),
            "pairs": len(paired), "missing": len(missing), "output": str(output)}


def ensure_fixed_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") == PYTHON_HASH_SEED:
        return
    if Path(sys.argv[0]).resolve() != Path(__file__).resolve():
        raise RuntimeError(f"Set PYTHONHASHSEED={PYTHON_HASH_SEED} before imported main()")
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = PYTHON_HASH_SEED
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


def main() -> None:
    if sys.argv[1:]:
        raise ValueError(f"This script uses top-level MODE; unsupported arguments: {sys.argv[1:]}")
    settings = current_settings()
    validate_configuration(settings)
    if MODE in {"run", "all"}:
        ensure_fixed_hash_seed()
        manifest = run_experiments(settings)
        if not manifest["complete"]:
            raise RuntimeError("Run incomplete; analysis withheld")
    if MODE in {"analyze", "all"}:
        run_analysis(settings)


if __name__ == "__main__":
    main()
