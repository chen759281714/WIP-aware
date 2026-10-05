"""Pre-ablation screening experiment for MGS, SIS, and archive cooperation.

Verify stable, directional contributions before formal parameter calibration.
This is not the final ablation experiment.
"""

from __future__ import annotations

import csv
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
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation
from src.algorithms.wip_graph_dual_population_wo_ac import WIPGraphDualPopulationWithoutAC
from src.algorithms.wip_graph_dual_population_wo_mgs import WIPGraphDualPopulationWithoutMGS
from src.algorithms.wip_graph_dual_population_wo_sis import WIPGraphDualPopulationWithoutSIS


# ============================================================
# USER CONFIGURATION
# ============================================================
MODE = "run"  # "run", "analyze", or "all"
VARIANT_REGISTRY = {
    "full": WIPGraphDualPopulation,
    "wo_mgs": WIPGraphDualPopulationWithoutMGS,
    "wo_sis": WIPGraphDualPopulationWithoutSIS,
    "wo_ac": WIPGraphDualPopulationWithoutAC,
}
SELECTED_VARIANTS = ["full", "wo_mgs", "wo_sis", "wo_ac"]
SELECTED_INSTANCES = [
    "WIPHFSP_S1_P1_B1", "WIPHFSP_S1_P2_B2",
    "WIPHFSP_S1_P3_B3", "WIPHFSP_S1_P4_B1",
    "WIPHFSP_S4_P1_B2", "WIPHFSP_S4_P2_B3",
    "WIPHFSP_S4_P3_B1", "WIPHFSP_S4_P4_B2",
    "WIPHFSP_S8_P1_B3", "WIPHFSP_S8_P2_B1",
    "WIPHFSP_S8_P3_B2", "WIPHFSP_S8_P4_B3",
]
SEEDS = [0, 1, 2, 3, 4]
FE_MAX = 10000

# Matches the current formal comparison runner (POP_SIZE=200, N=POP_SIZE//2).
N = 100
N_A = 200
P_MU = 0.1
RHO = 0.8
ETA = 0.3
T_COOP = 10

USE_MULTIPROCESSING = True
N_WORKERS = 5
INSTANCE_DIR = PROJECT_ROOT / "data/final_benchmark/instances"
OUTPUT_ROOT = PROJECT_ROOT / "experiments/results/pre_ablation"
OVERWRITE_EXISTING = False
SAVE_FINAL_ARCHIVE = True  # Required for pooled HV and IGD analysis.
ANALYZE_ALLOW_INCOMPLETE = False
HV_REFERENCE_POINT = (1.1, 1.1)
EPS = 1e-12
TOL = 1e-12
PYTHON_HASH_SEED = "0"  # Fixed at process start for serial/worker equivalence.
# ============================================================


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
    eta: float
    t_coop: int

    def algorithm_parameters(self, seed: int) -> dict:
        return {
            "N": self.n, "N_A": self.n_a, "p_mut": self.p_mu,
            "rho": self.rho, "eta": self.eta, "T_coop": self.t_coop,
            "FE_max": self.fe_max, "seed": seed,
        }


def current_settings() -> Settings:
    return Settings(INSTANCE_DIR, OUTPUT_ROOT, FE_MAX, N, N_A, P_MU,
                    RHO, ETA, T_COOP)


def validate_configuration(settings: Settings) -> None:
    if MODE not in {"run", "analyze", "all"}:
        raise ValueError(f"Invalid MODE: {MODE}")
    for name, values in (("variants", SELECTED_VARIANTS),
                         ("instances", SELECTED_INSTANCES), ("seeds", SEEDS)):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be nonempty and unique")
    if any(variant not in VARIANT_REGISTRY for variant in SELECTED_VARIANTS):
        raise ValueError("Unknown selected variant")
    if any(not isinstance(seed, int) for seed in SEEDS):
        raise ValueError("SEEDS must contain integers")
    if settings.n < 1 or settings.n_a < 2 or settings.fe_max < 2 * settings.n:
        raise ValueError("Require N>=1, N_A>=2, FE_MAX>=2*N")
    if N_WORKERS < 1 or not SAVE_FINAL_ARCHIVE:
        raise ValueError("N_WORKERS>=1 and SAVE_FINAL_ARCHIVE=True are required")
    if tuple(HV_REFERENCE_POINT) != (1.1, 1.1) or EPS <= 0 or TOL < 0:
        raise ValueError("Expected HV reference (1.1,1.1), EPS>0, TOL>=0")
    if MODE in {"run", "all"}:
        missing = [name for name in SELECTED_INSTANCES
                   if not (settings.instance_dir / f"{name}.json").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing benchmark instances: {missing}")


def build_tasks() -> List[Task]:
    return [Task(instance, variant, seed)
            for instance in SELECTED_INSTANCES
            for variant in SELECTED_VARIANTS
            for seed in SEEDS]


def raw_path(task: Task, settings: Settings) -> Path:
    return (settings.output_root / "raw" / task.instance / task.variant
            / f"seed_{task.seed}.json")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validated_points(archive: Sequence[dict]) -> List[Tuple[float, float]]:
    if not isinstance(archive, list) or not archive:
        raise ValueError("final_archive must be a nonempty list")
    points = []
    for solution in archive:
        x, y = float(solution["makespan"]), float(solution["shortage"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Nonfinite objective in final_archive")
        points.append((x, y))
    return points


def load_valid_raw(path: Path, task: Task, settings: Settings) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    for key in ("instance", "variant", "seed", "FE_max", "actual_FE",
                "runtime_seconds", "final_archive_size", "best_makespan",
                "best_shortage", "final_archive", "algorithm_parameters"):
        if key not in record:
            raise ValueError(f"Missing field {key}")
    if ((record["instance"], record["variant"], record["seed"])
            != (task.instance, task.variant, task.seed)):
        raise ValueError("Task identity mismatch")
    if record["FE_max"] != settings.fe_max or record["actual_FE"] != settings.fe_max:
        raise ValueError("FE mismatch")
    if record["algorithm_parameters"] != settings.algorithm_parameters(task.seed):
        raise ValueError("Algorithm parameter mismatch")
    points = validated_points(record["final_archive"])
    if record["final_archive_size"] != len(points):
        raise ValueError("Archive size mismatch")
    if len(points) > settings.n_a:
        raise ValueError("Archive exceeds N_A")
    if (record["best_makespan"] != min(x for x, _ in points)
            or record["best_shortage"] != min(y for _, y in points)):
        raise ValueError("Archive extreme mismatch")
    if not math.isfinite(float(record["runtime_seconds"])) or record["runtime_seconds"] < 0:
        raise ValueError("Invalid runtime")
    return record


def run_single_task(task: Task, settings: Settings) -> dict:
    start = time.perf_counter()
    try:
        instance_path = settings.instance_dir / f"{task.instance}.json"
        with instance_path.open("r", encoding="utf-8") as handle:
            instance = json.load(handle)
        parameters = settings.algorithm_parameters(task.seed)
        algorithm = VARIANT_REGISTRY[task.variant](instance["operations"],
                                                   instance["buffers"], **parameters)
        archive = algorithm.run()
        points = [(float(ind.makespan), float(ind.shortage)) for ind in archive]
        if algorithm.n_evaluations != settings.fe_max or not points:
            raise ValueError("Incomplete FE budget or empty final archive")
        if (algorithm.diagnostics["algorithm_variant"] != task.variant
                or len(algorithm.P_M) != settings.n or len(algorithm.P_S) != settings.n
                or len(archive) > settings.n_a):
            raise ValueError("Variant identity or population/archive size mismatch")
        record = {
            "instance": task.instance,
            "variant": task.variant,
            "seed": task.seed,
            "FE_max": settings.fe_max,
            "actual_FE": algorithm.n_evaluations,
            "runtime_seconds": time.perf_counter() - start,
            "final_archive_size": len(points),
            "best_makespan": min(x for x, _ in points),
            "best_shortage": min(y for _, y in points),
            "final_archive": [{"makespan": x, "shortage": y} for x, y in points],
            "algorithm_parameters": parameters,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "diagnostics": {
                "algorithm_variant": algorithm.diagnostics["algorithm_variant"],
                "pm_specialized_kind": algorithm.diagnostics["pm_specialized_kind"],
                "ps_specialized_kind": algorithm.diagnostics["ps_specialized_kind"],
                "pm_archive_guidance_enabled": algorithm.diagnostics[
                    "pm_archive_guidance_enabled"],
                "ps_archive_guidance_enabled": algorithm.diagnostics[
                    "ps_archive_guidance_enabled"],
                "pm_moves": algorithm.diagnostics["pm_moves"],
                "ps_moves": algorithm.diagnostics["ps_moves"],
                "cooperation": algorithm.diagnostics["cooperation"],
            },
        }
        write_json_atomic(raw_path(task, settings), record)
        return {"task": task.__dict__, "status": "completed",
                "runtime_seconds": record["runtime_seconds"]}
    except Exception as exc:
        return {"task": task.__dict__, "status": "failed", "exception": repr(exc)}


def inspect_raw(tasks: Sequence[Task], settings: Settings) -> Tuple[Dict[Task, dict], List[dict]]:
    found, missing = {}, []
    for task in tasks:
        path = raw_path(task, settings)
        try:
            found[task] = load_valid_raw(path, task, settings)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            missing.append({**task.__dict__, "reason": str(exc)})
    return found, missing


def run_experiments(settings: Optional[Settings] = None) -> dict:
    settings = settings or current_settings()
    tasks = build_tasks()
    start = time.perf_counter()
    pending, skipped = [], []
    for task in tasks:
        if not OVERWRITE_EXISTING:
            try:
                load_valid_raw(raw_path(task, settings), task, settings)
                skipped.append(task)
                continue
            except (OSError, ValueError, KeyError, TypeError):
                pass
        pending.append(task)
    completed, failed = [], []
    total = len(tasks)
    print(f"Pre-ablation: {total} expected, {len(skipped)} skipped, "
          f"{len(pending)} to run", flush=True)

    def report(result: dict) -> None:
        (completed if result["status"] == "completed" else failed).append(result)
        task = result["task"]
        duration = result.get("runtime_seconds", 0.0)
        print(f"[{len(skipped) + len(completed) + len(failed)}/{total}] "
              f"{result['status']} {task['instance']} {task['variant']} "
              f"seed={task['seed']} runtime={duration:.2f}s", flush=True)

    if USE_MULTIPROCESSING and pending:
        with ProcessPoolExecutor(max_workers=N_WORKERS,
                                 mp_context=get_context("spawn")) as executor:
            futures = {executor.submit(run_single_task, task, settings): task
                       for task in pending}
            for future in as_completed(futures):
                try:
                    report(future.result())
                except Exception as exc:
                    report({"task": futures[future].__dict__,
                            "status": "failed", "exception": repr(exc)})
    else:
        for task in pending:
            report(run_single_task(task, settings))

    _, missing = inspect_raw(tasks, settings)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected": total, "completed": len(completed), "skipped": len(skipped),
        "failed": len(failed), "missing": len(missing),
        "total_runtime_seconds": time.perf_counter() - start,
        "failed_tasks": failed, "missing_tasks": missing,
    }
    write_json_atomic(settings.output_root / "run_manifest.json", manifest)
    print("Run summary: " + ", ".join(f"{k}={manifest[k]}" for k in
          ("expected", "completed", "skipped", "failed", "missing")), flush=True)
    return manifest


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
        "wo_sis": (("nhv", "igd", "shortage"), ("overall", "buffer_level")),
        "wo_ac": (("nhv", "igd"), ("overall", "size_group")),
    }
    lines = ["Pre-ablation directional summary (positive delta = Full better).",
             "Descriptive only; no automatic mechanism verdict.", ""]
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
    metric_sanity_checks()
    records, missing = inspect_raw(build_tasks(), settings)
    if missing:
        print(f"Missing or invalid raw results ({len(missing)}):", flush=True)
        for item in missing:
            print(f"  {item['instance']} {item['variant']} seed={item['seed']}: "
                  f"{item['reason']}", flush=True)
        if not ANALYZE_ALLOW_INCOMPLETE:
            raise RuntimeError("Analysis stopped: raw results are incomplete")
    if not records:
        raise RuntimeError("No valid raw results to analyze")
    run_rows = build_run_metrics(records)
    instance_rows = build_instance_summary(run_rows)
    paired = build_pairwise_deltas(run_rows)
    grouped = build_group_summary(paired)
    wtl = [row for row in grouped if row["scope"] == "overall"]
    wilcoxon = build_wilcoxon(paired)
    output = settings.output_root / "analysis"
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
    settings = current_settings()
    validate_configuration(settings)
    if MODE in {"run", "all"}:
        ensure_fixed_hash_seed()
        run_experiments(settings)
    if MODE in {"analyze", "all"}:
        run_analysis(settings)


if __name__ == "__main__":
    main()
