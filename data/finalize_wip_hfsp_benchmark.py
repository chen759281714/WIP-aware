#!/usr/bin/env python3
"""Finalize 96 WIP-HFSP benchmark instances from the screened Top-5 pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.baseline_nsga2 import BaselineNSGA2
from src.solution.decoder import StageBufferWIPScheduler
from src.solution.encoder import Encoder


LOW_WIP_DIAGNOSTIC_VERSION = "wip_hfsp_finalize_v1"
BUFFER_RELEVANCE_VERSION = "wip_hfsp_buffer_relevance_v2"
HARDNESS_VERSION = "wip_hfsp_hardness_v2"
FINAL_SELECTION_VERSION = "wip_hfsp_finalize_v2"
DEFAULT_MASTER_SEED = 20260917
FINAL_LOW_WIP_ALPHA = 1.0 / 3.0  # Current formal candidate; user confirmation may change it later.
LOW_WIP_LEVELS = {"L25": 0.25, "L33": 1.0 / 3.0, "L50": 0.50}
N_LOW_WIP = 200
N_BUFFER = 200
SMALL_BUDGET = 1000
LARGE_BUDGET = 4000
HARDNESS_REPLICATES = 2
NSGA2_POP_SIZE = 100
HV_REFERENCE_POINT = (1.1, 1.1)
EPS = 1e-12

INPUT_ROOT = PROJECT_ROOT / "data" / "benchmark_candidates"
SELECTION_MANIFEST = INPUT_ROOT / "manifests" / "selection_manifest.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "final_benchmark"


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def save_manifest(output_root: Path, name: str, rows: Sequence[Dict[str, Any]]) -> None:
    manifest_dir = output_root / "manifests"
    save_json(manifest_dir / f"{name}.json", list(rows))
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (manifest_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Sequence[float]) -> Dict[str, float]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "q05": 0.0, "q95": 0.0, "non_zero_ratio": 0.0}
    return {
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "std": statistics.pstdev(numeric),
        "q05": percentile(numeric, 0.05),
        "q95": percentile(numeric, 0.95),
        "non_zero_ratio": sum(value > 0 for value in numeric) / len(numeric),
    }


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = rank
        start = end
    return ranks


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a)) * math.sqrt(sum((y - mean_b) ** 2 for y in b))
    return numerator / denominator if denominator > 0 else 0.0


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson(average_ranks(a), average_ranks(b))


def pairwise_conflict_ratio(makespans: Sequence[float], shortages: Sequence[float]) -> float:
    conflicts = 0
    comparable = 0
    for left in range(len(makespans)):
        for right in range(left + 1, len(makespans)):
            dc = makespans[left] - makespans[right]
            ds = shortages[left] - shortages[right]
            if dc == 0 or ds == 0:
                continue
            comparable += 1
            if dc * ds < 0:
                conflicts += 1
    return conflicts / comparable if comparable else 0.0


def rank_inversion_ratio(relaxed: Sequence[float], finite: Sequence[float]) -> float:
    inversions = 0
    comparable = 0
    for left in range(len(relaxed)):
        for right in range(left + 1, len(relaxed)):
            dr = relaxed[left] - relaxed[right]
            df = finite[left] - finite[right]
            if dr == 0 or df == 0:
                continue
            comparable += 1
            if dr * df < 0:
                inversions += 1
    return inversions / comparable if comparable else 0.0


def with_low_wip(candidate: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    copied = deepcopy(candidate)
    for info in copied["buffers"].values():
        info["low_wip"] = max(1, math.ceil(alpha * int(info["capacity"])))
    return copied


def generate_solutions(operations: Dict[str, Any], count: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    rng = random.Random(seed)
    encoder = Encoder(operations, rng=rng)
    return [(encoder.generate_random_os(), encoder.generate_random_ms()) for _ in range(count)]


def ms_map(operations: Dict[str, Any], ms: Sequence[str]) -> Dict[Tuple[str, int], str]:
    return Encoder(operations).build_ms_map(list(ms))


def metric_summary(makespans: Sequence[float], shortages: Sequence[float]) -> Dict[str, Any]:
    c_stats = distribution(makespans)
    s_stats = distribution(shortages)
    return {
        "mean_makespan": c_stats["mean"],
        "median_makespan": c_stats["median"],
        "D_C": (c_stats["q95"] - c_stats["q05"]) / (c_stats["median"] + EPS),
        "mean_shortage": s_stats["mean"],
        "median_shortage": s_stats["median"],
        "std_shortage": s_stats["std"],
        "Q05_shortage": s_stats["q05"],
        "Q95_shortage": s_stats["q95"],
        "D_S": (s_stats["q95"] - s_stats["q05"]) / (s_stats["median"] + EPS),
        "P_short": s_stats["non_zero_ratio"],
        "spearman_cmax_shortage": spearman(makespans, shortages),
        "pairwise_conflict_ratio": pairwise_conflict_ratio(makespans, shortages),
    }


def cache_matches(path: Path, cache_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    data = load_json(path)
    return data if data.get("cache_key") == cache_key else None


def load_top5() -> List[Dict[str, Any]]:
    rows = load_json(SELECTION_MANIFEST)
    if len(rows) != 96:
        raise ValueError(f"expected 96 cells in selection manifest, found {len(rows)}")
    loaded: List[Dict[str, Any]] = []
    seen_cells = set()
    for row in rows:
        cell_id = row["cell_id"]
        top5 = list(row.get("top_candidate_ids", []))
        if cell_id in seen_cells:
            raise ValueError(f"duplicate cell in selection manifest: {cell_id}")
        if len(top5) != 5 or len(set(top5)) != 5:
            raise ValueError(f"{cell_id}: expected five unique Top-5 candidate IDs")
        seen_cells.add(cell_id)
        for candidate_id in top5:
            path = INPUT_ROOT / "candidates" / cell_id / f"{candidate_id}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            candidate = load_json(path)
            meta = candidate["meta"]["benchmark"]
            if meta["candidate_id"] != candidate_id or meta["cell_id"] != cell_id:
                raise ValueError(f"candidate metadata mismatch: {path}")
            loaded.append({"cell_id": cell_id, "candidate_id": candidate_id, "path": path, "candidate": candidate, "top5": top5})
    if len(loaded) != 480:
        raise AssertionError(f"expected 480 Top-5 candidates, found {len(loaded)}")
    return loaded


def run_low_wip_diagnostic(entry: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    candidate = entry["candidate"]
    meta = candidate["meta"]["benchmark"]
    candidate_id = entry["candidate_id"]
    cache_key = {
        # Keep the original version token so the existing low-WIP caches remain reusable.
        "version": LOW_WIP_DIAGNOSTIC_VERSION,
        "candidate_id": candidate_id,
        "candidate_seed": meta["candidate_seed"],
        "n_solutions": N_LOW_WIP,
        "levels": LOW_WIP_LEVELS,
    }
    path = output_root / "diagnostics" / "low_wip" / f"{candidate_id}.json"
    cached = cache_matches(path, cache_key)
    if cached is not None:
        return cached

    seed = stable_seed(meta["candidate_seed"], "low_wip_sensitivity")
    solutions = generate_solutions(candidate["operations"], N_LOW_WIP, seed)
    schedulers = {
        label: StageBufferWIPScheduler(candidate["operations"], with_low_wip(candidate, alpha)["buffers"])
        for label, alpha in LOW_WIP_LEVELS.items()
    }
    makespans_by_level = {label: [] for label in LOW_WIP_LEVELS}
    shortages_by_level = {label: [] for label in LOW_WIP_LEVELS}

    # low_wip affects analyze(), not decode(); decode once and analyze the same schedule/trace at all levels.
    decode_scheduler = schedulers["L33"]
    for os_seq, ms_list in solutions:
        mapping = ms_map(candidate["operations"], ms_list)
        makespan, schedule, trace = decode_scheduler.decode(os_seq, mapping)
        sample_makespans = []
        for label, scheduler in schedulers.items():
            stats = scheduler.analyze(schedule, trace, makespan)
            makespans_by_level[label].append(float(makespan))
            shortages_by_level[label].append(float(stats["shortage"]["total_shortage_area"]))
            sample_makespans.append(makespan)
        if len(set(sample_makespans)) != 1:
            raise RuntimeError(f"{candidate_id}: makespan changed across low_wip levels")

    levels = {
        label: metric_summary(makespans_by_level[label], shortages_by_level[label])
        for label in LOW_WIP_LEVELS
    }
    result = {
        "cache_key": cache_key,
        "candidate_id": candidate_id,
        "cell_id": entry["cell_id"],
        "diagnostic_seed": seed,
        "valid": True,
        "makespan_consistent": True,
        "levels": levels,
    }
    save_json(path, result)
    return result


def relaxed_hfsp_makespan(
    operations: Dict[str, List[Dict[str, Any]]],
    os_seq: Sequence[str],
    machine_map: Dict[Tuple[str, int], str],
) -> int:
    expected_counts = {job: len(ops) for job, ops in operations.items()}
    actual_counts = {job: 0 for job in operations}
    for job in os_seq:
        if job not in operations:
            raise ValueError(f"relaxed decoder OS contains unknown job {job}")
        actual_counts[job] += 1
    if actual_counts != expected_counts:
        raise ValueError("relaxed decoder OS operation counts do not match the instance")

    expected_ops = {
        (job, op_idx)
        for job, ops in operations.items()
        for op_idx in range(len(ops))
    }
    if set(machine_map) != expected_ops:
        raise ValueError("relaxed decoder machine map does not cover exactly all operations")
    for (job, op_idx), machine in machine_map.items():
        if machine not in operations[job][op_idx]["machines"]:
            raise ValueError(f"invalid machine {machine} for {(job, op_idx)}")

    machines = sorted(
        {
            machine
            for ops in operations.values()
            for op in ops
            for machine in op["machines"]
        }
    )
    job_next = {job: 0 for job in operations}
    job_done = {job: False for job in operations}
    job_ready_at = {job: 0 for job in operations}
    machine_free_at = {machine: 0 for machine in machines}
    running: List[Tuple[int, str, int, str]] = []
    schedule: List[Dict[str, Any]] = []
    t = 0
    os_ptr = 0
    safety_iter = 0
    max_iter = 200000

    while not all(job_done.values()):
        safety_iter += 1
        if safety_iter > max_iter:
            raise RuntimeError("relaxed decoder exceeded its scheduling iteration limit")

        finished = [event for event in running if event[0] == t]
        if finished:
            running = [event for event in running if event[0] != t]
            for _, job, op_idx, _ in finished:
                if op_idx == len(operations[job]) - 1:
                    job_done[job] = True

        while True:
            selected: Optional[Tuple[str, int, str, int]] = None
            for offset in range(len(os_seq)):
                idx = (os_ptr + offset) % len(os_seq)
                job = os_seq[idx]
                if job_done[job]:
                    continue
                op_idx = job_next[job]
                if op_idx >= len(operations[job]) or job_ready_at[job] > t:
                    continue
                machine = machine_map[(job, op_idx)]
                if machine_free_at[machine] > t:
                    continue
                selected = (job, op_idx, machine, (idx + 1) % len(os_seq))
                break

            if selected is None:
                break

            job, op_idx, machine, os_ptr = selected
            op = operations[job][op_idx]
            end = t + int(op["machines"][machine])
            schedule.append(
                {
                    "job": job,
                    "op": op_idx,
                    "machine": machine,
                    "start": t,
                    "end": end,
                    "release": end,
                    "buffer_in": op.get("buffer_in"),
                    "buffer_out": op.get("buffer_out"),
                }
            )
            machine_free_at[machine] = end
            job_ready_at[job] = end
            job_next[job] += 1
            running.append((end, job, op_idx, machine))

        if all(job_done.values()):
            break
        future_times = [event[0] for event in running if event[0] > t]
        if not future_times:
            raise RuntimeError(f"relaxed decoder deadlock at t={t}")
        t = min(future_times)

    if len(schedule) != sum(expected_counts.values()):
        raise RuntimeError("relaxed decoder produced an incomplete schedule")
    return max((record["release"] for record in schedule), default=0)


def run_buffer_relevance(entry: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    source = entry["candidate"]
    candidate = with_low_wip(source, FINAL_LOW_WIP_ALPHA)
    meta = source["meta"]["benchmark"]
    candidate_id = entry["candidate_id"]
    cache_key = {
        "version": BUFFER_RELEVANCE_VERSION,
        "candidate_id": candidate_id,
        "candidate_seed": meta["candidate_seed"],
        "n_solutions": N_BUFFER,
        "final_low_wip_alpha": FINAL_LOW_WIP_ALPHA,
    }
    path = output_root / "diagnostics" / "buffer_relevance" / f"{candidate_id}.json"
    cached = cache_matches(path, cache_key)
    if cached is not None:
        return cached

    seed = stable_seed(meta["candidate_seed"], "finite_buffer_relevance")
    solutions = generate_solutions(candidate["operations"], N_BUFFER, seed)
    scheduler = StageBufferWIPScheduler(candidate["operations"], candidate["buffers"])
    finite_values: List[float] = []
    relaxed_values: List[float] = []
    blocking_values: List[float] = []
    penalties: List[float] = []
    relative_penalties: List[float] = []

    for os_seq, ms_list in solutions:
        mapping = ms_map(candidate["operations"], ms_list)
        finite, schedule, trace = scheduler.decode(os_seq, mapping)
        stats = scheduler.analyze(schedule, trace, finite)
        relaxed = relaxed_hfsp_makespan(candidate["operations"], os_seq, mapping)
        penalty = float(finite - relaxed)
        finite_values.append(float(finite))
        relaxed_values.append(float(relaxed))
        blocking_values.append(float(stats["blocking"]["total_blocking_time"]))
        penalties.append(penalty)
        relative_penalties.append(penalty / max(EPS, relaxed))

    result = {
        "cache_key": cache_key,
        "candidate_id": candidate_id,
        "cell_id": entry["cell_id"],
        "diagnostic_seed": seed,
        "valid": True,
        "finite_relaxed_spearman": spearman(finite_values, relaxed_values),
        "mean_relative_buffer_penalty": statistics.fmean(relative_penalties),
        "positive_penalty_ratio": sum(value > 0 for value in penalties) / len(penalties),
        "mean_absolute_buffer_penalty": statistics.fmean(penalties),
        "rank_inversion_ratio": rank_inversion_ratio(relaxed_values, finite_values),
        "mean_blocking": statistics.fmean(blocking_values),
        "P_block": sum(value > 0 for value in blocking_values) / len(blocking_values),
        "finite_makespan": distribution(finite_values),
        "relaxed_makespan": distribution(relaxed_values),
        "blocking": distribution(blocking_values),
    }
    save_json(path, result)
    return result


def ranking_positions(candidate_ids: Sequence[str], diagnostics: Dict[str, Dict[str, Any]], label: str) -> Dict[str, int]:
    ordered = sorted(
        candidate_ids,
        key=lambda candidate_id: (
            int(diagnostics[candidate_id]["valid"]),
            int(diagnostics[candidate_id]["levels"][label]["D_C"] > 0)
            + int(diagnostics[candidate_id]["levels"][label]["D_S"] > 0),
            min(
                diagnostics[candidate_id]["levels"][label]["D_C"],
                diagnostics[candidate_id]["levels"][label]["D_S"],
            ),
            diagnostics[candidate_id]["levels"][label]["D_S"],
        ),
        reverse=True,
    )
    return {candidate_id: rank for rank, candidate_id in enumerate(ordered, start=1)}


def select_top2(
    entries: Sequence[Dict[str, Any]],
    low_results: Dict[str, Dict[str, Any]],
    buffer_results: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        by_cell.setdefault(entry["cell_id"], []).append(entry)
    selected: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for cell_id, cell_entries in sorted(by_cell.items()):
        candidate_ids = [entry["candidate_id"] for entry in cell_entries]
        positions = {label: ranking_positions(candidate_ids, low_results, label) for label in LOW_WIP_LEVELS}
        correlations = {
            "rank_corr_L25_L33": spearman(
                [positions["L25"][cid] for cid in candidate_ids],
                [positions["L33"][cid] for cid in candidate_ids],
            ),
            "rank_corr_L33_L50": spearman(
                [positions["L33"][cid] for cid in candidate_ids],
                [positions["L50"][cid] for cid in candidate_ids],
            ),
            "rank_corr_L25_L50": spearman(
                [positions["L25"][cid] for cid in candidate_ids],
                [positions["L50"][cid] for cid in candidate_ids],
            ),
        }
        mean_stability = statistics.fmean(correlations.values())
        candidate_robustness: Dict[str, float] = {}
        candidate_rank_details: Dict[str, Dict[str, Any]] = {}
        for cid in candidate_ids:
            ranks = {f"rank_{label}": positions[label][cid] for label in LOW_WIP_LEVELS}
            rank_span = max(ranks.values()) - min(ranks.values())
            robustness = 1.0 - rank_span / 4.0
            candidate_robustness[cid] = robustness
            candidate_rank_details[cid] = {
                **ranks,
                "rank_span": rank_span,
                "candidate_low_wip_robustness": robustness,
            }
            low_results[cid].update(candidate_rank_details[cid])

        def top2_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
            cid = entry["candidate_id"]
            low = low_results[cid]
            buf = buffer_results[cid]
            shortage_robustness = min(low["levels"][label]["D_S"] for label in LOW_WIP_LEVELS)
            return (
                int(low["valid"] and buf["valid"]),
                candidate_robustness[cid],
                shortage_robustness,
                buf["mean_relative_buffer_penalty"],
                buf["rank_inversion_ratio"],
            )

        ordered = sorted(cell_entries, key=top2_key, reverse=True)
        top2 = ordered[:2]
        if len(top2) != 2:
            raise AssertionError(f"{cell_id}: failed to select exactly two candidates")
        selected.extend(top2)
        rows.append(
            {
                "cell_id": cell_id,
                "top5_candidate_ids": candidate_ids,
                "top2_candidate_ids": [entry["candidate_id"] for entry in top2],
                "top2_candidate_low_wip_robustness": {
                    entry["candidate_id"]: candidate_robustness[entry["candidate_id"]]
                    for entry in top2
                },
                "candidate_rank_details": candidate_rank_details,
                **correlations,
                "mean_rank_stability": mean_stability,
                "selection_method": "hierarchical_low_wip_buffer_relevance",
            }
        )
        for entry in cell_entries:
            cid = entry["candidate_id"]
            low_results[cid]["cell_rank_stability"] = {
                **correlations,
                "mean_rank_stability": mean_stability,
            }
    if len(selected) != 192:
        raise AssertionError(f"expected 192 Top-2 candidates, found {len(selected)}")
    return selected, rows


def nondominated(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    unique = sorted(set((float(a), float(b)) for a, b in points))
    result = []
    for point in unique:
        if not any(
            other[0] <= point[0]
            and other[1] <= point[1]
            and (other[0] < point[0] or other[1] < point[1])
            for other in unique
            if other != point
        ):
            result.append(point)
    return sorted(result)


def normalization_bounds(fronts: Sequence[Sequence[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    points = [point for front in fronts for point in front]
    if not points:
        return 0.0, 1.0, 0.0, 1.0
    return (
        min(point[0] for point in points),
        max(point[0] for point in points),
        min(point[1] for point in points),
        max(point[1] for point in points),
    )


def normalize_point(point: Tuple[float, float], bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    m_min, m_max, s_min, s_max = bounds
    return (
        (point[0] - m_min) / (m_max - m_min) if m_max > m_min else 0.0,
        (point[1] - s_min) / (s_max - s_min) if s_max > s_min else 0.0,
    )


def compute_hv(front: Sequence[Tuple[float, float]], bounds: Tuple[float, float, float, float]) -> float:
    points = [normalize_point(point, bounds) for point in front]
    points = nondominated(
        point
        for point in points
        if point[0] <= HV_REFERENCE_POINT[0] and point[1] <= HV_REFERENCE_POINT[1]
    )
    hv = 0.0
    previous_y = HV_REFERENCE_POINT[1]
    for x, y in sorted(points):
        hv += max(0.0, HV_REFERENCE_POINT[0] - x) * max(0.0, previous_y - y)
        previous_y = min(previous_y, y)
    return hv


def compute_igd(
    front: Sequence[Tuple[float, float]],
    reference: Sequence[Tuple[float, float]],
    bounds: Tuple[float, float, float, float],
) -> float:
    if not front or not reference:
        return math.inf
    normalized_front = [normalize_point(point, bounds) for point in front]
    normalized_reference = [normalize_point(point, bounds) for point in reference]
    return statistics.fmean(
        min(math.dist(reference_point, point) for point in normalized_front)
        for reference_point in normalized_reference
    )


def run_nsga2_once(
    candidate: Dict[str, Any],
    candidate_id: str,
    budget: int,
    replicate: int,
    output_root: Path,
) -> Dict[str, Any]:
    meta = candidate["meta"]["benchmark"]
    seed = stable_seed(meta["candidate_seed"], "empirical_hardness", budget, replicate)
    cache_key = {
        "version": HARDNESS_VERSION,
        "candidate_id": candidate_id,
        "candidate_seed": meta["candidate_seed"],
        "budget": budget,
        "replicate": replicate,
        "seed": seed,
        "population_size": NSGA2_POP_SIZE,
        "final_low_wip_alpha": FINAL_LOW_WIP_ALPHA,
        "algorithm": "BaselineNSGA2",
    }
    label = "small" if budget == SMALL_BUDGET else "large"
    path = output_root / "diagnostics" / "hardness" / "runs" / f"{candidate_id}_{label}_r{replicate}.json"
    cached = cache_matches(path, cache_key)
    if cached is not None:
        return cached

    configured = with_low_wip(candidate, FINAL_LOW_WIP_ALPHA)
    search = BaselineNSGA2(
        operations=configured["operations"],
        buffers=configured["buffers"],
        pop_size=NSGA2_POP_SIZE,
        max_evaluations=budget,
        snapshot_interval=None,
        seed=seed,
    )
    search.run(store_stats_init=False, store_stats_generations=False, verbose=False)
    front = nondominated(
        (float(ind.makespan), float(ind.shortage))
        for ind in search.get_pareto_front(search.population)
    )
    result = {
        "cache_key": cache_key,
        "candidate_id": candidate_id,
        "budget_label": label,
        "budget": budget,
        "replicate": replicate,
        "seed": seed,
        "n_evaluations": search.n_evaluations,
        "pareto_front": [[a, b] for a, b in front],
    }
    if search.n_evaluations != budget:
        raise RuntimeError(f"{candidate_id} {label} r{replicate}: expected {budget} FE, got {search.n_evaluations}")
    save_json(path, result)
    return result


def run_hardness_candidate(entry_payload: Dict[str, Any]) -> Dict[str, Any]:
    candidate_path = Path(entry_payload["path"])
    output_root = Path(entry_payload["output_root"])
    candidate = load_json(candidate_path)
    candidate_id = entry_payload["candidate_id"]
    start = time.monotonic()
    small_runs = []
    large_runs = []
    for replicate in range(1, HARDNESS_REPLICATES + 1):
        print(f"[hardness] {candidate_id} small run {replicate}/{HARDNESS_REPLICATES}")
        small_runs.append(run_nsga2_once(candidate, candidate_id, SMALL_BUDGET, replicate, output_root))
    for replicate in range(1, HARDNESS_REPLICATES + 1):
        print(f"[hardness] {candidate_id} large run {replicate}/{HARDNESS_REPLICATES}")
        large_runs.append(run_nsga2_once(candidate, candidate_id, LARGE_BUDGET, replicate, output_root))
    small_fronts = [nondominated(tuple(point) for point in run["pareto_front"]) for run in small_runs]
    large_fronts = [nondominated(tuple(point) for point in run["pareto_front"]) for run in large_runs]
    reference = nondominated(point for front in large_fronts for point in front)
    bounds = normalization_bounds(small_fronts + large_fronts + [reference])
    small_hv = [compute_hv(front, bounds) for front in small_fronts]
    large_hv = [compute_hv(front, bounds) for front in large_fronts]
    small_igd = [compute_igd(front, reference, bounds) for front in small_fronts]
    large_igd = [compute_igd(front, reference, bounds) for front in large_fronts]
    small_mean_hv = statistics.fmean(small_hv)
    large_mean_hv = statistics.fmean(large_hv)
    small_mean_igd = statistics.fmean(small_igd)
    large_mean_igd = statistics.fmean(large_igd)
    result = {
        "candidate_id": candidate_id,
        "cell_id": entry_payload["cell_id"],
        "valid": True,
        "small_budget": SMALL_BUDGET,
        "large_budget": LARGE_BUDGET,
        "replicates": HARDNESS_REPLICATES,
        "small_HV": small_hv,
        "large_HV": large_hv,
        "small_IGD": small_igd,
        "large_IGD": large_igd,
        "small_mean_HV": small_mean_hv,
        "large_mean_HV": large_mean_hv,
        "small_mean_IGD": small_mean_igd,
        "large_mean_IGD": large_mean_igd,
        "HV_gap": (large_mean_hv - small_mean_hv) / (large_mean_hv + EPS),
        "IGD_gap": small_mean_igd - large_mean_igd,
        "reference_front_size": len(reference),
        "normalization_bounds": {
            "makespan_min": bounds[0],
            "makespan_max": bounds[1],
            "shortage_min": bounds[2],
            "shortage_max": bounds[3],
        },
        "elapsed_seconds": time.monotonic() - start,
    }
    save_json(output_root / "diagnostics" / "hardness" / f"{candidate_id}.json", result)
    return result


def run_all_hardness(entries: Sequence[Dict[str, Any]], output_root: Path, workers: int) -> Dict[str, Dict[str, Any]]:
    payloads = [
        {
            "candidate_id": entry["candidate_id"],
            "cell_id": entry["cell_id"],
            "path": str(entry["path"]),
            "output_root": str(output_root),
        }
        for entry in entries
    ]
    results: Dict[str, Dict[str, Any]] = {}
    if workers <= 1:
        for index, payload in enumerate(payloads, start=1):
            print(f"[hardness] candidate {index}/{len(payloads)} {payload['candidate_id']}")
            results[payload["candidate_id"]] = run_hardness_candidate(payload)
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_hardness_candidate, payload): payload for payload in payloads}
        completed = 0
        for future in as_completed(futures):
            payload = futures[future]
            result = future.result()
            results[payload["candidate_id"]] = result
            completed += 1
            print(f"[hardness] candidate {completed}/{len(payloads)} {payload['candidate_id']} complete")
    return results


def diagnostic_worker(payload: Dict[str, Any], diagnostic: str) -> Dict[str, Any]:
    path = Path(payload["path"])
    entry = {
        "cell_id": payload["cell_id"],
        "candidate_id": payload["candidate_id"],
        "path": path,
        "candidate": load_json(path),
    }
    output_root = Path(payload["output_root"])
    if diagnostic == "low_wip":
        return run_low_wip_diagnostic(entry, output_root)
    if diagnostic == "buffer_relevance":
        return run_buffer_relevance(entry, output_root)
    raise ValueError(diagnostic)


def run_parallel_diagnostics(
    entries: Sequence[Dict[str, Any]],
    output_root: Path,
    workers: int,
    diagnostic: str,
) -> Dict[str, Dict[str, Any]]:
    payloads = [
        {
            "cell_id": entry["cell_id"],
            "candidate_id": entry["candidate_id"],
            "path": str(entry["path"]),
            "output_root": str(output_root),
        }
        for entry in entries
    ]
    results: Dict[str, Dict[str, Any]] = {}
    label = "low-wip" if diagnostic == "low_wip" else "buffer-relevance"
    if workers <= 1:
        for index, payload in enumerate(payloads, start=1):
            print(f"[{label}] candidate {index}/{len(payloads)} {payload['candidate_id']}")
            result = diagnostic_worker(payload, diagnostic)
            results[payload["candidate_id"]] = result
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(diagnostic_worker, payload, diagnostic): payload
            for payload in payloads
        }
        completed = 0
        for future in as_completed(futures):
            payload = futures[future]
            result = future.result()
            results[payload["candidate_id"]] = result
            completed += 1
            print(f"[{label}] candidate {completed}/{len(payloads)} {payload['candidate_id']}")
    return results


def low_wip_manifest_rows(
    entries: Sequence[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for entry in entries:
        result = results[entry["candidate_id"]]
        row: Dict[str, Any] = {
            "cell_id": entry["cell_id"],
            "candidate_id": entry["candidate_id"],
            "valid": result["valid"],
            "makespan_consistent": result["makespan_consistent"],
        }
        for label in LOW_WIP_LEVELS:
            for name, value in result["levels"][label].items():
                row[f"{name}_{label}"] = value
        stability = result.get("cell_rank_stability", {})
        row.update(stability)
        for label in LOW_WIP_LEVELS:
            row[f"rank_{label}"] = result[f"rank_{label}"]
        row["rank_span"] = result["rank_span"]
        row["candidate_low_wip_robustness"] = result["candidate_low_wip_robustness"]
        rows.append(row)
    return rows


def buffer_manifest_rows(
    entries: Sequence[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    names = (
        "finite_relaxed_spearman",
        "mean_relative_buffer_penalty",
        "positive_penalty_ratio",
        "mean_absolute_buffer_penalty",
        "rank_inversion_ratio",
        "mean_blocking",
        "P_block",
    )
    return [
        {
            "cell_id": entry["cell_id"],
            "candidate_id": entry["candidate_id"],
            "valid": results[entry["candidate_id"]]["valid"],
            **{name: results[entry["candidate_id"]][name] for name in names},
        }
        for entry in entries
    ]


def hardness_manifest_rows(
    entries: Sequence[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    names = (
        "small_mean_HV",
        "large_mean_HV",
        "small_mean_IGD",
        "large_mean_IGD",
        "HV_gap",
        "IGD_gap",
        "reference_front_size",
        "elapsed_seconds",
    )
    return [
        {
            "cell_id": entry["cell_id"],
            "candidate_id": entry["candidate_id"],
            **{name: results[entry["candidate_id"]][name] for name in names},
        }
        for entry in entries
    ]


def select_final(
    top2_entries: Sequence[Dict[str, Any]],
    top2_rows: Sequence[Dict[str, Any]],
    low_results: Dict[str, Dict[str, Any]],
    buffer_results: Dict[str, Dict[str, Any]],
    hardness_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    stability_by_cell = {row["cell_id"]: row for row in top2_rows}
    entries_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for entry in top2_entries:
        entries_by_cell.setdefault(entry["cell_id"], []).append(entry)

    selections = []
    for cell_id, entries in sorted(entries_by_cell.items()):
        stability = stability_by_cell[cell_id]["mean_rank_stability"]

        def final_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
            cid = entry["candidate_id"]
            low = low_results[cid]
            buf = buffer_results[cid]
            hard = hardness_results[cid]
            shortage_robustness = min(low["levels"][label]["D_S"] for label in LOW_WIP_LEVELS)
            hardness_informative = int(hard["HV_gap"] > 0 and hard["IGD_gap"] > 0)
            return (
                int(low["valid"] and buf["valid"] and hard["valid"]),
                low["candidate_low_wip_robustness"],
                shortage_robustness,
                buf["mean_relative_buffer_penalty"],
                buf["rank_inversion_ratio"],
                hardness_informative,
                hard["HV_gap"],
            )

        ordered = sorted(entries, key=final_key, reverse=True)
        selected = ordered[0]
        cid = selected["candidate_id"]
        low = low_results[cid]
        buf = buffer_results[cid]
        hard = hardness_results[cid]
        formal_id = f"WIPHFSP_{cell_id}"
        selections.append(
            {
                "cell_id": cell_id,
                "top5_candidate_ids": stability_by_cell[cell_id]["top5_candidate_ids"],
                "top2_candidate_ids": stability_by_cell[cell_id]["top2_candidate_ids"],
                "selected_candidate_id": cid,
                "source_path": str(selected["path"]),
                "formal_instance_id": formal_id,
                "formal_filename": f"{formal_id}.json",
                "low_wip_rank_corr_L25_L33": stability_by_cell[cell_id]["rank_corr_L25_L33"],
                "low_wip_rank_corr_L33_L50": stability_by_cell[cell_id]["rank_corr_L33_L50"],
                "low_wip_rank_corr_L25_L50": stability_by_cell[cell_id]["rank_corr_L25_L50"],
                "mean_rank_stability": stability,
                "selected_candidate_low_wip_robustness": low["candidate_low_wip_robustness"],
                "selected_rank_L25": low["rank_L25"],
                "selected_rank_L33": low["rank_L33"],
                "selected_rank_L50": low["rank_L50"],
                "selected_rank_span": low["rank_span"],
                "selected_D_S_L25": low["levels"]["L25"]["D_S"],
                "selected_D_S_L33": low["levels"]["L33"]["D_S"],
                "selected_D_S_L50": low["levels"]["L50"]["D_S"],
                "selected_finite_relaxed_spearman": buf["finite_relaxed_spearman"],
                "selected_mean_relative_buffer_penalty": buf["mean_relative_buffer_penalty"],
                "selected_positive_penalty_ratio": buf["positive_penalty_ratio"],
                "selected_rank_inversion_ratio": buf["rank_inversion_ratio"],
                "selected_mean_blocking": buf["mean_blocking"],
                "selected_P_block": buf["P_block"],
                "small_mean_HV": hard["small_mean_HV"],
                "large_mean_HV": hard["large_mean_HV"],
                "small_mean_IGD": hard["small_mean_IGD"],
                "large_mean_IGD": hard["large_mean_IGD"],
                "HV_gap": hard["HV_gap"],
                "IGD_gap": hard["IGD_gap"],
                "hardness_informative": int(hard["HV_gap"] > 0 and hard["IGD_gap"] > 0),
                "selection_reason": "candidate robustness, shortage robustness, finite-buffer relevance, rank inversion, hardness tie-break",
                "empirical_hardness_used": True,
            }
        )
    if len(selections) != 96 or len({row["cell_id"] for row in selections}) != 96:
        raise AssertionError("final selection must contain exactly one candidate for each of 96 cells")
    return selections


def copy_formal_instances(
    selections: Sequence[Dict[str, Any]],
    output_root: Path,
    timestamp: str,
) -> List[Dict[str, Any]]:
    instances_dir = output_root / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    for stale in instances_dir.glob("*.json"):
        stale.unlink()

    mappings = []
    for selection in selections:
        source_path = Path(selection["source_path"])
        source = load_json(source_path)
        source_meta = source["meta"]["benchmark"]
        formal = with_low_wip(source, FINAL_LOW_WIP_ALPHA)
        formal_meta = formal["meta"]["benchmark"]
        formal_meta.update(
            {
                "formal_instance_id": selection["formal_instance_id"],
                "source_candidate_id": selection["selected_candidate_id"],
                "source_candidate_seed": source_meta["candidate_seed"],
                "finalized": True,
                "final_low_wip_alpha": FINAL_LOW_WIP_ALPHA,
                "final_low_wip_rule": "ceil(alpha_times_capacity)",
                "final_selection_version": FINAL_SELECTION_VERSION,
                "selection_timestamp_utc": timestamp,
            }
        )
        destination = instances_dir / selection["formal_filename"]
        save_json(destination, formal)
        mappings.append(
            {
                "formal_instance_id": selection["formal_instance_id"],
                "formal_filename": selection["formal_filename"],
                "cell_id": selection["cell_id"],
                "size_id": source_meta["size_id"],
                "profile_id": source_meta["profile_id"],
                "tightness_id": source_meta["tightness_id"],
                "source_candidate_id": selection["selected_candidate_id"],
                "source_candidate_path": str(source_path.relative_to(PROJECT_ROOT)),
                "source_candidate_seed": source_meta["candidate_seed"],
            }
        )
    return mappings


def validate_formal_instances(output_root: Path, master_seed: int) -> None:
    paths = sorted((output_root / "instances").glob("*.json"))
    if len(paths) != 96:
        raise AssertionError(f"expected 96 formal JSON files, found {len(paths)}")
    ids = set()
    for path in paths:
        candidate = load_json(path)
        meta = candidate["meta"]["benchmark"]
        formal_id = meta["formal_instance_id"]
        if formal_id in ids:
            raise AssertionError(f"duplicate formal instance ID: {formal_id}")
        ids.add(formal_id)
        for info in candidate["buffers"].values():
            if not 1 <= int(info["low_wip"]) <= int(info["capacity"]):
                raise AssertionError(f"{formal_id}: low_wip outside valid range")

        seed = stable_seed(master_seed, formal_id, "final_validation")
        solution = generate_solutions(candidate["operations"], 1, seed)[0]
        scheduler = StageBufferWIPScheduler(candidate["operations"], candidate["buffers"])
        makespan, schedule, trace = scheduler.decode(
            solution[0], ms_map(candidate["operations"], solution[1])
        )
        stats = scheduler.analyze(schedule, trace, makespan)
        if len(schedule) != sum(len(ops) for ops in candidate["operations"].values()):
            raise AssertionError(f"{formal_id}: incomplete validation schedule")
        if stats["makespan"] != makespan:
            raise AssertionError(f"{formal_id}: validation makespan mismatch")
    if len(ids) != 96:
        raise AssertionError("formal instance IDs must be unique")


def summarize_rows(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, float]:
    return distribution([float(row[field]) for row in rows])


def build_summary(
    low_rows: Sequence[Dict[str, Any]],
    buffer_rows: Sequence[Dict[str, Any]],
    hardness_rows: Sequence[Dict[str, Any]],
    top2_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "version": FINAL_SELECTION_VERSION,
        "low_wip": {
            label: {
                "mean_shortage": summarize_rows(low_rows, f"mean_shortage_{label}"),
                "D_S": summarize_rows(low_rows, f"D_S_{label}"),
            }
            for label in LOW_WIP_LEVELS
        },
        "rank_stability": summarize_rows(top2_rows, "mean_rank_stability"),
        "candidate_low_wip_robustness": summarize_rows(
            low_rows, "candidate_low_wip_robustness"
        ),
        "highly_sensitive_cell_count": sum(float(row["mean_rank_stability"]) < 0.5 for row in top2_rows),
        "buffer_relevance": {
            field: summarize_rows(buffer_rows, field)
            for field in (
                "finite_relaxed_spearman",
                "mean_relative_buffer_penalty",
                "positive_penalty_ratio",
                "rank_inversion_ratio",
            )
        },
        "hardness": {
            field: summarize_rows(hardness_rows, field)
            for field in (
                "small_mean_HV",
                "large_mean_HV",
                "small_mean_IGD",
                "large_mean_IGD",
                "HV_gap",
                "IGD_gap",
            )
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the 96-instance WIP-HFSP benchmark.")
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    start = time.monotonic()
    print(f"[load] selection_manifest={SELECTION_MANIFEST}")
    entries = load_top5()
    print(f"[load] loaded 96 cells / {len(entries)} candidates")

    low_results = run_parallel_diagnostics(entries, output_root, args.workers, "low_wip")
    buffer_results = run_parallel_diagnostics(entries, output_root, args.workers, "buffer_relevance")
    top2_entries, top2_rows = select_top2(entries, low_results, buffer_results)
    print(f"[top2] selected {len(top2_entries)} candidates")

    low_rows = low_wip_manifest_rows(entries, low_results)
    buffer_rows = buffer_manifest_rows(entries, buffer_results)
    save_manifest(output_root, "low_wip_sensitivity", low_rows)
    save_manifest(output_root, "buffer_relevance", buffer_rows)
    save_manifest(output_root, "top2_manifest", top2_rows)

    hardness_results = run_all_hardness(top2_entries, output_root, args.workers)
    hardness_rows = hardness_manifest_rows(top2_entries, hardness_results)
    save_manifest(output_root, "hardness_results", hardness_rows)

    selections = select_final(top2_entries, top2_rows, low_results, buffer_results, hardness_results)
    print(f"[final] selected {len(selections)} formal instances")
    timestamp = datetime.now(timezone.utc).isoformat()
    source_mapping = copy_formal_instances(selections, output_root, timestamp)
    save_manifest(output_root, "final_selection_manifest", selections)
    save_manifest(output_root, "source_mapping", source_mapping)
    print(f"[copy] copied {len(source_mapping)} files to {output_root / 'instances'}")

    if len(top2_entries) != 192 or len(selections) != 96 or len(source_mapping) != 96:
        raise AssertionError("final benchmark cardinality check failed")
    if len({row["selected_candidate_id"] for row in selections}) != 96:
        raise AssertionError("selected source candidates must be unique")
    validate_formal_instances(output_root, args.master_seed)
    summary = build_summary(low_rows, buffer_rows, hardness_rows, top2_rows)
    summary.update(
        {
            "master_seed": args.master_seed,
            "workers": args.workers,
            "top5_count": len(entries),
            "top2_count": len(top2_entries),
            "final_count": len(selections),
            "elapsed_seconds": time.monotonic() - start,
            "final_low_wip_alpha": FINAL_LOW_WIP_ALPHA,
        }
    )
    save_json(output_root / "manifests" / "finalization_summary.json", summary)
    print(f"[done] elapsed={summary['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
