#!/usr/bin/env python3
"""Validate the frozen WIP-HFSP shortage objective on random schedules.

This script does not tune ``low_wip`` and does not compare optimization
algorithms. It evaluates three structural questions: decision sensitivity,
information beyond makespan, and agreement with independent flow-state
proxies. Raw shortage is not normalized by capacity, active horizon, job count,
or makespan, so its magnitude should not be used for absolute comparisons
between differently sized instances. The frozen ``low_wip`` is capacity-based;
raw differences between tightness levels are therefore not pure capacity
effects either.
"""

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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.solution.decoder import StageBufferWIPScheduler
from src.solution.encoder import Encoder


DEFAULT_INSTANCES_DIR = PROJECT_ROOT / "data" / "final_benchmark" / "instances"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "shortage_validation"
DEFAULT_SAMPLES = 500
DEFAULT_MASTER_SEED = 20260921
K_MATCHED = 20
EPS = 1e-12

SOLUTION_FIELDS = (
    "instance_id",
    "size_id",
    "profile_id",
    "tightness_id",
    "sample_index",
    "makespan",
    "shortage_total",
    "total_below_low_time",
    "mean_below_low_ratio",
    "mean_buffer_empty_ratio",
    "max_buffer_empty_ratio",
    "total_buffer_empty_time",
    "empty_transition_count",
    "mean_buffer_full_ratio",
    "total_blocking_time",
    "downstream_supply_exposure",
)

INSTANCE_CSV_FIELDS = (
    "instance_id",
    "size_id",
    "profile_id",
    "tightness_id",
    "sample_count",
    "mean_makespan",
    "median_makespan",
    "std_makespan",
    "Q05_makespan",
    "Q95_makespan",
    "D_C",
    "mean_shortage",
    "median_shortage",
    "std_shortage",
    "Q05_shortage",
    "Q95_shortage",
    "D_S",
    "spearman_makespan_shortage",
    "pairwise_conflict_ratio",
    "pairwise_comparable_count",
    "near_makespan_pair_count",
    "median_near_makespan_shortage_diff",
    "Q75_near_makespan_shortage_diff",
    "Q95_near_makespan_shortage_diff",
    "median_near_makespan_shortage_abs_diff",
    "within_band_shortage_spread",
    "spearman_shortage_below_low_time",
    "spearman_shortage_below_low_ratio",
    "spearman_shortage_empty_ratio",
    "spearman_shortage_max_empty_ratio",
    "spearman_shortage_empty_time",
    "spearman_shortage_empty_events",
    "spearman_shortage_supply_exposure",
    "spearman_shortage_blocking",
    "matched_pair_count",
    "matched_pair_unique_count",
    "matched_pair_empty_ratio_consistency",
    "matched_pair_empty_time_consistency",
    "matched_pair_empty_event_consistency",
    "matched_pair_supply_consistency",
    "matched_pair_empty_ratio_tie_ratio",
    "matched_pair_empty_time_tie_ratio",
    "matched_pair_empty_event_tie_ratio",
    "matched_pair_supply_tie_ratio",
)

GLOBAL_METRICS = (
    "D_C",
    "D_S",
    "spearman_makespan_shortage",
    "pairwise_conflict_ratio",
    "median_near_makespan_shortage_diff",
    "within_band_shortage_spread",
    "spearman_shortage_empty_ratio",
    "spearman_shortage_empty_time",
    "spearman_shortage_empty_events",
    "spearman_shortage_supply_exposure",
    "matched_pair_empty_ratio_consistency",
    "matched_pair_empty_time_consistency",
    "matched_pair_empty_event_consistency",
    "matched_pair_supply_consistency",
)


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_formal_instances(instances_dir: Path) -> List[Dict[str, Any]]:
    paths = sorted(instances_dir.glob("*.json"))
    if len(paths) != 96:
        raise ValueError(f"expected 96 formal JSON instances in {instances_dir}, found {len(paths)}")

    entries: List[Dict[str, Any]] = []
    seen_ids = set()
    for path in paths:
        candidate = load_json(path)
        try:
            meta = candidate["meta"]["benchmark"]
            instance_id = str(meta["formal_instance_id"])
            size_id = str(meta["size_id"])
            profile_id = str(meta["profile_id"])
            tightness_id = str(meta["tightness_id"])
            buffers = candidate["buffers"]
            operations = candidate["operations"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid formal instance structure: {path}") from exc

        if instance_id in seen_ids:
            raise ValueError(f"duplicate formal_instance_id: {instance_id}")
        if not buffers:
            raise ValueError(f"{instance_id}: instance has no buffers")
        if not operations:
            raise ValueError(f"{instance_id}: instance has no operations")
        for buffer_id, info in buffers.items():
            capacity = int(info["capacity"])
            low_wip = int(info["low_wip"])
            if not 1 <= low_wip <= capacity:
                raise ValueError(
                    f"{instance_id}/{buffer_id}: expected 1 <= low_wip <= capacity, "
                    f"got {low_wip} and {capacity}"
                )

        seen_ids.add(instance_id)
        entries.append(
            {
                "path": str(path),
                "instance_id": instance_id,
                "size_id": size_id,
                "profile_id": profile_id,
                "tightness_id": tightness_id,
            }
        )

    if len(seen_ids) != 96:
        raise AssertionError("formal instance IDs must be unique")
    return sorted(entries, key=lambda entry: entry["instance_id"])


def percentile(values: Sequence[float], probability: float) -> Optional[float]:
    numeric = sorted(float(value) for value in values)
    if not numeric:
        return None
    if len(numeric) == 1:
        return numeric[0]
    position = (len(numeric) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return numeric[lower]
    fraction = position - lower
    return numeric[lower] * (1.0 - fraction) + numeric[upper] * fraction


def distribution_summary(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "q05": None,
            "q95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "std": statistics.pstdev(numeric),
        "q05": percentile(numeric, 0.05),
        "q95": percentile(numeric, 0.95),
        "min": min(numeric),
        "max": max(numeric),
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


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    rank_left = average_ranks(left)
    rank_right = average_ranks(right)
    mean_left = statistics.fmean(rank_left)
    mean_right = statistics.fmean(rank_right)
    numerator = sum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(rank_left, rank_right)
    )
    denominator = math.sqrt(sum((a - mean_left) ** 2 for a in rank_left)) * math.sqrt(
        sum((b - mean_right) ** 2 for b in rank_right)
    )
    return numerator / denominator if denominator > 0 else 0.0


def count_empty_transitions(
    buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
) -> int:
    transitions = 0
    for events in buffer_trace.values():
        previous_level: Optional[int] = None
        for _, level, _, _ in events:
            current_level = int(level)
            if previous_level is not None and previous_level > 0 and current_level == 0:
                transitions += 1
            previous_level = current_level
    return transitions


def build_downstream_context(
    operations: Dict[str, List[Dict[str, Any]]],
    buffer_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    context: Dict[str, Dict[str, Any]] = {}
    for buffer_id in buffer_ids:
        operation_keys = []
        machine_set = set()
        for job, job_operations in operations.items():
            for op_idx, operation in enumerate(job_operations):
                if operation.get("buffer_in") != buffer_id:
                    continue
                operation_keys.append((job, op_idx))
                machine_set.update(operation["machines"])
        if not operation_keys:
            raise ValueError(f"buffer {buffer_id} has no downstream operation")
        if not machine_set:
            raise ValueError(f"buffer {buffer_id} has no downstream machine")
        context[buffer_id] = {
            "operation_keys": operation_keys,
            "machine_set": sorted(machine_set),
        }
    return context


def compute_downstream_supply_exposure(
    schedule: Sequence[Dict[str, Any]],
    buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
    stats: Dict[str, Any],
    downstream_context: Dict[str, Dict[str, Any]],
) -> float:
    """Compute an event-driven flow-risk proxy, not true machine starvation.

    Exposure accumulates while a buffer is empty, future downstream demand
    remains, and at least one downstream machine is not occupied on its
    ``[start, release)`` interval.
    """

    records_by_operation = {
        (str(record["job"]), int(record["op"])): record for record in schedule
    }
    intervals_by_machine: Dict[str, List[Tuple[int, int]]] = {}
    for record in schedule:
        intervals_by_machine.setdefault(str(record["machine"]), []).append(
            (int(record["start"]), int(record["release"]))
        )

    active_starts = stats["shortage"]["per_buffer_active_start"]
    active_ends = stats["shortage"]["per_buffer_active_end"]
    total_exposure = 0.0

    for buffer_id, definition in downstream_context.items():
        active_start = int(active_starts[buffer_id])
        active_end = int(active_ends[buffer_id])
        if active_end <= active_start:
            continue

        try:
            downstream_records = [
                records_by_operation[key] for key in definition["operation_keys"]
            ]
        except KeyError as exc:
            raise ValueError(f"schedule is missing downstream operation {exc.args[0]}") from exc

        downstream_starts = [int(record["start"]) for record in downstream_records]
        last_downstream_start = max(downstream_starts)
        machine_set = definition["machine_set"]

        # The last event at a timestamp defines the level on the following interval.
        level_after_time: Dict[int, int] = {}
        event_points = {active_start, active_end, *downstream_starts}
        for time_value, level, _, _ in buffer_trace[buffer_id]:
            time_int = int(time_value)
            level_after_time[time_int] = int(level)
            event_points.add(time_int)

        machine_deltas: Dict[int, Dict[str, int]] = {}
        for machine in machine_set:
            for start, release in intervals_by_machine.get(machine, []):
                event_points.add(start)
                event_points.add(release)
                machine_deltas.setdefault(start, {}).setdefault(machine, 0)
                machine_deltas[start][machine] += 1
                machine_deltas.setdefault(release, {}).setdefault(machine, 0)
                machine_deltas[release][machine] -= 1

        points = sorted(event_points)
        occupied = {machine: 0 for machine in machine_set}
        current_level = 0
        for index in range(len(points) - 1):
            left = points[index]
            right = points[index + 1]
            if left in level_after_time:
                current_level = level_after_time[left]
            for machine, delta in machine_deltas.get(left, {}).items():
                occupied[machine] += delta

            interval_start = max(left, active_start)
            interval_end = min(right, active_end)
            if interval_end <= interval_start:
                continue
            remaining_demand = left < last_downstream_start
            capacity_available = any(occupied[machine] == 0 for machine in machine_set)
            if current_level == 0 and remaining_demand and capacity_available:
                total_exposure += interval_end - interval_start

    return total_exposure


def solution_metrics(
    entry: Dict[str, Any],
    sample_index: int,
    makespan: int,
    schedule: Sequence[Dict[str, Any]],
    buffer_trace: Dict[str, List[Tuple[int, int, str, Optional[str]]]],
    stats: Dict[str, Any],
    downstream_context: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    below_ratios = list(stats["shortage"]["per_buffer_below_low_ratio"].values())
    empty_ratios = list(stats["buffers"]["per_buffer_empty_ratio"].values())
    full_ratios = list(stats["buffers"]["per_buffer_full_ratio"].values())
    empty_times = list(stats["buffers"]["per_buffer_empty_time"].values())
    if not below_ratios or not empty_ratios or not full_ratios:
        raise ValueError(f"{entry['instance_id']}: missing per-buffer statistics")

    return {
        "instance_id": entry["instance_id"],
        "size_id": entry["size_id"],
        "profile_id": entry["profile_id"],
        "tightness_id": entry["tightness_id"],
        "sample_index": sample_index,
        "makespan": float(makespan),
        "shortage_total": float(stats["shortage"]["total_shortage_area"]),
        "total_below_low_time": float(stats["shortage"]["total_below_low_time"]),
        "mean_below_low_ratio": statistics.fmean(float(value) for value in below_ratios),
        "mean_buffer_empty_ratio": statistics.fmean(float(value) for value in empty_ratios),
        "max_buffer_empty_ratio": max(float(value) for value in empty_ratios),
        "total_buffer_empty_time": sum(float(value) for value in empty_times),
        "empty_transition_count": count_empty_transitions(buffer_trace),
        "mean_buffer_full_ratio": statistics.fmean(float(value) for value in full_ratios),
        "total_blocking_time": float(stats["blocking"]["total_blocking_time"]),
        "downstream_supply_exposure": compute_downstream_supply_exposure(
            schedule, buffer_trace, stats, downstream_context
        ),
    }


def generate_random_solutions(
    entry: Dict[str, Any], samples: int, master_seed: int
) -> List[Dict[str, Any]]:
    candidate = load_json(Path(entry["path"]))
    operations = candidate["operations"]
    buffers = candidate["buffers"]
    instance_seed = stable_seed(
        master_seed, entry["instance_id"], "shortage_objective_validation"
    )
    rng = random.Random(instance_seed)
    encoder = Encoder(operations, rng=rng)
    scheduler = StageBufferWIPScheduler(operations, buffers)
    downstream_context = build_downstream_context(operations, buffers)
    rows = []
    for sample_index in range(samples):
        os_sequence = encoder.generate_random_os()
        ms_sequence = encoder.generate_random_ms()
        machine_map = encoder.build_ms_map(ms_sequence)
        makespan, schedule, buffer_trace = scheduler.decode(os_sequence, machine_map)
        stats = scheduler.analyze(schedule, buffer_trace, makespan)
        rows.append(
            solution_metrics(
                entry,
                sample_index,
                makespan,
                schedule,
                buffer_trace,
                stats,
                downstream_context,
            )
        )
    return rows


def pairwise_conflict_ratio(rows: Sequence[Dict[str, Any]]) -> Tuple[float, int]:
    conflicts = 0
    comparable = 0
    for left_index in range(len(rows)):
        for right_index in range(left_index + 1, len(rows)):
            delta_makespan = rows[left_index]["makespan"] - rows[right_index]["makespan"]
            delta_shortage = rows[left_index]["shortage_total"] - rows[right_index]["shortage_total"]
            if delta_makespan == 0 or delta_shortage == 0:
                continue
            comparable += 1
            if delta_makespan * delta_shortage < 0:
                conflicts += 1
    return (conflicts / comparable if comparable else 0.0), comparable


def near_makespan_analysis(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ordered_indices = sorted(range(len(rows)), key=lambda index: rows[index]["makespan"])
    pairs = []
    for position, left_index in enumerate(ordered_indices):
        left = rows[left_index]
        for right_index in ordered_indices[position + 1 :]:
            right = rows[right_index]
            relative_makespan = abs(right["makespan"] - left["makespan"]) / min(
                right["makespan"], left["makespan"]
            )
            if relative_makespan > 0.01:
                break
            absolute_shortage = abs(right["shortage_total"] - left["shortage_total"])
            if absolute_shortage == 0:
                continue
            relative_shortage = absolute_shortage / (
                min(right["shortage_total"], left["shortage_total"]) + EPS
            )
            pairs.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "relative_shortage_difference": relative_shortage,
                    "absolute_shortage_difference": absolute_shortage,
                }
            )

    relative_values = [pair["relative_shortage_difference"] for pair in pairs]
    absolute_values = [pair["absolute_shortage_difference"] for pair in pairs]
    return {
        "pairs": pairs,
        "near_makespan_pair_count": len(pairs),
        "median_near_makespan_shortage_diff": (
            statistics.median(relative_values) if relative_values else None
        ),
        "Q75_near_makespan_shortage_diff": percentile(relative_values, 0.75),
        "Q95_near_makespan_shortage_diff": percentile(relative_values, 0.95),
        "median_near_makespan_shortage_abs_diff": (
            statistics.median(absolute_values) if absolute_values else None
        ),
    }


def makespan_band_analysis(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["makespan"])
    band_spreads = []
    for band_index in range(5):
        start = band_index * len(ordered) // 5
        end = (band_index + 1) * len(ordered) // 5
        shortages = [row["shortage_total"] for row in ordered[start:end]]
        if not shortages:
            band_spreads.append(None)
            continue
        median_shortage = statistics.median(shortages)
        q10 = percentile(shortages, 0.10)
        q90 = percentile(shortages, 0.90)
        band_spreads.append((q90 - q10) / (median_shortage + EPS))
    valid = [value for value in band_spreads if value is not None]
    return {
        "band_shortage_spreads": band_spreads,
        "within_band_shortage_spread": statistics.fmean(valid) if valid else None,
    }


def matched_pair_analysis(
    rows: Sequence[Dict[str, Any]], near_pairs: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    ordered = sorted(
        near_pairs,
        key=lambda pair: pair["relative_shortage_difference"],
        reverse=True,
    )
    selected = []
    selected_keys = set()
    used_samples = set()
    for pair in ordered:
        left = pair["left_index"]
        right = pair["right_index"]
        if left in used_samples or right in used_samples:
            continue
        selected.append(pair)
        selected_keys.add((left, right))
        used_samples.update((left, right))
        if len(selected) == K_MATCHED:
            break
    unique_count = len(selected)

    if len(selected) < K_MATCHED:
        for pair in ordered:
            key = (pair["left_index"], pair["right_index"])
            if key in selected_keys:
                continue
            selected.append(pair)
            selected_keys.add(key)
            if len(selected) == K_MATCHED:
                break

    comparisons = {
        "empty_ratio": "mean_buffer_empty_ratio",
        "empty_time": "total_buffer_empty_time",
        "empty_event": "empty_transition_count",
        "supply": "downstream_supply_exposure",
    }
    successes = {name: 0 for name in comparisons}
    ties = {name: 0 for name in comparisons}
    pair_details = []
    for pair in selected:
        first = rows[pair["left_index"]]
        second = rows[pair["right_index"]]
        low, high = (
            (first, second)
            if first["shortage_total"] < second["shortage_total"]
            else (second, first)
        )
        for name, field in comparisons.items():
            if low[field] < high[field]:
                successes[name] += 1
            elif low[field] == high[field]:
                ties[name] += 1
        pair_details.append(
            {
                "low_shortage_sample_index": low["sample_index"],
                "high_shortage_sample_index": high["sample_index"],
                "low_makespan": low["makespan"],
                "high_makespan": high["makespan"],
                "low_shortage": low["shortage_total"],
                "high_shortage": high["shortage_total"],
                "relative_shortage_difference": pair["relative_shortage_difference"],
            }
        )

    count = len(selected)
    result: Dict[str, Any] = {
        "matched_pair_count": count,
        "matched_pair_unique_count": unique_count,
        "matched_pairs": pair_details,
    }
    for name in comparisons:
        result[f"matched_pair_{name}_consistency"] = (
            successes[name] / count if count else None
        )
        result[f"matched_pair_{name}_tie_ratio"] = ties[name] / count if count else None
    return result


def build_instance_summary(
    entry: Dict[str, Any], rows: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    makespans = [float(row["makespan"]) for row in rows]
    shortages = [float(row["shortage_total"]) for row in rows]
    makespan_stats = distribution_summary(makespans)
    shortage_stats = distribution_summary(shortages)
    near = near_makespan_analysis(rows)
    bands = makespan_band_analysis(rows)
    matched = matched_pair_analysis(rows, near["pairs"])
    conflict_ratio, comparable_count = pairwise_conflict_ratio(rows)

    def values(field: str) -> List[float]:
        return [float(row[field]) for row in rows]

    summary = {
        "instance_id": entry["instance_id"],
        "size_id": entry["size_id"],
        "profile_id": entry["profile_id"],
        "tightness_id": entry["tightness_id"],
        "sample_count": len(rows),
        "mean_makespan": makespan_stats["mean"],
        "median_makespan": makespan_stats["median"],
        "std_makespan": makespan_stats["std"],
        "Q05_makespan": makespan_stats["q05"],
        "Q95_makespan": makespan_stats["q95"],
        "D_C": (makespan_stats["q95"] - makespan_stats["q05"])
        / (makespan_stats["median"] + EPS),
        "mean_shortage": shortage_stats["mean"],
        "median_shortage": shortage_stats["median"],
        "std_shortage": shortage_stats["std"],
        "Q05_shortage": shortage_stats["q05"],
        "Q95_shortage": shortage_stats["q95"],
        "D_S": (shortage_stats["q95"] - shortage_stats["q05"])
        / (shortage_stats["median"] + EPS),
        "spearman_makespan_shortage": spearman(makespans, shortages),
        "pairwise_conflict_ratio": conflict_ratio,
        "pairwise_comparable_count": comparable_count,
        "spearman_shortage_below_low_time": spearman(
            shortages, values("total_below_low_time")
        ),
        "spearman_shortage_below_low_ratio": spearman(
            shortages, values("mean_below_low_ratio")
        ),
        "spearman_shortage_empty_ratio": spearman(
            shortages, values("mean_buffer_empty_ratio")
        ),
        "spearman_shortage_max_empty_ratio": spearman(
            shortages, values("max_buffer_empty_ratio")
        ),
        "spearman_shortage_empty_time": spearman(
            shortages, values("total_buffer_empty_time")
        ),
        "spearman_shortage_empty_events": spearman(
            shortages, values("empty_transition_count")
        ),
        "spearman_shortage_supply_exposure": spearman(
            shortages, values("downstream_supply_exposure")
        ),
        "spearman_shortage_blocking": spearman(
            shortages, values("total_blocking_time")
        ),
        **{key: value for key, value in near.items() if key != "pairs"},
        **bands,
        **matched,
    }
    return summary


def summarize_instance_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "instance_count": len(rows),
        "instances_with_no_near_makespan_pairs": sum(
            int(row["near_makespan_pair_count"] == 0) for row in rows
        ),
        "instances_with_D_S_zero_or_near_zero": sum(
            int(float(row["D_S"]) < 1e-6) for row in rows
        ),
        "instances_with_positive_conflict_ratio": sum(
            int(float(row["pairwise_conflict_ratio"]) > 0) for row in rows
        ),
        "instances_with_positive_supply_correlation": sum(
            int(float(row["spearman_shortage_supply_exposure"]) > 0) for row in rows
        ),
        "instances_with_matched_supply_consistency_above_0_5": sum(
            int(
                row["matched_pair_supply_consistency"] is not None
                and float(row["matched_pair_supply_consistency"]) > 0.5
            )
            for row in rows
        ),
        "metric_distributions": {
            metric: distribution_summary(row.get(metric) for row in rows)
            for metric in GLOBAL_METRICS
        },
    }


def build_global_summary(
    summaries: Sequence[Dict[str, Any]],
    samples_per_instance: int,
    master_seed: int,
    workers: int,
) -> Dict[str, Any]:
    def grouped(field: str) -> Dict[str, Any]:
        values = sorted({str(row[field]) for row in summaries})
        return {
            value: summarize_instance_group(
                [row for row in summaries if str(row[field]) == value]
            )
            for value in values
        }

    return {
        "validation_type": "shortage_objective_validation",
        "shortage_definition": "sum_buffer_integral_max_0_low_wip_minus_level",
        "shortage_interpretation": "WIP supply insufficiency / flow-continuity risk proxy",
        "internal_consistency_note": (
            "below-low-time and below-low-ratio correlations are definition-linked "
            "internal consistency metrics, not independent validation evidence"
        ),
        "cross_instance_note": (
            "raw shortage is not normalized by capacity, active horizon, job count, "
            "or makespan and should not be compared as absolute quality across instances"
        ),
        "low_wip_note": (
            "the frozen low_wip is capacity-dependent, so tightness-level raw shortage "
            "differences are not pure capacity effects"
        ),
        "master_seed": master_seed,
        "samples_per_instance": samples_per_instance,
        "workers": workers,
        "overall": summarize_instance_group(summaries),
        "by_size": grouped("size_id"),
        "by_profile": grouped("profile_id"),
        "by_tightness": grouped("tightness_id"),
    }


def select_representative_instances(
    summaries: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    criteria = (
        ("lowest_makespan_shortage_spearman", "spearman_makespan_shortage", False),
        ("highest_makespan_shortage_spearman", "spearman_makespan_shortage", True),
        ("highest_pairwise_conflict", "pairwise_conflict_ratio", True),
        ("highest_supply_exposure_spearman", "spearman_shortage_supply_exposure", True),
        ("lowest_matched_supply_consistency", "matched_pair_supply_consistency", False),
        ("highest_matched_supply_consistency", "matched_pair_supply_consistency", True),
    )
    selected = []
    used = set()
    for reason, field, reverse in criteria:
        candidates = [row for row in summaries if row.get(field) is not None]
        candidates.sort(key=lambda row: float(row[field]), reverse=reverse)
        choice = next((row for row in candidates if row["instance_id"] not in used), None)
        if choice is None:
            continue
        used.add(choice["instance_id"])
        selected.append(
            {
                "instance_id": choice["instance_id"],
                "reason": reason,
                "selection_metric": field,
            }
        )
    return selected


def plot_representative_figures(
    representatives: Sequence[Dict[str, str]],
    summaries: Sequence[Dict[str, Any]],
    solution_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate representative figures") from exc

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_by_id = {row["instance_id"]: row for row in summaries}
    rows_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for row in solution_rows:
        rows_by_id.setdefault(row["instance_id"], []).append(row)

    plot_specs = (
        (
            "makespan",
            "shortage_total",
            "Makespan",
            "Shortage",
            "makespan_shortage",
            "spearman_makespan_shortage",
        ),
        (
            "shortage_total",
            "mean_buffer_empty_ratio",
            "Shortage",
            "Mean buffer empty ratio",
            "shortage_empty_ratio",
            "spearman_shortage_empty_ratio",
        ),
        (
            "shortage_total",
            "downstream_supply_exposure",
            "Shortage",
            "Downstream supply exposure",
            "shortage_supply_exposure",
            "spearman_shortage_supply_exposure",
        ),
    )
    for representative in representatives:
        instance_id = representative["instance_id"]
        rows = rows_by_id[instance_id]
        summary = summary_by_id[instance_id]
        for x_field, y_field, x_label, y_label, suffix, rho_field in plot_specs:
            figure, axis = plt.subplots(figsize=(6.4, 4.8))
            axis.scatter(
                [row[x_field] for row in rows],
                [row[y_field] for row in rows],
                s=12,
                alpha=0.55,
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(f"{instance_id} | Spearman rho={summary[rho_field]:.3f}")
            axis.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(figure_dir / f"{instance_id}_{suffix}.png", dpi=200)
            plt.close(figure)


def analyze_instance_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    entry = payload["entry"]
    rows = generate_random_solutions(entry, payload["samples"], payload["master_seed"])
    return {"rows": rows, "summary": build_instance_summary(entry, rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen shortage objective on random formal schedules."
    )
    parser.add_argument("--instances-dir", type=Path, default=DEFAULT_INSTANCES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    instances_dir = args.instances_dir.resolve()
    output_dir = args.output_dir.resolve()
    entries = load_formal_instances(instances_dir)
    payloads = [
        {"entry": entry, "samples": args.samples, "master_seed": args.master_seed}
        for entry in entries
    ]

    if args.workers == 1:
        results = []
        for index, payload in enumerate(payloads, start=1):
            print(f"[validation] instance {index}/96 {payload['entry']['instance_id']}")
            results.append(analyze_instance_payload(payload))
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, result in enumerate(executor.map(analyze_instance_payload, payloads), start=1):
                results.append(result)
                print(f"[validation] instance {index}/96 complete")

    solution_rows = [row for result in results for row in result["rows"]]
    summaries = [result["summary"] for result in results]
    solution_rows.sort(key=lambda row: (row["instance_id"], row["sample_index"]))
    summaries.sort(key=lambda row: row["instance_id"])

    expected_samples = len(entries) * args.samples
    if len(solution_rows) != expected_samples or len(summaries) != 96:
        raise AssertionError("validation output cardinality mismatch")

    save_csv(output_dir / "solution_samples.csv", solution_rows, SOLUTION_FIELDS)
    save_csv(output_dir / "instance_summary.csv", summaries, INSTANCE_CSV_FIELDS)
    save_json(output_dir / "instance_summary.json", summaries)

    representatives = select_representative_instances(summaries)
    global_summary = build_global_summary(
        summaries, args.samples, args.master_seed, args.workers
    )
    global_summary["representative_instances"] = representatives
    save_json(output_dir / "global_summary.json", global_summary)
    plot_representative_figures(
        representatives, summaries, solution_rows, output_dir
    )
    print(f"[done] wrote shortage-objective validation results to {output_dir}")


if __name__ == "__main__":
    main()
