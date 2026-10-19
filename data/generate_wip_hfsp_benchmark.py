#!/usr/bin/env python3
"""Reproducible benchmark candidate generation and screening for WIP-HFSP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.solution.decoder import StageBufferWIPScheduler
from src.solution.encoder import Encoder


GENERATOR_VERSION = "wip_hfsp_v1"
DEFAULT_MASTER_SEED = 20260917
DELTA = 0.40
BASE_PT_RANGE = (20, 100)
THETA_RANGE = (0.90, 1.10)
EPSILON_RANGE = (0.85, 1.15)
LOW_WIP_RULE = "pilot_ceil_capacity_over_3"
FORMAL_CANDIDATES_PER_CELL = 100
N_COARSE = 20
N_FINE = 200
DEFAULT_MAX_FINE_PER_CELL = 20
EPS = 1e-12


SIZE_CONFIGS = (
    ("S1", 30, 4, (2, 3)),
    ("S2", 30, 6, (2, 3)),
    ("S3", 60, 4, (2, 3, 4)),
    ("S4", 60, 6, (2, 3, 4)),
    ("S5", 90, 6, (3, 4)),
    ("S6", 90, 8, (3, 4)),
    ("S7", 120, 6, (3, 4, 5)),
    ("S8", 120, 8, (3, 4, 5)),
)

PROFILE_CONFIGS = (
    ("P1", "balanced"),
    ("P2", "upstream_bottleneck"),
    ("P3", "middle_bottleneck"),
    ("P4", "downstream_bottleneck"),
)

BUFFER_CONFIGS = (
    ("B1", "tight", 0.65),
    ("B2", "moderate", 1.00),
    ("B3", "loose", 1.50),
)

# Six representative cells: all tightness levels plus small/medium/large and all profiles.
SMOKE_CELL_IDS = (
    "S1_P1_B1",
    "S1_P1_B2",
    "S1_P1_B3",
    "S3_P2_B1",
    "S6_P3_B3",
    "S8_P4_B3",
)


@dataclass(frozen=True)
class StructuralCell:
    cell_id: str
    size_id: str
    n_jobs: int
    n_stages: int
    allowed_machine_counts: Tuple[int, ...]
    profile_id: str
    stage_profile: str
    tightness_id: str
    buffer_tightness: str
    beta: float
    cell_seed: int


@dataclass(frozen=True)
class ScreeningThresholds:
    min_p_block: float = 0.0
    min_p_short: float = 0.0
    min_d_c: float = 0.0
    min_d_s: float = 0.0
    min_d_b: float = 0.0


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def clip(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def build_structural_cells(master_seed: int) -> List[StructuralCell]:
    cells: List[StructuralCell] = []
    for size_id, n_jobs, n_stages, allowed_counts in SIZE_CONFIGS:
        for profile_id, stage_profile in PROFILE_CONFIGS:
            for tightness_id, tightness, beta in BUFFER_CONFIGS:
                cell_id = f"{size_id}_{profile_id}_{tightness_id}"
                cells.append(
                    StructuralCell(
                        cell_id=cell_id,
                        size_id=size_id,
                        n_jobs=n_jobs,
                        n_stages=n_stages,
                        allowed_machine_counts=allowed_counts,
                        profile_id=profile_id,
                        stage_profile=stage_profile,
                        tightness_id=tightness_id,
                        buffer_tightness=tightness,
                        beta=beta,
                        cell_seed=stable_seed(GENERATOR_VERSION, master_seed, cell_id),
                    )
                )
    assert len(cells) == 96
    assert len({cell.cell_id for cell in cells}) == 96
    return cells


def generate_machine_counts(cell: StructuralCell, rng: random.Random) -> List[int]:
    allowed = tuple(sorted(cell.allowed_machine_counts))
    counts = [rng.choice(allowed)]
    for _ in range(1, cell.n_stages):
        feasible = [m for m in allowed if abs(m - counts[-1]) <= 1]
        counts.append(rng.choice(feasible))

    if len(set(counts)) == 1 and len(allowed) > 1:
        positions = list(range(cell.n_stages))
        rng.shuffle(positions)
        for pos in positions:
            alternatives = []
            for value in allowed:
                if value == counts[pos]:
                    continue
                left_ok = pos == 0 or abs(value - counts[pos - 1]) <= 1
                right_ok = pos == cell.n_stages - 1 or abs(value - counts[pos + 1]) <= 1
                if left_ok and right_ok:
                    alternatives.append(value)
            if alternatives:
                counts[pos] = rng.choice(alternatives)
                break

    validate_machine_counts(counts, cell)
    return counts


def validate_machine_counts(counts: Sequence[int], cell: StructuralCell) -> None:
    if len(counts) != cell.n_stages:
        raise ValueError(f"{cell.cell_id}: machine-count length mismatch")
    if any(value not in cell.allowed_machine_counts for value in counts):
        raise ValueError(f"{cell.cell_id}: machine count outside allowed range: {counts}")
    if any(abs(a - b) > 1 for a, b in zip(counts, counts[1:])):
        raise ValueError(f"{cell.cell_id}: adjacent machine counts differ by more than one: {counts}")
    if max(counts) - min(counts) > 2:
        raise ValueError(f"{cell.cell_id}: machine-count range exceeds two: {counts}")


def build_target_load_profile(n_stages: int, profile: str) -> List[float]:
    if n_stages < 2:
        raise ValueError("n_stages must be at least two")

    raw: List[float] = []
    center = (n_stages + 1) / 2.0
    sigma = max(1.0, n_stages / 5.0)
    for stage_index in range(1, n_stages + 1):
        ratio = (stage_index - 1) / (n_stages - 1)
        if profile == "balanced":
            value = 1.0
        elif profile == "upstream_bottleneck":
            value = 1.0 + DELTA * (1.0 - ratio)
        elif profile == "downstream_bottleneck":
            value = 1.0 + DELTA * ratio
        elif profile == "middle_bottleneck":
            exponent = -((stage_index - center) ** 2) / (2.0 * sigma ** 2)
            value = 1.0 + DELTA * math.exp(exponent)
        else:
            raise ValueError(f"unknown stage profile: {profile}")
        raw.append(value)

    raw_mean = statistics.fmean(raw)
    return [value / raw_mean for value in raw]


def normalize_mean(values: Sequence[float]) -> List[float]:
    mean_value = statistics.fmean(values)
    if mean_value <= 0:
        raise ValueError("cannot normalize non-positive mean")
    return [value / mean_value for value in values]


def compute_stage_statistics(
    operations: Dict[str, List[Dict[str, Any]]],
    machine_counts: Sequence[int],
) -> Tuple[List[float], List[float], List[float]]:
    n_stages = len(machine_counts)
    mu: List[float] = []
    cv: List[float] = []
    tau: List[float] = []
    for stage_idx in range(n_stages):
        values = [
            float(pt)
            for job_ops in operations.values()
            for pt in job_ops[stage_idx]["machines"].values()
        ]
        stage_mean = statistics.fmean(values)
        stage_cv = statistics.pstdev(values) / stage_mean if stage_mean > 0 else 0.0
        mu.append(stage_mean)
        cv.append(stage_cv)
        tau.append(stage_mean / machine_counts[stage_idx])
    return mu, cv, tau


def analyze_fastest_machine_dominance(
    operations: Dict[str, List[Dict[str, Any]]],
    machine_counts: Sequence[int],
) -> Tuple[List[Dict[str, Any]], bool]:
    stage_results: List[Dict[str, Any]] = []
    any_warning = False
    for stage_idx, _ in enumerate(machine_counts):
        machine_ids = list(next(iter(operations.values()))[stage_idx]["machines"].keys())
        fastest_counts = {machine_id: 0 for machine_id in machine_ids}
        for job_ops in operations.values():
            times = job_ops[stage_idx]["machines"]
            fastest = min(times, key=lambda machine_id: (times[machine_id], machine_id))
            fastest_counts[fastest] += 1
        n_jobs = len(operations)
        shares = {
            machine_id: count / n_jobs for machine_id, count in fastest_counts.items()
        }
        max_machine = max(shares, key=shares.get)
        warning = shares[max_machine] > 0.90
        any_warning = any_warning or warning
        stage_results.append(
            {
                "stage": stage_idx + 1,
                "fastest_job_counts": fastest_counts,
                "fastest_job_shares": shares,
                "dominant_machine": max_machine,
                "dominant_share": shares[max_machine],
                "warning_over_90_percent": warning,
            }
        )
    return stage_results, any_warning


def generate_candidate(cell: StructuralCell, candidate_index: int, master_seed: int) -> Dict[str, Any]:
    candidate_id = f"WIPHFSP_{cell.cell_id}_C{candidate_index:03d}"
    candidate_seed = stable_seed(cell.cell_seed, candidate_index, "candidate")
    rng = random.Random(candidate_seed)

    machine_counts = generate_machine_counts(cell, rng)
    target_load = build_target_load_profile(cell.n_stages, cell.stage_profile)
    mean_machines = statistics.fmean(machine_counts)
    stage_scales = [
        target_load[idx] * machine_counts[idx] / mean_machines
        for idx in range(cell.n_stages)
    ]

    machine_ids_by_stage: List[List[str]] = []
    theta_by_stage: List[Dict[str, float]] = []
    for stage_idx, count in enumerate(machine_counts, start=1):
        machine_ids = [f"S{stage_idx}_M{machine_idx}" for machine_idx in range(1, count + 1)]
        raw_theta = [rng.uniform(*THETA_RANGE) for _ in machine_ids]
        theta = normalize_mean(raw_theta)
        machine_ids_by_stage.append(machine_ids)
        theta_by_stage.append(dict(zip(machine_ids, theta)))

    operations: Dict[str, List[Dict[str, Any]]] = {}
    for job_idx in range(1, cell.n_jobs + 1):
        job = f"J{job_idx}"
        job_ops: List[Dict[str, Any]] = []
        for stage_idx in range(cell.n_stages):
            base_pt = rng.randint(*BASE_PT_RANGE)
            mean_pt = base_pt * stage_scales[stage_idx]
            machine_times: Dict[str, int] = {}
            for machine_id in machine_ids_by_stage[stage_idx]:
                epsilon = rng.uniform(*EPSILON_RANGE)
                processing_time = max(
                    1,
                    round_half_up(mean_pt * theta_by_stage[stage_idx][machine_id] * epsilon),
                )
                machine_times[machine_id] = processing_time

            buffer_in = None if stage_idx == 0 else f"B{stage_idx - 1}{stage_idx}"
            buffer_out = None if stage_idx == cell.n_stages - 1 else f"B{stage_idx}{stage_idx + 1}"
            job_ops.append(
                {
                    "machines": machine_times,
                    "buffer_in": buffer_in,
                    "buffer_out": buffer_out,
                }
            )
        operations[job] = job_ops

    mu, stage_cv, tau = compute_stage_statistics(operations, machine_counts)
    capacities: List[int] = []
    buffers: Dict[str, Dict[str, int]] = {}
    base_capacities: List[int] = []
    for stage_idx in range(cell.n_stages - 1):
        ratio = tau[stage_idx + 1] / max(EPS, tau[stage_idx])
        pair_cv = (stage_cv[stage_idx] + stage_cv[stage_idx + 1]) / 2.0
        base_capacity = round_half_up(4.0 + 3.0 * abs(ratio - 1.0) + 2.0 * pair_cv)
        capacity = clip(round_half_up(cell.beta * base_capacity), 2, 10)
        low_wip = max(1, math.ceil(capacity / 3))
        bid = f"B{stage_idx}{stage_idx + 1}"
        base_capacities.append(base_capacity)
        capacities.append(capacity)
        buffers[bid] = {"capacity": capacity, "low_wip": low_wip}

    dominance_by_stage, dominance_warning = analyze_fastest_machine_dominance(
        operations, machine_counts
    )

    metadata = {
        "candidate_id": candidate_id,
        "generator_version": GENERATOR_VERSION,
        "master_seed": master_seed,
        "cell_seed": cell.cell_seed,
        "candidate_seed": candidate_seed,
        "cell_id": cell.cell_id,
        "size_id": cell.size_id,
        "profile_id": cell.profile_id,
        "tightness_id": cell.tightness_id,
        "n_jobs": cell.n_jobs,
        "n_stages": cell.n_stages,
        "machines_per_stage": machine_counts,
        "stage_profile": cell.stage_profile,
        "delta": DELTA,
        "target_effective_load_profile": target_load,
        "stage_scale": stage_scales,
        "buffer_tightness": cell.buffer_tightness,
        "beta": cell.beta,
        "machine_theta_range": list(THETA_RANGE),
        "machine_epsilon_range": list(EPSILON_RANGE),
        "processing_time_base_range": list(BASE_PT_RANGE),
        "machine_theta_by_stage": theta_by_stage,
        "buffer_base_capacities": base_capacities,
        "buffer_capacities": capacities,
        "low_wip_values": [buffers[bid]["low_wip"] for bid in buffers],
        "low_wip_rule": LOW_WIP_RULE,
        "mu_i": mu,
        "tau_i": tau,
        "cv_i": stage_cv,
        "fastest_machine_dominance": dominance_by_stage,
        "fastest_machine_dominance_warning": dominance_warning,
    }

    return {
        "spec": {
            "num_stages": cell.n_stages,
            "machines_per_stage": machine_counts,
            "n_jobs": cell.n_jobs,
            "buffer_caps": capacities,
            "pt_profile": cell.stage_profile,
            "pt_low": BASE_PT_RANGE[0],
            "pt_high": BASE_PT_RANGE[1],
            "seed": candidate_seed,
            "os_repeat": cell.n_stages,
        },
        "buffers": buffers,
        "operations": operations,
        "meta": {
            "format": "WIP-HFSP-BENCHMARK-JSON",
            "version": 1,
            "benchmark": metadata,
        },
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=False)


def load_candidate(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("spec", "buffers", "operations", "meta"):
        if key not in data:
            raise ValueError(f"{path}: missing top-level key {key}")
    return data


def manifest_row(candidate: Dict[str, Any], diagnostics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = candidate["meta"]["benchmark"]
    row: Dict[str, Any] = {
        key: meta[key]
        for key in (
            "candidate_id",
            "generator_version",
            "master_seed",
            "cell_seed",
            "candidate_seed",
            "cell_id",
            "size_id",
            "n_jobs",
            "n_stages",
            "machines_per_stage",
            "stage_profile",
            "delta",
            "buffer_tightness",
            "beta",
            "machine_theta_range",
            "machine_epsilon_range",
            "processing_time_base_range",
            "buffer_capacities",
            "low_wip_values",
            "low_wip_rule",
            "tau_i",
            "mu_i",
            "cv_i",
            "fastest_machine_dominance_warning",
        )
    }
    if diagnostics is not None:
        metrics = diagnostics.get("metrics", {})
        row.update(
            {
                "screening_phase": diagnostics.get("phase"),
                "valid": diagnostics.get("valid"),
                "mean_makespan": metrics.get("makespan", {}).get("mean"),
                "median_makespan": metrics.get("makespan", {}).get("median"),
                "D_C": metrics.get("D_C"),
                "mean_shortage": metrics.get("shortage", {}).get("mean"),
                "median_shortage": metrics.get("shortage", {}).get("median"),
                "D_S": metrics.get("D_S"),
                "mean_blocking": metrics.get("blocking", {}).get("mean"),
                "median_blocking": metrics.get("blocking", {}).get("median"),
                "D_B": metrics.get("D_B"),
                "P_block": metrics.get("P_block"),
                "P_short": metrics.get("P_short"),
                "spearman_correlation": metrics.get("spearman_correlation"),
                "pairwise_conflict_ratio": metrics.get("pairwise_conflict_ratio"),
            }
        )
    return row


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def save_manifest(output_root: Path, name: str, rows: Sequence[Dict[str, Any]]) -> None:
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest_dir / f"{name}.json", list(rows))
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (manifest_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "q05": math.nan,
            "q95": math.nan,
            "non_zero_ratio": 0.0,
        }
    numeric = [float(value) for value in values]
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
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average_rank
        start = end
    return ranks


def pearson_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    denominator_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    denominator = denominator_a * denominator_b
    return numerator / denominator if denominator > 0 else 0.0


def spearman_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson_correlation(average_ranks(a), average_ranks(b))


def pairwise_conflict_ratio(makespans: Sequence[float], shortages: Sequence[float]) -> float:
    conflicts = 0
    comparable = 0
    for left in range(len(makespans)):
        for right in range(left + 1, len(makespans)):
            delta_c = makespans[left] - makespans[right]
            delta_s = shortages[left] - shortages[right]
            if delta_c == 0 or delta_s == 0:
                continue
            comparable += 1
            if delta_c * delta_s < 0:
                conflicts += 1
    return conflicts / comparable if comparable else 0.0


def random_solution(
    operations: Dict[str, List[Dict[str, Any]]],
    rng: random.Random,
) -> Tuple[List[str], Dict[Tuple[str, int], str]]:
    encoder = Encoder(operations, rng=rng)
    os_seq = encoder.generate_random_os()
    ms_list = encoder.generate_random_ms()
    return os_seq, encoder.build_ms_map(ms_list)


def optional_finite_buffer_relevance() -> Dict[str, Any]:
    return {
        "status": "not_run",
        "reason": (
            "Optional diagnostic interface reserved. No infinite-buffer relaxation is "
            "executed in this version, so the formal decoder remains unchanged."
        ),
    }


def optional_empirical_hardness() -> Dict[str, Any]:
    return {
        "status": "not_run",
        "reason": (
            "Reserved for a later small-budget versus larger-budget NSGA-II comparison "
            "on the top candidates only."
        ),
    }


def screen_candidate(
    candidate: Dict[str, Any],
    n_samples: int,
    phase: str,
) -> Dict[str, Any]:
    meta = candidate["meta"]["benchmark"]
    operations = candidate["operations"]
    buffers = candidate["buffers"]
    diagnostic_seed = stable_seed(meta["candidate_seed"], phase, n_samples)
    rng = random.Random(diagnostic_seed)
    scheduler = StageBufferWIPScheduler(operations, buffers)

    makespans: List[float] = []
    shortages: List[float] = []
    blockings: List[float] = []
    full_ratios: List[float] = []
    empty_ratios: List[float] = []
    below_low_ratios: List[float] = []
    errors: List[str] = []

    for sample_index in range(1, n_samples + 1):
        try:
            os_seq, ms_map = random_solution(operations, rng)
            makespan, schedule, buffer_trace = scheduler.decode(os_seq, ms_map=ms_map)
            stats = scheduler.analyze(schedule, buffer_trace, makespan=makespan)
            if len(schedule) != sum(len(ops) for ops in operations.values()):
                raise RuntimeError("decoded schedule does not contain every operation exactly once")

            makespans.append(float(makespan))
            shortages.append(float(stats["shortage"]["total_shortage_area"]))
            blockings.append(float(stats["blocking"]["total_blocking_time"]))
            full_values = list(stats["buffers"]["per_buffer_full_ratio"].values())
            empty_values = list(stats["buffers"]["per_buffer_empty_ratio"].values())
            below_values = list(stats["shortage"]["per_buffer_below_low_ratio"].values())
            full_ratios.append(statistics.fmean(full_values) if full_values else 0.0)
            empty_ratios.append(statistics.fmean(empty_values) if empty_values else 0.0)
            below_low_ratios.append(statistics.fmean(below_values) if below_values else 0.0)
        except Exception as exc:
            message = f"sample {sample_index}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print(f"[screen-error] {meta['candidate_id']} {message}", file=sys.stderr)

    makespan_summary = summarize_distribution(makespans)
    shortage_summary = summarize_distribution(shortages)
    blocking_summary = summarize_distribution(blockings)
    valid = len(makespans) == n_samples and not errors

    d_c = (
        (makespan_summary["q95"] - makespan_summary["q05"])
        / max(EPS, makespan_summary["median"])
        if makespans
        else 0.0
    )
    d_s = (
        (shortage_summary["q95"] - shortage_summary["q05"])
        / (shortage_summary["median"] + EPS)
        if shortages
        else 0.0
    )
    d_b = (
        (blocking_summary["q95"] - blocking_summary["q05"])
        / (blocking_summary["q95"] + EPS)
        if blockings
        else 0.0
    )

    metrics = {
        "makespan": makespan_summary,
        "shortage": shortage_summary,
        "blocking": blocking_summary,
        "buffer_full_ratio": summarize_distribution(full_ratios),
        "buffer_empty_ratio": summarize_distribution(empty_ratios),
        "below_low_ratio": summarize_distribution(below_low_ratios),
        "D_C": d_c,
        "D_S": d_s,
        "D_B": d_b,
        "P_block": blocking_summary["non_zero_ratio"],
        "P_short": shortage_summary["non_zero_ratio"],
        "spearman_correlation": (
            spearman_correlation(makespans, shortages) if len(makespans) >= 2 else 0.0
        ),
        "pairwise_conflict_ratio": pairwise_conflict_ratio(makespans, shortages),
    }
    return {
        "candidate_id": meta["candidate_id"],
        "cell_id": meta["cell_id"],
        "phase": phase,
        "n_requested": n_samples,
        "n_successful": len(makespans),
        "diagnostic_seed": diagnostic_seed,
        "valid": valid,
        "errors": errors,
        "metrics": metrics,
        "finite_buffer_relevance": optional_finite_buffer_relevance(),
        "empirical_hardness": optional_empirical_hardness(),
    }


def passes_coarse_screen(
    diagnostics: Dict[str, Any],
    thresholds: ScreeningThresholds,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not diagnostics["valid"]:
        reasons.append("invalid_or_incomplete_decode")
        return False, reasons

    metrics = diagnostics["metrics"]
    if metrics["P_block"] < thresholds.min_p_block:
        reasons.append("P_block_below_configured_threshold")
    if metrics["P_short"] < thresholds.min_p_short:
        reasons.append("P_short_below_configured_threshold")
    if metrics["D_C"] < thresholds.min_d_c:
        reasons.append("D_C_below_configured_threshold")
    if metrics["D_S"] < thresholds.min_d_s:
        reasons.append("D_S_below_configured_threshold")
    if metrics["D_B"] < thresholds.min_d_b:
        reasons.append("D_B_below_configured_threshold")

    completely_inactive = metrics["P_block"] == 0.0 and metrics["P_short"] == 0.0
    completely_insensitive = metrics["D_C"] == 0.0 and metrics["D_S"] == 0.0 and metrics["D_B"] == 0.0
    if completely_inactive:
        reasons.append("wip_mechanisms_completely_inactive")
    if completely_insensitive:
        reasons.append("all_primary_diagnostics_constant")
    return not reasons, reasons


def hierarchical_rank_key(diagnostics: Dict[str, Any]) -> Tuple[Any, ...]:
    metrics = diagnostics["metrics"]
    wip_active_count = int(metrics["P_block"] > 0) + int(metrics["P_short"] > 0)
    objective_sensitivity_count = int(metrics["D_C"] > 0) + int(metrics["D_S"] > 0)
    min_objective_sensitivity = min(metrics["D_C"], metrics["D_S"])
    state_diversity = (
        metrics["buffer_full_ratio"]["std"]
        + metrics["buffer_empty_ratio"]["std"]
        + metrics["below_low_ratio"]["std"]
    )
    return (
        int(diagnostics["valid"]),
        wip_active_count,
        objective_sensitivity_count,
        min_objective_sensitivity,
        metrics["D_B"],
        state_diversity,
    )


def candidate_path(output_root: Path, cell_id: str, candidate_id: str) -> Path:
    return output_root / "candidates" / cell_id / f"{candidate_id}.json"


def generate_candidates(
    cells: Sequence[StructuralCell],
    candidates_per_cell: int,
    master_seed: int,
    output_root: Path,
) -> List[Path]:
    if candidates_per_cell <= 0:
        raise ValueError("candidates_per_cell must be positive")
    total = len(cells) * candidates_per_cell
    generated_paths: List[Path] = []
    manifest_rows: List[Dict[str, Any]] = []
    start_time = time.monotonic()
    progress = 0

    for cell_position, cell in enumerate(cells, start=1):
        print(f"[generate] cell {cell_position}/{len(cells)}: {cell.cell_id}")
        for candidate_index in range(1, candidates_per_cell + 1):
            candidate = generate_candidate(cell, candidate_index, master_seed)
            candidate_id = candidate["meta"]["benchmark"]["candidate_id"]
            path = candidate_path(output_root, cell.cell_id, candidate_id)
            save_json(path, candidate)
            generated_paths.append(path)
            manifest_rows.append(manifest_row(candidate))
            progress += 1
            elapsed = time.monotonic() - start_time
            print(
                f"  candidate {candidate_index}/{candidates_per_cell} | "
                f"total {progress}/{total} | elapsed {elapsed:.1f}s"
            )

    save_manifest(output_root, "generation_manifest", manifest_rows)
    save_json(
        output_root / "manifests" / "run_metadata.json",
        {
            "generator_version": GENERATOR_VERSION,
            "master_seed": master_seed,
            "cell_count": len(cells),
            "candidates_per_cell": candidates_per_cell,
            "candidate_count": len(generated_paths),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return generated_paths


def discover_candidates(output_root: Path) -> List[Path]:
    return sorted((output_root / "candidates").glob("*/*.json"))


def screen_candidates(
    candidate_paths: Sequence[Path],
    output_root: Path,
    coarse_samples: int,
    fine_samples: int,
    top_n: int,
    thresholds: ScreeningThresholds,
    max_fine_per_cell: int = 0,
) -> Dict[str, Any]:
    if not candidate_paths:
        raise ValueError(f"no candidate JSON files found under {output_root / 'candidates'}")

    candidates_by_cell: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = {}
    for path in candidate_paths:
        candidate = load_candidate(path)
        cell_id = candidate["meta"]["benchmark"]["cell_id"]
        candidates_by_cell.setdefault(cell_id, []).append((path, candidate))

    all_coarse: List[Dict[str, Any]] = []
    all_fine: List[Dict[str, Any]] = []
    selections: List[Dict[str, Any]] = []
    total_candidates = len(candidate_paths)
    coarse_progress = 0

    for cell_position, (cell_id, entries) in enumerate(sorted(candidates_by_cell.items()), start=1):
        print(f"[coarse] cell {cell_position}/{len(candidates_by_cell)}: {cell_id}")
        survivors: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []
        for path, candidate in entries:
            diagnostics = screen_candidate(candidate, coarse_samples, "coarse")
            passed, reasons = passes_coarse_screen(diagnostics, thresholds)
            diagnostics["coarse_pass"] = passed
            diagnostics["coarse_rejection_reasons"] = reasons
            save_json(output_root / "diagnostics" / "coarse" / f"{diagnostics['candidate_id']}.json", diagnostics)
            all_coarse.append(manifest_row(candidate, diagnostics))
            if passed:
                survivors.append((path, candidate, diagnostics))
            coarse_progress += 1
            print(
                f"  {diagnostics['candidate_id']} | {coarse_progress}/{total_candidates} | "
                f"pass={passed} P_block={diagnostics['metrics']['P_block']:.2f} "
                f"P_short={diagnostics['metrics']['P_short']:.2f}"
            )

        survivors.sort(key=lambda item: hierarchical_rank_key(item[2]), reverse=True)
        if max_fine_per_cell > 0:
            survivors = survivors[:max_fine_per_cell]

        print(f"[fine] {cell_id}: {len(survivors)} coarse survivors")
        fine_results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for _, candidate, _ in survivors:
            diagnostics = screen_candidate(candidate, fine_samples, "fine")
            save_json(output_root / "diagnostics" / "fine" / f"{diagnostics['candidate_id']}.json", diagnostics)
            all_fine.append(manifest_row(candidate, diagnostics))
            fine_results.append((candidate, diagnostics))

        fine_results.sort(key=lambda item: hierarchical_rank_key(item[1]), reverse=True)
        top_results = fine_results[:top_n]
        selections.append(
            {
                "cell_id": cell_id,
                "coarse_candidate_count": len(entries),
                "coarse_survivor_count": len(survivors),
                "fine_candidate_count": len(fine_results),
                "top_candidate_ids": [
                    diagnostics["candidate_id"] for _, diagnostics in top_results
                ],
                "provisional_formal_candidate_id": (
                    top_results[0][1]["candidate_id"] if top_results else None
                ),
                "selection_method": "hierarchical_validity_wip_sensitivity_state_diversity",
                "empirical_hardness_used": False,
            }
        )

    save_manifest(output_root, "coarse_screening_manifest", all_coarse)
    save_manifest(output_root, "fine_screening_manifest", all_fine)
    save_json(output_root / "manifests" / "selection_manifest.json", selections)
    return {
        "coarse_count": len(all_coarse),
        "fine_count": len(all_fine),
        "selections": selections,
    }


def validate_candidate_structure(candidate: Dict[str, Any]) -> Dict[str, Any]:
    meta = candidate["meta"]["benchmark"]
    operations = candidate["operations"]
    buffers = candidate["buffers"]
    machine_counts = meta["machines_per_stage"]
    allowed_lookup = {size_id: allowed for size_id, _, _, allowed in SIZE_CONFIGS}
    allowed = allowed_lookup[meta["size_id"]]

    if len(operations) != meta["n_jobs"]:
        raise ValueError(f"{meta['candidate_id']}: wrong job count")
    if any(len(ops) != meta["n_stages"] for ops in operations.values()):
        raise ValueError(f"{meta['candidate_id']}: wrong operation count")
    if any(value not in allowed for value in machine_counts):
        raise ValueError(f"{meta['candidate_id']}: disallowed machine count")
    if any(abs(a - b) > 1 for a, b in zip(machine_counts, machine_counts[1:])):
        raise ValueError(f"{meta['candidate_id']}: adjacent machine-count constraint violated")
    if max(machine_counts) - min(machine_counts) > 2:
        raise ValueError(f"{meta['candidate_id']}: global machine-count constraint violated")

    processing_times = [
        value
        for ops in operations.values()
        for op in ops
        for value in op["machines"].values()
    ]
    if not all(isinstance(value, int) and value > 0 for value in processing_times):
        raise ValueError(f"{meta['candidate_id']}: processing times must be positive integers")
    if not all(2 <= info["capacity"] <= 10 for info in buffers.values()):
        raise ValueError(f"{meta['candidate_id']}: buffer capacity outside [2, 10]")
    if not all(1 <= info["low_wip"] <= info["capacity"] for info in buffers.values()):
        raise ValueError(f"{meta['candidate_id']}: invalid pilot low_wip")

    actual_tau = [float(value) for value in meta["tau_i"]]
    target_q = [float(value) for value in meta["target_effective_load_profile"]]
    normalized_tau = normalize_mean(actual_tau)
    profile_rmse = math.sqrt(
        statistics.fmean((actual - target) ** 2 for actual, target in zip(normalized_tau, target_q))
    )
    return {
        "processing_time_min": min(processing_times),
        "processing_time_max": max(processing_times),
        "normalized_effective_load": normalized_tau,
        "target_effective_load": target_q,
        "profile_rmse": profile_rmse,
        "machine_counts": machine_counts,
        "buffer_capacities": [info["capacity"] for info in buffers.values()],
        "low_wip_values": [info["low_wip"] for info in buffers.values()],
        "fastest_machine_dominance_warning": meta["fastest_machine_dominance_warning"],
    }


def run_smoke(master_seed: int, output_root: Path) -> Dict[str, Any]:
    all_cells = {cell.cell_id: cell for cell in build_structural_cells(master_seed)}
    cells = [all_cells[cell_id] for cell_id in SMOKE_CELL_IDS]
    candidate_paths = generate_candidates(
        cells=cells,
        candidates_per_cell=2,
        master_seed=master_seed,
        output_root=output_root,
    )

    structure_checks: Dict[str, Dict[str, Any]] = {}
    reproducibility_checks: Dict[str, bool] = {}
    for path in candidate_paths:
        candidate = load_candidate(path)
        candidate_id = candidate["meta"]["benchmark"]["candidate_id"]
        structure_checks[candidate_id] = validate_candidate_structure(candidate)
        meta = candidate["meta"]["benchmark"]
        regenerated = generate_candidate(
            all_cells[meta["cell_id"]],
            int(candidate_id.rsplit("C", 1)[1]),
            master_seed,
        )
        reproducibility_checks[candidate_id] = regenerated == candidate
        if not reproducibility_checks[candidate_id]:
            raise AssertionError(f"{candidate_id}: seed reproduction failed")

    screening = screen_candidates(
        candidate_paths=candidate_paths,
        output_root=output_root,
        coarse_samples=6,
        fine_samples=12,
        top_n=2,
        thresholds=ScreeningThresholds(),
        max_fine_per_cell=2,
    )

    coarse_manifest_path = output_root / "manifests" / "coarse_screening_manifest.json"
    with coarse_manifest_path.open("r", encoding="utf-8") as handle:
        coarse_rows = json.load(handle)

    summary = {
        "generator_version": GENERATOR_VERSION,
        "master_seed": master_seed,
        "smoke_cells": list(SMOKE_CELL_IDS),
        "candidate_count": len(candidate_paths),
        "coarse_samples_per_candidate": 6,
        "fine_samples_per_candidate": 12,
        "structure_checks": structure_checks,
        "all_reproducible": all(reproducibility_checks.values()),
        "decode_error_candidate_count": sum(not row.get("valid", False) for row in coarse_rows),
        "always_zero_blocking_candidate_count": sum(row.get("P_block") == 0 for row in coarse_rows),
        "always_zero_shortage_candidate_count": sum(row.get("P_short") == 0 for row in coarse_rows),
        "constant_makespan_candidate_count": sum(row.get("D_C") == 0 for row in coarse_rows),
        "constant_shortage_candidate_count": sum(row.get("D_S") == 0 for row in coarse_rows),
        "constant_blocking_candidate_count": sum(row.get("D_B") == 0 for row in coarse_rows),
        "screening": screening,
    }
    save_json(output_root / "diagnostics" / "smoke_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and screen reproducible finite-buffer WIP-HFSP benchmark candidates."
    )
    parser.add_argument("--mode", choices=("smoke", "generate", "screen", "full"), required=True)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--coarse-samples", type=int, default=N_COARSE)
    parser.add_argument("--fine-samples", type=int, default=N_FINE)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--max-fine-candidates-per-cell",
        type=int,
        default=DEFAULT_MAX_FINE_PER_CELL,
        help="0 screens every coarse survivor; positive values provide an explicit pilot cap.",
    )
    parser.add_argument("--min-p-block", type=float, default=0.0)
    parser.add_argument("--min-p-short", type=float, default=0.0)
    parser.add_argument("--min-d-c", type=float, default=0.0)
    parser.add_argument("--min-d-s", type=float, default=0.0)
    parser.add_argument("--min-d-b", type=float, default=0.0)
    return parser.parse_args()


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.resolve()
    directory = "smoke_test" if args.mode == "smoke" else "benchmark_candidates"
    return (PROJECT_ROOT / "data" / directory).resolve()


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args)
    cells = build_structural_cells(args.master_seed)
    formal_candidate_count = len(cells) * FORMAL_CANDIDATES_PER_CELL
    assert formal_candidate_count == 9600

    thresholds = ScreeningThresholds(
        min_p_block=args.min_p_block,
        min_p_short=args.min_p_short,
        min_d_c=args.min_d_c,
        min_d_s=args.min_d_s,
        min_d_b=args.min_d_b,
    )

    print(f"generator_version={GENERATOR_VERSION}")
    print(f"master_seed={args.master_seed}")
    print(f"structural_cells={len(cells)} formal_candidates={formal_candidate_count}")
    print(f"output_root={output_root}")

    if args.mode == "smoke":
        summary = run_smoke(args.master_seed, output_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.mode in {"generate", "full"}:
        paths = generate_candidates(
            cells=cells,
            candidates_per_cell=FORMAL_CANDIDATES_PER_CELL,
            master_seed=args.master_seed,
            output_root=output_root,
        )
        if len(paths) != 9600:
            raise AssertionError(f"expected 9600 candidates, generated {len(paths)}")

    if args.mode in {"screen", "full"}:
        paths = discover_candidates(output_root)
        result = screen_candidates(
            candidate_paths=paths,
            output_root=output_root,
            coarse_samples=args.coarse_samples,
            fine_samples=args.fine_samples,
            top_n=args.top_n,
            thresholds=thresholds,
            max_fine_per_cell=args.max_fine_candidates_per_cell,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
