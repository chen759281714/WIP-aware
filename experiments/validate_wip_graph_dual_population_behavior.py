"""Short-budget behavior diagnostics for WIPGraphDualPopulation, not a benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation


DEFAULT_INSTANCES = [
    ROOT / "data/final_benchmark/instances" / f"WIPHFSP_S1_P1_B{tightness}.json"
    for tightness in (1, 2, 3)
]
DEFAULT_OUTPUT = ROOT / "experiments/results/wip_graph_behavior/diagnostics.json"


def numeric_summary(values: List[float]) -> Dict[str, Optional[float]]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def percentile(values: List[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def population_snapshot(algo: WIPGraphDualPopulation) -> Dict[str, Any]:
    def nearest(population):
        if len(population) < 2:
            return {"mean": None, "median": None}
        distances = [min(algo.structural_distance(x, y)
                         for y in population if y is not x) for x in population]
        assert all(0.0 <= d <= 1.0 for d in distances)
        return {"mean": statistics.mean(distances),
                "median": statistics.median(distances)}

    left = {algo.genotype_key(x) for x in algo.P_M}
    right = {algo.genotype_key(x) for x in algo.P_S}
    overlap = len(left & right)
    return {"fe": algo.n_evaluations,
            "pm_nearest_distance": nearest(algo.P_M),
            "ps_nearest_distance": nearest(algo.P_S),
            "overlap_count": overlap,
            "overlap_ratio": overlap / algo.N,
            "archive_size": len(algo.A)}


def assert_search_invariants(algo: WIPGraphDualPopulation) -> None:
    assert algo.n_evaluations <= algo.FE_max
    assert len(algo.P_M) == algo.N and len(algo.P_S) == algo.N
    assert all(not algo.dominates(x, y) for x in algo.A for y in algo.A if x is not y)
    expected = set(algo.encoder.ms_index_order)
    for ind in algo.P_M + algo.P_S + algo.A:
        assert algo.encoder.validate_os(ind.os_seq)
        assert set(ind.ms_map) == expected
        for (job, op_idx), machine in ind.ms_map.items():
            assert machine in algo.operations[job][op_idx]["machines"]
        assert 0.0 <= algo.structural_distance(ind, ind) <= 1.0
    for record in algo.diagnostics["relinking"]:
        assert record["executed"] <= record["requested"]
        assert record["fe_before"] == record["fe_after"]
        assert record["os_moves"] + record["ms_moves"] == record["executed"]
    generated = sum(item["generated"] for group in ("pm_moves", "ps_moves")
                    for item in algo.diagnostics[group].values())
    assert generated == algo.n_evaluations - 2 * algo.N


def summarize_run(algo: WIPGraphDualPopulation, checkpoints: Dict[str, dict],
                  instance_name: str, seed: int) -> Dict[str, Any]:
    diag = algo.diagnostics
    pm = {name: {**counts, "acceptance_rate": rate(counts["accepted"], counts["generated"])}
          for name, counts in diag["pm_moves"].items()}
    ps = {name: {**counts, "acceptance_rate": rate(counts["accepted"], counts["generated"])}
          for name, counts in diag["ps_moves"].items()}
    spans = numeric_summary(algo.machine_priority_move_spans)
    spans["p90"] = percentile(algo.machine_priority_move_spans, 0.9)
    action_sets = diag["pm_action_sets"]
    action_kinds = ("machine_priority", "machine_reassign")
    action_composition = {
        name: numeric_summary([record[name] for record in action_sets])
        for name in (*action_kinds, "total_actions")
    }
    action_availability = {
        name: {"available_calls": sum(record[name] > 0 for record in action_sets),
               "specialized_calls": len(action_sets),
               "rate": rate(sum(record[name] > 0 for record in action_sets), len(action_sets))}
        for name in action_kinds
    }
    selected_depths = {name: numeric_summary(diag["pm_selected_depths"][name])
                       for name in action_kinds}
    selected_origins = {
        kind: {tag: {**counts, "acceptance_rate": rate(counts["accepted"], counts["generated"])}
               for tag, counts in tags.items()}
        for kind, tags in diag["pm_selected_origins"].items()
    }
    via_unblocking = diag["pm_via_unblocking_records"]
    diagnosis = dict(diag["shortage_diagnosis"])
    diagnosis["unanchored_shortage_intervals"] = algo.unanchored_shortage_intervals
    diagnosis["phi_mass_mismatch_count"] = algo.phi_mass_mismatch_count
    diagnosis["complete_phi_rate"] = rate(diagnosis["complete_phi_calls"], diagnosis["calls"])
    hs = {"size": numeric_summary(diag["hs"]["sizes"]),
          "ratio": numeric_summary(diag["hs"]["ratios"]),
          "coverage": numeric_summary(diag["hs"]["coverages"]),
          "coverage_violations": diag["hs"]["coverage_violations"]}
    relinks = diag["relinking"]
    relinking = {name: numeric_summary([record[name] for record in relinks])
                 for name in ("D0", "requested", "executed", "os_moves", "ms_moves")}
    cooperation = dict(diag["cooperation"])
    cooperation["valid_archive_guide_rate"] = rate(cooperation["ps_valid_guide"], cooperation["ps_probes"])
    cooperation["self_fallback_rate"] = rate(cooperation["ps_fallback_self"], cooperation["ps_probes"])
    cooperation["pm_archive_usable_rate"] = rate(cooperation["pm_usable"], cooperation["pm_attempts"])
    final = checkpoints["100%"]
    source_counts = Counter(algo._first_source_by_genotype.get(algo.genotype_key(x), "unknown")
                            for x in algo.A)
    result = {
        "instance": instance_name,
        "seed": seed,
        "fe": algo.n_evaluations,
        "pm_moves": pm,
        "pm_direct_action_kinds": list(action_kinds),
        "pm_action_composition": action_composition,
        "pm_action_availability": action_availability,
        "pm_propagation": diag["pm_propagation"],
        "pm_action_origins": diag["pm_action_origins"],
        "pm_selected_origins": selected_origins,
        "pm_via_unblocking_schedule_changed": {
            "selected": len(via_unblocking),
            "changed": sum(record["schedule_changed"] for record in via_unblocking),
            "unchanged": sum(not record["schedule_changed"] for record in via_unblocking),
        },
        "pm_via_unblocking_records": via_unblocking,
        "pm_selected_depths": selected_depths,
        "pm_action_set_samples": action_sets,
        "pm_selected_depth_samples": diag["pm_selected_depths"],
        "machine_priority_span": spans,
        "machine_priority_span_samples": algo.machine_priority_move_spans,
        "ps_moves": ps,
        "shortage_diagnosis": diagnosis,
        "hs": hs,
        "relinking": relinking,
        "archive_cooperation": cooperation,
        "diversity_checkpoints": checkpoints,
        "source_generated": diag["source_generated"],
        "final_archive_sources": dict(source_counts),
        "final_archive": {"size": len(algo.A),
                          "min_makespan": min(x.makespan for x in algo.A),
                          "min_shortage": min(x.shortage for x in algo.A)},
    }
    specialized = sum(pm[name]["generated"] for name in action_kinds)
    pm_total = sum(item["generated"] for item in pm.values())
    ps_total = sum(item["generated"] for item in ps.values())
    result["pm_specialized_ratios"] = {
        name: rate(pm[name]["generated"], specialized)
        for name in action_kinds}
    result["ps_relink_ratio"] = rate(ps["shortage_relink"]["generated"], ps_total)
    result["ps_fallback_mutation_ratio"] = rate(sum(ps[name]["generated"] for name in ps
                                                      if name.startswith("fallback_")), ps_total)
    warnings_list = []
    if diagnosis["unanchored_shortage_intervals"]:
        warnings_list.append("Unanchored shortage intervals occurred.")
    if diagnosis["phi_mass_mismatch_count"]:
        warnings_list.append("Shortage influence mass mismatches occurred.")
    if specialized and pm["machine_reassign"]["generated"] / specialized > 0.8:
        warnings_list.append("P_M specialized search is dominated by machine reassignment. "
                             "Inspect action availability before formal benchmark.")
    if specialized and max(pm[name]["generated"] for name in result["pm_specialized_ratios"]) / specialized > 0.9:
        warnings_list.append("One P_M specialized move type exceeds 90%.")
    if ps_total and ps["shortage_relink"]["generated"] / ps_total < 0.1:
        warnings_list.append("P_S relink rate is below 10%.")
    if cooperation["ps_probes"] and cooperation["ps_valid_guide"] / cooperation["ps_probes"] < 0.1:
        warnings_list.append("Valid archive guide rate is below 10%.")
    if final["overlap_ratio"] > 0.8:
        warnings_list.append("Final P_M/P_S genotype overlap exceeds 80%.")
    for side in ("pm_nearest_distance", "ps_nearest_distance"):
        mean = final[side]["mean"]
        if mean is not None and mean < 1e-3:
            warnings_list.append(f"Final {side} mean is near zero.")
    result["warnings"] = warnings_list
    return result


def run_case(instance_path: Path, seed: int, N: int, N_A: int, FE_max: int) -> Dict[str, Any]:
    with open(instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)
    algo = WIPGraphDualPopulation(instance["operations"], instance["buffers"],
                                  N=N, N_A=N_A, FE_max=FE_max, seed=seed)
    checkpoints: Dict[str, dict] = {}
    targets = {name: math.ceil(fraction * FE_max)
               for name, fraction in (("25%", 0.25), ("50%", 0.5),
                                      ("75%", 0.75), ("100%", 1.0))}

    def callback(current: WIPGraphDualPopulation) -> None:
        if "initial" not in checkpoints:
            checkpoints["initial"] = population_snapshot(current)
        for name, threshold in targets.items():
            if name not in checkpoints and current.n_evaluations >= threshold:
                checkpoints[name] = population_snapshot(current)

    algo.run(diagnostic_callback=callback)
    assert_search_invariants(algo)
    assert "100%" in checkpoints
    return summarize_run(algo, checkpoints, instance_path.stem, seed)


def print_summary(result: Dict[str, Any]) -> None:
    print("=" * 50)
    print(f"Instance: {result['instance']} | Seed: {result['seed']} | FE: {result['fe']}")
    print("P_M:")
    for name, data in result["pm_moves"].items():
        print(f"  {name}: generated={data['generated']} accepted={data['accepted']} "
              f"rate={data['acceptance_rate']}")
    print(f"  specialized ratios: {result['pm_specialized_ratios']}")
    print(f"  direct action kinds: {result['pm_direct_action_kinds']}")
    print(f"  actionable-set composition: {result['pm_action_composition']}")
    print(f"  action availability: {result['pm_action_availability']}")
    print(f"  propagation: {result['pm_propagation']}")
    print(f"  selected origins: {result['pm_selected_origins']}")
    print(f"  via-unblocking schedule: {result['pm_via_unblocking_schedule_changed']}")
    print(f"  selected depths: {result['pm_selected_depths']}")
    print(f"  machine-priority span: {result['machine_priority_span']}")
    print("P_S:")
    for name, data in result["ps_moves"].items():
        print(f"  {name}: generated={data['generated']} accepted={data['accepted']}")
    print(f"  relink ratio={result['ps_relink_ratio']} "
          f"fallback mutation ratio={result['ps_fallback_mutation_ratio']}")
    print(f"Shortage diagnosis: {result['shortage_diagnosis']}")
    print(f"H_S: mean size={result['hs']['size']['mean']} "
          f"mean ratio={result['hs']['ratio']['mean']} "
          f"size min/max={result['hs']['size']['min']}/{result['hs']['size']['max']} "
          f"coverage violations={result['hs']['coverage_violations']}")
    print(f"Relinking: mean D0={result['relinking']['D0']['mean']} "
          f"requested={result['relinking']['requested']['mean']} "
          f"executed={result['relinking']['executed']['mean']} "
          f"OS moves={result['relinking']['os_moves']['mean']} "
          f"MS moves={result['relinking']['ms_moves']['mean']}")
    print(f"Archive cooperation: {result['archive_cooperation']}")
    for name, snap in result["diversity_checkpoints"].items():
        print(f"  {name}: FE={snap['fe']} PM NN={snap['pm_nearest_distance']['mean']} "
              f"PS NN={snap['ps_nearest_distance']['mean']} "
              f"overlap={snap['overlap_ratio']} archive={snap['archive_size']}")
    print(f"Final archive: {result['final_archive']} "
          f"sources={result['final_archive_sources']}")
    for message in result["warnings"]:
        print(f"WARNING: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--N-A", type=int, default=40)
    parser.add_argument("--fe-max", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print("Diagnostic configuration only; not benchmark setting.")
    print(f"Instances={len(args.instances)} seeds={args.seeds} N={args.N} "
          f"N_A={args.N_A} FE_max={args.fe_max}")
    results = []
    for instance_path in args.instances:
        for seed in args.seeds:
            result = run_case(instance_path, seed, args.N, args.N_A, args.fe_max)
            results.append(result)
            print_summary(result)
    output = {"purpose": "algorithm behavior diagnostics; not formal performance comparison",
              "config": {"instances": [str(path) for path in args.instances],
                         "seeds": args.seeds, "N": args.N, "N_A": args.N_A,
                         "FE_max": args.fe_max},
              "runs": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(f"Saved diagnostics: {args.output}")


if __name__ == "__main__":
    main()
