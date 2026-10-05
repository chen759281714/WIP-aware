"""Compare an on-disk corrected baseline with the current implementation.

Run with PYTHONHASHSEED fixed and pass --baseline to a source snapshot.
This is intentionally separate from unittest discovery: the snapshot is not
part of the repository and must be supplied explicitly.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms import wip_graph_dual_population as current_module


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def variant_module(base_module, variant: str):
    if variant == "full":
        return base_module
    path = ROOT / f"src/algorithms/wip_graph_dual_population_{variant}.py"
    package_name = "src.algorithms.wip_graph_dual_population"
    original = sys.modules[package_name]
    try:
        sys.modules[package_name] = base_module
        module = load_module(f"perf_{variant}_{id(base_module)}", path)
    finally:
        sys.modules[package_name] = original
    class_name = {"wo_mgs": "WIPGraphDualPopulationWithoutMGS",
                  "wo_sis": "WIPGraphDualPopulationWithoutSIS",
                  "wo_ac": "WIPGraphDualPopulationWithoutAC"}[variant]
    return SimpleNamespace(WIPGraphDualPopulation=getattr(module, class_name),
                           Move=base_module.Move)


def canonical(value):
    if dataclasses.is_dataclass(value):
        return tuple((field.name, canonical(getattr(value, field.name)))
                     for field in dataclasses.fields(value))
    if isinstance(value, dict):
        return tuple((canonical(key), canonical(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(canonical(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((canonical(item) for item in value), key=repr))
    return value


def population_state(population):
    return tuple((tuple(ind.os_seq),
                  tuple((key, ind.ms_map[key]) for key in ind.operation_order),
                  ind.makespan, ind.shortage)
                 for ind in population)


def signature(value):
    # Hash all fields in deterministic order; avoid retaining every event and diagnostic copy.
    return hashlib.sha256(repr(canonical(value)).encode("utf-8")).hexdigest()


def finish_result(search, trace, elapsed):
    from experiments.validate_wip_graph_dual_population_behavior import assert_search_invariants, sis_violation_counts
    assert_search_invariants(search)
    relinks = search.diagnostics["relinking"]
    summary = {"applied_os_units": sum(r["applied_os_units"] for r in relinks),
               "applied_ms_units": sum(r["applied_ms_units"] for r in relinks),
               "total_os_discrepancy_reduction": sum(r["total_os_discrepancy_reduction"] for r in relinks),
               "violations": sis_violation_counts(search)}
    return trace, signature(search.diagnostics), population_state(search.A), search.rng.getstate(), elapsed, summary


def run_case(module, operations, buffers, seed, fe_max, trace_enabled=True):
    search = module.WIPGraphDualPopulation(
        operations, buffers, N=3, N_A=12, FE_max=fe_max, seed=seed,
        T_coop=2)
    if not trace_enabled:
        start = time.perf_counter()
        search.run()
        elapsed = time.perf_counter() - start
        return finish_result(search, (population_state(search.P_M), population_state(search.P_S)), elapsed)
    trace = []
    choices = []
    selections = []
    original_choice = search.rng.choice

    def traced_choice(options):
        selected = original_choice(options)
        if isinstance(selected, module.Move):
            choices.append(canonical(selected))
        return selected

    search.rng.choice = traced_choice

    for name in ("tournament", "archive_guide", "choose_guide",
                 "choose_archive_guide_for_shortage"):
        original = getattr(search, name)

        def record_selection(*args, _name=name, _original=original, **kwargs):
            selected = _original(*args, **kwargs)
            selections.append((_name, tuple(selected.os_seq) if selected else None,
                               tuple(selected.ms_map.items()) if selected else None))
            return selected

        setattr(search, name, record_selection)

    original_evaluate = search.evaluate

    def record_evaluate(ind):
        result = original_evaluate(ind)
        trace.append({
            "fe": search.n_evaluations,
            "child": signature(result),
            "child_genotype": (tuple(result.os_seq), tuple(result.ms_map.items()),
                               result.makespan, result.shortage),
            "rng_after_evaluation": search.rng.getstate(),
            "selected_actions": tuple(choices),
            "selected_parents_guides": tuple(selections),
            "pm_kind": search._last_pm_move_kind,
            "ps_kind": search._last_ps_move_kind,
            "ps_guide_source": search._last_ps_guide_source,
            "pm_action": canonical(search._last_pm_selected_action),
        })
        choices.clear()
        selections.clear()
        return result

    search.evaluate = record_evaluate
    original_replace = search.similarity_replace

    def record_replace(population, child, objective):
        accepted = original_replace(population, child, objective)
        trace[-1]["replacement"] = (objective, accepted)
        trace[-1]["pm"] = population_state(search.P_M)
        trace[-1]["ps"] = population_state(search.P_S)
        trace[-1]["archive_at_replacement"] = population_state(search.A)
        trace[-1]["diagnostics_at_replacement"] = signature(search.diagnostics)
        trace[-1]["rng_after_replacement"] = search.rng.getstate()
        return accepted

    search.similarity_replace = record_replace
    original_archive = search.update_archive

    def record_archive(candidates):
        result = original_archive(candidates)
        if trace:
            trace[-1]["archive"] = population_state(search.A)
        return result

    search.update_archive = record_archive
    start = time.perf_counter()
    search.run()
    elapsed = time.perf_counter() - start
    return finish_result(search, trace, elapsed)


def first_difference(left, right, label):
    if left == right:
        return
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        for index, (a, b) in enumerate(zip(left, right)):
            if a != b:
                first_difference(a, b, f"{label}[{index}]")
        if len(left) != len(right):
            raise AssertionError(f"{label}: lengths {len(left)} != {len(right)}")
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() | right.keys():
            if left.get(key) != right.get(key):
                first_difference(left.get(key), right.get(key), f"{label}.{key}")
    raise AssertionError(f"{label}: {repr(left)[:240]} != {repr(right)[:240]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-decoder", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 23])
    parser.add_argument("--fe", type=int, default=200)
    parser.add_argument("--variant", choices=("full", "wo_mgs", "wo_sis", "wo_ac"),
                        default="full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing-only", action="store_true",
                        help="Time uninstrumented runs; per-FE validation must be run separately")
    args = parser.parse_args()
    baseline = load_module("perf_baseline_algorithm", args.baseline)
    if not hasattr(baseline.WIPGraphDualPopulation, "_build_relink_units"):
        parser.error("Use a correct global-SIS baseline, not the historical stage-wise implementation")
    if args.baseline_decoder:
        decoder = load_module("perf_baseline_decoder", args.baseline_decoder)
        baseline.StageBufferWIPScheduler = decoder.StageBufferWIPScheduler
    baseline = variant_module(baseline, args.variant)
    optimized = variant_module(current_module, args.variant)
    instance_dir = ROOT / "data/final_benchmark/instances"
    results = []
    for size in args.sizes:
        data = json.loads((instance_dir / f"WIPHFSP_S{size}_P1_B1.json").read_text())
        for seed in args.seeds:
            old = run_case(baseline, data["operations"], data["buffers"], seed, args.fe,
                           trace_enabled=not args.timing_only)
            new = run_case(optimized, data["operations"], data["buffers"], seed, args.fe,
                           trace_enabled=not args.timing_only)
            for label, left, right in zip(("per_fe", "diagnostics", "archive", "rng"),
                                          old[:4], new[:4]):
                first_difference(left, right, f"S{size}/seed{seed}/{label}")
            print(f"{args.variant} S{size} seed={seed} FE={args.fe}: equal; "
                  f"baseline={old[4]:.3f}s optimized={new[4]:.3f}s "
                  f"speedup={old[4] / new[4]:.3f}x", flush=True)
            results.append({"variant": args.variant, "size": size, "seed": seed,
                            "FE": args.fe, "N": 3, "N_A": 12, "T_coop": 2,
                            "trajectory_equal": None if args.timing_only else True,
                            "final_state_equal": True, "instrumented": not args.timing_only,
                            "baseline_seconds": old[4],
                            "optimized_seconds": new[4], "speedup": old[4] / new[4],
                            "behavior": new[5]})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
