"""Compare frozen SIS units and best gaps against a supplied correct baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms import wip_graph_dual_population as optimized
from performance_trajectory_regression import canonical, load_module


def compare(base, current, guide, high, phi, operations, buffers):
    def units_and_gaps(module):
        search = module.WIPGraphDualPopulation(operations, buffers, N=2, FE_max=4)
        candidate = module.Individual(current[:], {key: next(iter(operations[key[0]][key[1]]["machines"]))
                                                  for key in current},
                                      provenance=module.DecodeProvenance())
        target = module.Individual(guide[:], candidate.ms_map.copy())
        diagnosis = module.ShortageDiagnosis([], phi, set(), [])
        units = search._build_relink_units(candidate, target, high, diagnosis)
        gaps = [(key, search._best_os_relink_action(candidate, target, key)) for key in sorted(high)]
        return canonical(units), canonical(gaps)

    left, right = units_and_gaps(base), units_and_gaps(optimized)
    if left != right:
        raise AssertionError(f"SIS units or best gaps differ: {left!r} != {right!r}")
    return len(left[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    base = load_module("perf_sis_baseline", args.baseline)
    if not hasattr(base.WIPGraphDualPopulation, "_build_relink_units"):
        parser.error("Baseline must implement frozen global SIS; stage-wise snapshots are incompatible")
    directory = ROOT / "data/final_benchmark/instances"
    for size in (1, 4, 8):
        data = json.loads((directory / f"WIPHFSP_S{size}_P1_B1.json").read_text())
        operations, buffers = data["operations"], data["buffers"]
        search = base.WIPGraphDualPopulation(operations, buffers, N=2, FE_max=4,
                                             seed=17)
        rng = random.Random(4)
        compared = 0
        for _ in range(24):
            current = search.encoder.generate_random_os()
            guide = search.encoder.generate_random_os()
            high = set(rng.sample(current, min(12, len(current))))
            phi = {key: rng.random() for key in high}
            compare(base, current, guide, high, phi, operations, buffers)
            compared += 1
        same = search.encoder.generate_random_os()
        if compare(base, same, same, set(), {}, operations, buffers) != 0:
            raise AssertionError("No-action SIS case unexpectedly produced an action")
        print(f"S{size}: {compared} ordered unit/gap comparisons and no-unit case equal")


if __name__ == "__main__":
    main()
