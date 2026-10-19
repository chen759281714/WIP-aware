"""Compare optimized global incomparable-pair distance with a supplied baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.algorithms import wip_graph_dual_population as optimized
from performance_trajectory_regression import load_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    base = load_module("perf_distance_baseline", args.baseline)
    directory = ROOT / "data/final_benchmark/instances"
    for size in (1, 4, 8):
        data = json.loads((directory / f"WIPHFSP_S{size}_P1_B1.json").read_text())
        operations, buffers = data["operations"], data["buffers"]
        old = base.WIPGraphDualPopulation(operations, buffers, N=2, FE_max=4, seed=9)
        new = optimized.WIPGraphDualPopulation(operations, buffers, N=2, FE_max=4, seed=9)
        for index in range(60):
            x_os = old.encoder.generate_random_os()
            y_os = old.encoder.generate_random_os()
            x_ms = old.encoder.build_ms_map(old.encoder.generate_random_ms())
            y_ms = old.encoder.build_ms_map(old.encoder.generate_random_ms())
            x = base.Individual(x_os, x_ms)
            y = base.Individual(y_os, y_ms)
            left, right = old.structural_distance(x, y), new.structural_distance(x, y)
            if left != right:
                raise AssertionError(f"S{size} pair {index}: {left} != {right}")
        print(f"S{size}: 60 distances exactly equal")

    operations = {job: [{"machines": {"M": 1}} for _ in range(2)]
                  for job in ("A", "B")}
    search = optimized.WIPGraphDualPopulation(operations, {}, N=1, FE_max=2)
    ms = {(job, op): "M" for job in operations for op in range(2)}
    x = optimized.Individual([("A", 0), ("A", 1), ("B", 0), ("B", 1)], ms)
    y = optimized.Individual([("A", 0), ("B", 0), ("A", 1), ("B", 1)], ms)
    if search.structural_distance(x, y) != 0.125:
        raise AssertionError("Cross-stage incomparable-pair counterexample changed")
    print("Cross-stage counterexample: 0.125")


if __name__ == "__main__":
    main()
