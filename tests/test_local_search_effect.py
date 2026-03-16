import os
import sys
import time
import statistics

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import load_instance_from_json
from src.algorithms.baseline_ga import BaselineGA
from src.algorithms.elite_ls_ga import EliteLSGA


def run_once(json_path, seed, algorithm_name):
    spec, operations, buffers, _ = load_instance_from_json(json_path)

    common_params = dict(
        operations=operations,
        buffers=buffers,
        pop_size=10,
        n_generations=5,
        crossover_rate=0.8,
        os_mutation_rate=0.2,
        ms_mutation_rate=0.1,
        tournament_size=2,
        seed=seed,
    )

    if algorithm_name == "baseline":
        search = BaselineGA(**common_params)

    elif algorithm_name == "elite_ls":
        search = EliteLSGA(
            **common_params,
            elite_ls_count=2,
            ls_max_tries=6,
            ls_blocking_threshold=3,
            ls_require_positive_blocking=True,
        )

    else:
        raise ValueError(f"未知算法名称: {algorithm_name}")

    t0 = time.time()
    best = search.run(
        store_stats_init=True,
        store_stats_generations=False,
        verbose=False
    )
    t1 = time.time()

    # 为确保 best.stats 存在，必要时补一次完整评价
    if best.stats is None:
        search.evaluate_individual(best, store_stats=True)

    blocking = best.stats["blocking"]["total_blocking_time"] if best.stats is not None else None

    return {
        "makespan": best.makespan,
        "history": search.history_best_fitness[:],
        "runtime": t1 - t0,
        "blocking": blocking,
    }


def summarize_results(results):
    makespans = [r["makespan"] for r in results]
    runtimes = [r["runtime"] for r in results]
    blockings = [r["blocking"] for r in results if r["blocking"] is not None]

    return {
        "best_makespan": min(makespans),
        "avg_makespan": statistics.mean(makespans),
        "std_makespan": statistics.pstdev(makespans) if len(makespans) > 1 else 0.0,
        "avg_runtime": statistics.mean(runtimes),
        "avg_blocking": statistics.mean(blockings) if blockings else None,
    }


def compare_on_instance(json_path, seeds):
    print("\n" + "=" * 60)
    print("Instance:", os.path.basename(json_path))

    baseline_runs = []
    elite_ls_runs = []

    for seed in seeds:
        baseline_runs.append(run_once(json_path, seed, algorithm_name="baseline"))
        elite_ls_runs.append(run_once(json_path, seed, algorithm_name="elite_ls"))

    base_summary = summarize_results(baseline_runs)
    elite_summary = summarize_results(elite_ls_runs)

    print("\n[Baseline]")
    print(base_summary)

    print("\n[Elite-LS]")
    print(elite_summary)

    print("\n[Delta: Elite-LS - Baseline]")
    print({
        "best_makespan_delta": elite_summary["best_makespan"] - base_summary["best_makespan"],
        "avg_makespan_delta": elite_summary["avg_makespan"] - base_summary["avg_makespan"],
        "avg_runtime_delta": elite_summary["avg_runtime"] - base_summary["avg_runtime"],
        "avg_blocking_delta": (
            None if base_summary["avg_blocking"] is None or elite_summary["avg_blocking"] is None
            else elite_summary["avg_blocking"] - base_summary["avg_blocking"]
        ),
    })


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    instance_files = [
        r"data/instances/WIP-FMS/WIP-FMS_07.json",
        r"data/instances/WIP-FMS/WIP-FMS_10.json",
        r"data/instances/WIP-FMS/WIP-FMS_13.json",
        r"data/instances/WIP-FMS/WIP-FMS_14.json",
        r"data/instances/WIP-FMS/WIP-FMS_17.json",
        r"data/instances/WIP-FMS/WIP-FMS_20.json",
    ]

    for path in instance_files:
        compare_on_instance(path, seeds)