import os
import sys
import time
import statistics

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import load_instance_from_json
from src.algorithms.base_population_search import BasePopulationSearch


def run_once(json_path, seed, use_local_search):
    spec, operations, buffers, _ = load_instance_from_json(json_path)

    search = BasePopulationSearch(
        operations=operations,
        buffers=buffers,
        pop_size=10,
        n_generations=10,
        crossover_rate=0.8,
        os_mutation_rate=0.2,
        ms_mutation_rate=0.1,
        tournament_size=2,
        seed=seed,
        use_local_search=use_local_search,
        elite_ls_count=2,
        ls_max_tries=4,
        ls_blocking_threshold=3,
        ls_require_positive_blocking=True,
    )

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
    ls_runs = []

    for seed in seeds:
        baseline_runs.append(run_once(json_path, seed, use_local_search=False))
        ls_runs.append(run_once(json_path, seed, use_local_search=True))

    base_summary = summarize_results(baseline_runs)
    ls_summary = summarize_results(ls_runs)

    print("\n[Baseline]")
    print(base_summary)

    print("\n[LS-enhanced]")
    print(ls_summary)

    print("\n[Delta: LS - Baseline]")
    print({
        "best_makespan_delta": ls_summary["best_makespan"] - base_summary["best_makespan"],
        "avg_makespan_delta": ls_summary["avg_makespan"] - base_summary["avg_makespan"],
        "avg_runtime_delta": ls_summary["avg_runtime"] - base_summary["avg_runtime"],
    })


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]

    instance_files = [
        r"data/instances/WIP-FMS/WIP-FMS_small_S3_K2_N20_balanced_seed55.json",
        r"data/instances/WIP-FMS/WIP-FMS_small_S3_K2_N20_mid_bottleneck_seed44.json",
        r"data/instances/WIP-FMS/WIP-FMS_small_S3_K2_N20_downstream_bottleneck_seed22.json",
    ]

    for path in instance_files:
        compare_on_instance(path, seeds)