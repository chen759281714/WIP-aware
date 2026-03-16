import os
import sys
import time
import json
import statistics

from openpyxl import Workbook
from openpyxl.styles import Font

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import load_instance_from_json
from src.algorithms.baseline_ga import BaselineGA
from src.algorithms.elite_ls_ga import EliteLSGA


# =========================
# 实验参数
# =========================

ALGORITHMS = {
    "BaselineGA": BaselineGA,
    "EliteLSGA": EliteLSGA,
}

SEEDS = list(range(1, 11))

INSTANCE_DIR = "data/instances/WIP-FMS"

RUN_RESULT_DIR = "experiments/results/runs"
SUMMARY_DIR = "experiments/results/summary"

POP_SIZE = 100
GENERATIONS = 50


# =========================
# 获取实例
# =========================

def get_instances():

    files = []

    for f in os.listdir(INSTANCE_DIR):
        if f.endswith(".json"):
            files.append(os.path.join(INSTANCE_DIR, f))

    files.sort()

    return files


# =========================
# 单次运行
# =========================

def run_once(instance_path, seed, algo_name):

    spec, operations, buffers, _ = load_instance_from_json(instance_path)

    AlgoClass = ALGORITHMS[algo_name]

    if algo_name == "BaselineGA":

        search = AlgoClass(
            operations=operations,
            buffers=buffers,
            pop_size=POP_SIZE,
            n_generations=GENERATIONS,
            seed=seed
        )

    else:

        search = AlgoClass(
            operations=operations,
            buffers=buffers,
            pop_size=POP_SIZE,
            n_generations=GENERATIONS,
            seed=seed,
            elite_ls_count=2,
            ls_max_tries=6,
            ls_blocking_threshold=3
        )

    t0 = time.time()

    best = search.run(
        store_stats_init=True,
        store_stats_generations=False,
        verbose=False
    )
    if best.stats is None:
        search.evaluate_individual(best, store_stats=True)

    blocking = best.stats["blocking"]["total_blocking_time"]

    runtime = time.time() - t0

    return {
        "makespan": best.makespan,
        "blocking":blocking,
        "runtime": runtime,
        "history": search.history_best_fitness
    }


# =========================
# 保存 run JSON
# =========================

def save_run(instance_name, algo_name, seed, result):

    algo_dir = os.path.join(RUN_RESULT_DIR, algo_name)
    os.makedirs(algo_dir, exist_ok=True)

    filename = f"{instance_name}_seed{seed}.json"

    path = os.path.join(algo_dir, filename)

    data = {
        "algorithm": algo_name,
        "instance": instance_name,
        "seed": seed,
        "parameters": {
            "pop_size": POP_SIZE,
            "generations": GENERATIONS
        },
        "result": {
            "makespan": result["makespan"],
            "runtime": result["runtime"],
            "blocking": result["blocking"]
        },
        "history": {
            "best_fitness_per_generation": result["history"]
        }
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# =========================
# 实验主循环
# =========================

def run_experiments():

    os.makedirs(RUN_RESULT_DIR, exist_ok=True)

    instance_paths = get_instances()

    total_runs = len(instance_paths) * len(ALGORITHMS) * len(SEEDS)
    run_counter = 0

    summary = {}

    for path in instance_paths:

        instance_name = os.path.splitext(os.path.basename(path))[0]

        print("\n==============================")
        print("Instance:", instance_name)

        summary[instance_name] = {}

        for algo_name in ALGORITHMS:

            print("Algorithm:", algo_name)

            makespans = []

            for seed in SEEDS:

                result = run_once(path, seed, algo_name)

                makespans.append(result["makespan"])

                save_run(instance_name, algo_name, seed, result)
                run_counter += 1

                print(
                        f"[RUN {run_counter}/{total_runs}] "
                        f"[RUN] instance={instance_name}  "
                        f"algo={algo_name}  "
                        f"seed={seed}  "
                        f"makespan={result['makespan']}  "
                        f"blocking={result['blocking']}  "
                        f"time={result['runtime']:.2f}s"
                    )

            best = min(makespans)
            avg = statistics.mean(makespans)

            summary[instance_name][algo_name] = {
                "best": best,
                "avg": avg
            }

            print(
                    f"[SUMMARY] instance={instance_name}  "
                    f"algo={algo_name}  "
                    f"best={best}  "
                    f"avg={avg:.2f}"
                )

    return summary


# =========================
# 保存 Excel
# =========================

def save_excel(summary):

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    path = os.path.join(SUMMARY_DIR, "compare_results.xlsx")

    wb = Workbook()
    ws = wb.active

    algorithms = list(ALGORITHMS.keys())

    header = ["Instance"]

    for algo in algorithms:
        header.append(f"{algo}_best")
        header.append(f"{algo}_avg")

    ws.append(header)

    bold_font = Font(bold=True)

    for instance in summary:

        row = [instance]

        best_values = []

        for algo in algorithms:
            best = summary[instance][algo]["best"]
            avg = summary[instance][algo]["avg"]

            best_values.append(best)

            row.append(best)
            row.append(avg)

        ws.append(row)

        # 找最优 best makespan
        min_value = min(best_values)

        row_id = ws.max_row

        col = 2

        for best in best_values:

            if best == min_value:
                ws.cell(row=row_id, column=col).font = bold_font

            col += 2

    wb.save(path)

    print("\nExcel summary saved to:")
    print(path)


# =========================
# main
# =========================

if __name__ == "__main__":

    summary = run_experiments()

    save_excel(summary)