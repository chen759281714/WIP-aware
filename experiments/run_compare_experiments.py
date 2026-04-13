import os
import sys
import time
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import load_instance_from_json
from src.algorithms.baseline_nsga2 import BaselineNSGA2
from src.algorithms.baseline_moead import BaselineMOEAD
from src.algorithms.emt_glocal_ga_v2 import EMTGLocalGAV2

# =========================
# 实验参数
# =========================

ALGORITHMS = {
    "EMTGLocalGAV2": EMTGLocalGAV2,
    "BaselineNSGA2": BaselineNSGA2,
    "BaselineMOEAD": BaselineMOEAD,
}
SEEDS = list(range(1, 2))

INSTANCE_DIR = "data/instances/WIP-FMS"
RUN_RESULT_DIR = "experiments/results/runs"


POP_SIZE = 300
MAX_EVALUATIONS = 100000
SNAPSHOT_INTERVAL = 2000

# EMT 三个种群的规模分配（总和应等于 POP_SIZE）
EMT_MAIN_POP_SIZE = POP_SIZE//3
EMT_GLOBAL_POP_SIZE = POP_SIZE//3
EMT_LOCAL_POP_SIZE = POP_SIZE - EMT_MAIN_POP_SIZE - EMT_GLOBAL_POP_SIZE

def validate_experiment_config():
    if EMT_MAIN_POP_SIZE + EMT_GLOBAL_POP_SIZE + EMT_LOCAL_POP_SIZE != POP_SIZE:
        raise ValueError(
            "EMT_MAIN_POP_SIZE + EMT_GLOBAL_POP_SIZE + EMT_LOCAL_POP_SIZE "
            "必须等于 POP_SIZE"
        )
    
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
def to_jsonable(obj):
    """
    将常见自定义对象递归转换为可 JSON 序列化的基础类型
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    if is_dataclass(obj):
        return to_jsonable(asdict(obj))

    if hasattr(obj, "__dict__"):
        return to_jsonable(vars(obj))

    return str(obj)


def run_once(instance_path, seed, algo_name):
    spec, operations, buffers, _ = load_instance_from_json(instance_path)

    # 保证 low_wip 字段存在，避免后续分析脚本还要回头补
    for bid in buffers:
        if "low_wip" not in buffers[bid]:
            buffers[bid]["low_wip"] = 1

    AlgoClass = ALGORITHMS[algo_name]

    algo_params = {}
    stop_condition = {
        "type": "max_evaluations",
        "value": MAX_EVALUATIONS
    }

    if algo_name == "BaselineNSGA2":
        algo_params = {
            "pop_size": POP_SIZE,
            "max_evaluations": MAX_EVALUATIONS,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "seed": seed,
        }
        search = AlgoClass(
            operations=operations,
            buffers=buffers,
            pop_size=POP_SIZE,
            max_evaluations=MAX_EVALUATIONS,
            snapshot_interval=SNAPSHOT_INTERVAL,
            seed=seed
        )

    elif algo_name == "BaselineMOEAD":
        algo_params = {
            "pop_size": POP_SIZE,
            "max_evaluations": MAX_EVALUATIONS,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "seed": seed,
            "neighborhood_size": max(10, POP_SIZE // 10),
            "neighbor_mating_prob": 0.9,
            "max_replace": 2,
        }
        search = AlgoClass(
            operations=operations,
            buffers=buffers,
            pop_size=POP_SIZE,
            max_evaluations=MAX_EVALUATIONS,
            snapshot_interval=SNAPSHOT_INTERVAL,
            seed=seed,
            neighborhood_size=max(5, POP_SIZE // 10),
            neighbor_mating_prob=0.9,
            max_replace=2,
        )

    elif algo_name == "EMTGLocalGAV2":
        algo_params = {
            "main_pop_size": EMT_MAIN_POP_SIZE,
            "global_pop_size": EMT_GLOBAL_POP_SIZE,
            "local_pop_size": EMT_LOCAL_POP_SIZE,
            "max_evaluations": MAX_EVALUATIONS,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "seed": seed,
            "crossover_rate": 0.7,
            "os_mutation_rate": 0.1,
            "ms_mutation_rate": 0.1,
            "tournament_size": 2,
            "gat_improve_window": 5,
            "gat_improve_threshold": 0.005,
            "local_elite_count": 12,
            "local_neighbors_per_elite": 6,
            "local_os_mutation_rate": 0.2,
            "local_ms_mutation_rate": 0.2,
        }
        search = AlgoClass(
            operations=operations,
            buffers=buffers,
            pop_size=EMT_MAIN_POP_SIZE,
            global_pop_size=EMT_GLOBAL_POP_SIZE,
            local_pop_size=EMT_LOCAL_POP_SIZE,
            max_evaluations=MAX_EVALUATIONS,
            snapshot_interval=SNAPSHOT_INTERVAL,
            seed=seed,
            crossover_rate=0.7,
            os_mutation_rate=0.1,
            ms_mutation_rate=0.1,
            tournament_size=2,
            gat_improve_window=5,
            gat_improve_threshold=0.005,
            local_elite_count=12,
            local_neighbors_per_elite=6,
            local_os_mutation_rate=0.2,
            local_ms_mutation_rate=0.2,
        )

    else:
        raise ValueError(f"未知算法: {algo_name}")

    t0 = time.time()
    best = search.run(
        store_stats_init=True,
        store_stats_generations=False,
        verbose=False
    )
    runtime = time.time() - t0

    if best.stats is None:
        raise RuntimeError("best.stats is None：算法返回的代表解未携带完整统计信息")

    pareto_front = search.get_pareto_front()

    rep_blocking = best.stats["blocking"]["total_blocking_time"]
    rep_shortage = best.shortage if best.shortage is not None else best.stats["shortage"]["total_shortage_area"]

    rep_crowding = getattr(best, "crowding_distance", None)
    if rep_crowding == float("inf"):
        rep_crowding = "inf"

    pareto_solutions = []
    for ind in pareto_front:
        if ind.stats is None:
            raise RuntimeError("Pareto front 中存在 stats is None 的个体，请检查算法返回逻辑")

        crowding_distance = getattr(ind, "crowding_distance", None)
        if crowding_distance == float("inf"):
            crowding_distance = "inf"

        pareto_solutions.append({
            "makespan": ind.makespan,
            "shortage": ind.shortage,
            "blocking": ind.stats["blocking"]["total_blocking_time"],
            "rank": getattr(ind, "rank", None),
            "crowding_distance": crowding_distance,
            "OS": ind.OS,
            "MS": ind.MS,
        })

    # ---------- Pareto 诊断 ----------
    pareto_points = [
        (float(sol["makespan"]), float(sol["shortage"]))
        for sol in pareto_solutions
    ]
    unique_pareto_points = sorted(set(pareto_points))

    pareto_size_raw = len(pareto_solutions)
    pareto_size_unique = len(unique_pareto_points)
    pareto_duplicate_count = pareto_size_raw - pareto_size_unique

    from collections import Counter
    point_counter = Counter(pareto_points)
    top_duplicate_points = [
        {"point": [pt[0], pt[1]], "count": cnt}
        for pt, cnt in point_counter.most_common(10)
        if cnt > 1
    ]

    population_size = None
    rank0_population_size = None
    unique_rank0_points = None

    if hasattr(search, "main_population") and search.main_population:
        population_size = len(search.main_population)

        rank0_inds = [
            ind for ind in search.main_population
            if getattr(ind, "rank", None) == 0
        ]
        rank0_population_size = len(rank0_inds)

        unique_rank0_points = len(set(
            (float(ind.makespan), float(ind.shortage))
            for ind in rank0_inds
            if ind.makespan is not None and ind.shortage is not None
        ))

    elif hasattr(search, "population") and search.population:
        population_size = len(search.population)

        rank0_inds = [
            ind for ind in search.population
            if getattr(ind, "rank", None) == 0
        ]
        rank0_population_size = len(rank0_inds)

        unique_rank0_points = len(set(
            (float(ind.makespan), float(ind.shortage))
            for ind in rank0_inds
            if ind.makespan is not None and ind.shortage is not None
        ))

    result = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "instance_name": os.path.splitext(os.path.basename(instance_path))[0],
        "instance_path": os.path.normpath(instance_path),
        "instance_file": os.path.basename(instance_path),
        "instance_spec": to_jsonable(spec),
        "buffer_config": to_jsonable(buffers),

        "algorithm": algo_name,
        "seed": seed,
        "algorithm_parameters": algo_params,
        "stop_condition": stop_condition,

        "experiment_config": {
            "pop_size_global": POP_SIZE,
            "max_evaluations": MAX_EVALUATIONS,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "emt_main_pop_size": EMT_MAIN_POP_SIZE,
            "emt_global_pop_size": EMT_GLOBAL_POP_SIZE,
            "emt_local_pop_size": EMT_LOCAL_POP_SIZE,
        },
        
        "representative_result": {
            "makespan": best.makespan,
            "shortage": rep_shortage,
            "blocking": rep_blocking,
            "runtime": runtime,
            "n_evaluations": getattr(search, "n_evaluations", None),
            "rank": getattr(best, "rank", None),
            "crowding_distance": rep_crowding,
        },

        "representative_solution": {
            "OS": best.OS,
            "MS": best.MS,
            "schedule": best.schedule,
            "buffer_trace": best.buffer_trace,
            "stats": best.stats,
        },

        "history": {
            "x_axis": "evaluation_count",
            "eval_counts": getattr(search, "history_eval_counts", []),
            "representative_makespan": getattr(search, "history_best_fitness", []),
            "representative_shortage": getattr(search, "history_best_shortage", []),
        },

        "front_history": {
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "x_axis": "evaluation_count",
            "y_axis_for_future_analysis": ["gd", "igd"],
            "snapshots": getattr(search, "history_fronts", []),
        },

        "pareto_summary": {
            "pareto_size": pareto_size_unique,
            "pareto_size_raw": pareto_size_raw,
            "pareto_duplicate_count": pareto_duplicate_count,
            "population_size": population_size,
            "rank0_population_size": rank0_population_size,
            "unique_rank0_points": unique_rank0_points,
            "top_duplicate_points": top_duplicate_points,
            "unique_pareto_points": [
                {"makespan": p[0], "shortage": p[1]}
                for p in unique_pareto_points
            ],
        },

        "pareto_front": pareto_solutions,
    }

    return result


# =========================
# 保存 run JSON
# =========================

def save_run(result):
    algo_name = result["algorithm"]
    instance_name = result["instance_name"]
    seed = result["seed"]

    algo_dir = os.path.join(RUN_RESULT_DIR, algo_name)
    os.makedirs(algo_dir, exist_ok=True)

    filename = f"{instance_name}_seed{seed}.json"
    path = os.path.join(algo_dir, filename)

    jsonable_result = to_jsonable(result)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable_result, f, indent=2, ensure_ascii=False)

    return path

# =========================
# 实验主循环
# =========================

def run_experiments():
    validate_experiment_config()
    os.makedirs(RUN_RESULT_DIR, exist_ok=True)

    instance_paths = get_instances()

    total_runs = len(instance_paths) * len(ALGORITHMS) * len(SEEDS)
    run_counter = 0

    manifest = []

    for path in instance_paths:
        instance_name = os.path.splitext(os.path.basename(path))[0]

        print("\n==============================")
        print("Instance:", instance_name)

        for algo_name in ALGORITHMS:
            print("Algorithm:", algo_name)

            for seed in SEEDS:
                result = run_once(path, seed, algo_name)
                save_path = save_run(result)

                run_counter += 1

                manifest.append({
                    "instance": instance_name,
                    "algorithm": algo_name,
                    "seed": seed,
                    "file": save_path,
                    "n_evaluations": result["representative_result"]["n_evaluations"],
                    "runtime": result["representative_result"]["runtime"],
                    "snapshot_interval": SNAPSHOT_INTERVAL,
                    "has_front_history": result["front_history"]["snapshots"] is not None,
                })

                ps = result["pareto_summary"]

                print(
                    f"[RUN {run_counter}/{total_runs}] "
                    f"instance={instance_name}  "
                    f"algo={algo_name}  "
                    f"seed={seed}  "
                    f"rep_makespan={result['representative_result']['makespan']}  "
                    f"rep_shortage={result['representative_result']['shortage']:.2f}  "
                    f"pareto_size={ps['pareto_size']}  "
                    f"pareto_size_raw={ps['pareto_size_raw']}  "
                    f"dup={ps['pareto_duplicate_count']}  "
                    f"rank0={ps['rank0_population_size']}  "
                    f"rank0_unique={ps['unique_rank0_points']}  "
                    f"evals={result['representative_result']['n_evaluations']}  "
                    f"time={result['representative_result']['runtime']:.2f}s"
                )

    manifest_path = os.path.join(RUN_RESULT_DIR, "_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nAll run JSON files saved.")
    print("Manifest saved to:")
    print(manifest_path)

# =========================
# main
# =========================

if __name__ == "__main__":
    run_experiments()