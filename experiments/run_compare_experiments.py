import os
import sys
import time
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import load_instance_from_json
from src.algorithms.baseline_nsga2 import BaselineNSGA2
from src.algorithms.baseline_moead import BaselineMOEAD
from src.algorithms.emt_glocal_ga_v2 import EMTGLocalGAV2
from src.algorithms.emt_glocal_ga_v2_no_gat import EMTGLocalGAV2_NoGAT
from src.algorithms.emt_glocal_ga_v2_no_lat import EMTGLocalGAV2_NoLAT
from src.algorithms.baseline_spea2 import BaselineSPEA2

# =========================
# 实验参数
# =========================

ALGORITHMS = {
    "EMTGLocalGAV2": EMTGLocalGAV2,
    "NoGAT": EMTGLocalGAV2_NoGAT,
    "NoLAT": EMTGLocalGAV2_NoLAT,
    #"BaselineNSGA2": BaselineNSGA2,
    #"BaselineMOEAD": BaselineMOEAD,
    #"BaselineSPEA2": BaselineSPEA2,
}
SEEDS = list(range(1, 11))

INSTANCE_DIR = "data/instances/WIP-FMS"
RUN_RESULT_DIR = "experiments/results/runs"


POP_SIZE = 300
MAX_EVALUATIONS = 40000
SNAPSHOT_INTERVAL = 1000

# 只跑部分算例；None 表示全部
# 例如 (11, 20) 表示只跑排序后第 12~20 个算例（Python 切片，右边不含）
INSTANCE_INDEX_RANGE = (2,15)

# 并行进程数；None 表示自动取 cpu_count()
N_PROCESSES = 5

# 是否跳过已经存在的 run json
SKIP_EXISTING = True

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

    if INSTANCE_INDEX_RANGE is not None:
        start, end = INSTANCE_INDEX_RANGE
        files = files[start:end]

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

    elif algo_name == "BaselineSPEA2":
        algo_params = {
            "pop_size": POP_SIZE,
            "archive_size": POP_SIZE,
            "max_evaluations": MAX_EVALUATIONS,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "seed": seed,
            "crossover_rate": 0.8,
            "os_mutation_rate": 0.2,
            "ms_mutation_rate": 0.2,
            "tournament_size": 2,
        }
        search = AlgoClass(
            operations=operations,
            buffers=buffers,
            **algo_params
        )

    elif algo_name in ["EMTGLocalGAV2", "NoGAT", "NoLAT"]:

        if algo_name == "EMTGLocalGAV2":
            main = EMT_MAIN_POP_SIZE
            global_ = EMT_GLOBAL_POP_SIZE
            local = EMT_LOCAL_POP_SIZE

        elif algo_name == "NoGAT":
            # 把 global 的资源分给 main + local
            main = POP_SIZE // 2
            local = POP_SIZE - main
            global_ = 0

        elif algo_name == "NoLAT":
            # 把 local 的资源分给 main + global
            main = POP_SIZE // 2
            global_ = POP_SIZE - main
            local = 0

        algo_params = {
            "main_pop_size": main,
            "global_pop_size": global_,
            "local_pop_size": local,
            "max_evaluations": MAX_EVALUATIONS,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "seed": seed,
            "crossover_rate": 0.7,
            "os_mutation_rate": 0.1,
            "ms_mutation_rate": 0.1,
            "tournament_size": 2,
            "local_elite_count": 12,
            "local_neighbors_per_elite": 6,
            "local_os_mutation_rate": 0.2,
            "local_ms_mutation_rate": 0.2,
        }

        search = AlgoClass(
            operations=operations,
            buffers=buffers,
            pop_size=main,
            global_pop_size=global_,
            local_pop_size=local,
            max_evaluations=MAX_EVALUATIONS,
            snapshot_interval=SNAPSHOT_INTERVAL,
            seed=seed,
            crossover_rate=0.7,
            os_mutation_rate=0.1,
            ms_mutation_rate=0.1,
            tournament_size=2,
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

    path = get_run_json_path(instance_name, algo_name, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    jsonable_result = to_jsonable(result)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable_result, f, indent=2, ensure_ascii=False)

    return path

def get_run_json_path(instance_name, algo_name, seed):
    algo_dir = os.path.join(RUN_RESULT_DIR, algo_name)
    filename = f"{instance_name}_seed{seed}.json"
    return os.path.join(algo_dir, filename)


def already_done(instance_name, algo_name, seed):
    path = get_run_json_path(instance_name, algo_name, seed)
    return os.path.exists(path)

def worker(task):
    path, instance_name, algo_name, seed = task

    try:
        result = run_once(path, seed, algo_name)
        save_path = save_run(result)

        ps = result["pareto_summary"]

        return {
            "ok": True,
            "instance": instance_name,
            "algorithm": algo_name,
            "seed": seed,
            "file": save_path,
            "n_evaluations": result["representative_result"]["n_evaluations"],
            "runtime": result["representative_result"]["runtime"],
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "has_front_history": result["front_history"]["snapshots"] is not None,

            "rep_makespan": result["representative_result"]["makespan"],
            "rep_shortage": result["representative_result"]["shortage"],
            "pareto_size": ps["pareto_size"],
            "pareto_size_raw": ps["pareto_size_raw"],
            "pareto_duplicate_count": ps["pareto_duplicate_count"],
            "rank0_population_size": ps["rank0_population_size"],
            "unique_rank0_points": ps["unique_rank0_points"],
        }

    except Exception as e:
        return {
            "ok": False,
            "instance": instance_name,
            "algorithm": algo_name,
            "seed": seed,
            "error": repr(e),
        }

# =========================
# 实验主循环
# =========================

def run_experiments():
    validate_experiment_config()
    os.makedirs(RUN_RESULT_DIR, exist_ok=True)

    instance_paths = get_instances()

    print("Instances selected:")
    for i, p in enumerate(instance_paths):
        print(f"  [{i}] {os.path.basename(p)}")

    tasks = []
    skipped = []

    for path in instance_paths:
        instance_name = os.path.splitext(os.path.basename(path))[0]

        for algo_name in ALGORITHMS:
            for seed in SEEDS:
                if SKIP_EXISTING and already_done(instance_name, algo_name, seed):
                    skipped.append((instance_name, algo_name, seed))
                    continue

                tasks.append((path, instance_name, algo_name, seed))

    print()
    print(f"Total tasks to run : {len(tasks)}")
    print(f"Total tasks skipped: {len(skipped)}")

    if skipped:
        print("Skipped existing runs:")
        for instance_name, algo_name, seed in skipped[:20]:
            print(f"  SKIP {instance_name} | {algo_name} | seed={seed}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    manifest = []

    if not tasks:
        print("\nNo pending tasks. Nothing to run.")
        manifest_path = os.path.join(RUN_RESULT_DIR, "_run_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        print("Manifest saved to:")
        print(manifest_path)
        return

    n_proc = N_PROCESSES if N_PROCESSES is not None else cpu_count()
    n_proc = max(1, n_proc)

    print(f"\nUsing {n_proc} processes...\n")

    done_count = 0

    with Pool(processes=n_proc) as pool:
        for res in pool.imap_unordered(worker, tasks):
            done_count += 1

            manifest.append(res)

            if res["ok"]:
                print(
                    f"[RUN {done_count}/{len(tasks)}] "
                    f"instance={res['instance']}  "
                    f"algo={res['algorithm']}  "
                    f"seed={res['seed']}  "
                    f"rep_makespan={res['rep_makespan']}  "
                    f"rep_shortage={res['rep_shortage']:.2f}  "
                    f"pareto_size={res['pareto_size']}  "
                    f"pareto_size_raw={res['pareto_size_raw']}  "
                    f"dup={res['pareto_duplicate_count']}  "
                    f"rank0={res['rank0_population_size']}  "
                    f"rank0_unique={res['unique_rank0_points']}  "
                    f"evals={res['n_evaluations']}  "
                    f"time={res['runtime']:.2f}s"
                )
            else:
                print(
                    f"[RUN {done_count}/{len(tasks)}] "
                    f"instance={res['instance']}  "
                    f"algo={res['algorithm']}  "
                    f"seed={res['seed']}  "
                    f"FAILED: {res['error']}"
                )

    manifest_path = os.path.join(RUN_RESULT_DIR, "_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nAll pending run JSON files processed.")
    print("Manifest saved to:")
    print(manifest_path)

# =========================
# main
# =========================

if __name__ == "__main__":
    run_experiments()
