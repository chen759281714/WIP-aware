import os
import sys
import json
import time
import math
import statistics
from copy import deepcopy
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import load_instance_from_json
from src.algorithms.emt_glocal_ga_v2 import EMTGLocalGAV2


# =====================================
# 基本配置
# =====================================

INSTANCE_PATH = "data/instances/WIP-FMS/WIP-FMS_07.json"
RESULT_DIR = "experiments/results/taguchi_emt_v2"
RUN_DIR = os.path.join(RESULT_DIR, "runs")
SUMMARY_DIR = os.path.join(RESULT_DIR, "summary")

SEEDS = list(range(1, 11))
MAX_EVALUATIONS = 100000
SNAPSHOT_INTERVAL = None   # 田口实验这里只关心最终 IGD，可不记录中间快照

# 可选: "run" / "analyze" / "both"
MODE = "analyze"


# =====================================
# 因子与水平
# =====================================

FACTOR_LEVELS = {
    "PS": [150, 200, 250, 300],
    "Pc": [0.7, 0.8, 0.9, 1.0],
    "Pm": [0.1, 0.2, 0.3, 0.4],
    "LS": [4, 8, 12, 16],
}

FACTOR_NAMES = ["PS", "Pc", "Pm", "LS"]

# L16(4^4)
L16_TABLE = [
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [1, 4, 4, 4],
    [2, 1, 2, 3],
    [2, 2, 1, 4],
    [2, 3, 4, 1],
    [2, 4, 3, 2],
    [3, 1, 3, 4],
    [3, 2, 4, 3],
    [3, 3, 1, 2],
    [3, 4, 2, 1],
    [4, 1, 4, 2],
    [4, 2, 3, 1],
    [4, 3, 2, 4],
    [4, 4, 1, 3],
]


# =====================================
# 多目标工具函数
# =====================================

def dominates_point(a, b):
    return (
        a[0] <= b[0] and
        a[1] <= b[1] and
        (a[0] < b[0] or a[1] < b[1])
    )


def filter_nondominated(points):
    unique_points = list(set(points))
    nd = []

    for p in unique_points:
        dominated = False
        for q in unique_points:
            if p == q:
                continue
            if dominates_point(q, p):
                dominated = True
                break
        if not dominated:
            nd.append(p)

    nd.sort(key=lambda x: (x[0], x[1]))
    return nd


def build_reference_front(all_run_fronts):
    all_points = []
    for front in all_run_fronts:
        for sol in front:
            all_points.append((float(sol["makespan"]), float(sol["shortage"])))
    return filter_nondominated(all_points)


def compute_normalization_bounds(reference_front):
    if not reference_front:
        return {"m_min": 0.0, "m_max": 1.0, "s_min": 0.0, "s_max": 1.0}

    m_values = [p[0] for p in reference_front]
    s_values = [p[1] for p in reference_front]

    return {
        "m_min": min(m_values),
        "m_max": max(m_values),
        "s_min": min(s_values),
        "s_max": max(s_values),
    }


def normalize_point(point, bounds):
    m, s = point

    if bounds["m_max"] > bounds["m_min"]:
        nm = (m - bounds["m_min"]) / (bounds["m_max"] - bounds["m_min"])
    else:
        nm = 0.0

    if bounds["s_max"] > bounds["s_min"]:
        ns = (s - bounds["s_min"]) / (bounds["s_max"] - bounds["s_min"])
    else:
        ns = 0.0

    return (nm, ns)


def compute_igd(pareto_front, reference_front, bounds):
    if not pareto_front or not reference_front:
        return float("inf")

    points = [
        normalize_point((float(sol["makespan"]), float(sol["shortage"])), bounds)
        for sol in pareto_front
    ]
    ref_points = [normalize_point(r, bounds) for r in reference_front]

    distances = []
    for r in ref_points:
        min_dist = min(math.dist(r, p) for p in points)
        distances.append(min_dist)

    return sum(distances) / len(distances)


# =====================================
# 田口试验设计
# =====================================

def decode_trial_row(trial_id, row):
    """
    row 是 L16 中的一行，比如 [1,2,2,3]
    返回该 trial 的真实参数
    """
    params = {}
    for factor_name, level_idx in zip(FACTOR_NAMES, row):
        params[factor_name] = FACTOR_LEVELS[factor_name][level_idx - 1]

    ps = params["PS"]
    main_pop = ps // 3
    global_pop = ps // 3
    local_pop = ps - main_pop - global_pop
    ls = params["LS"]

    decoded = {
        "trial_id": trial_id,
        "factor_levels": {
            "PS": params["PS"],
            "Pc": params["Pc"],
            "Pm": params["Pm"],
            "LS": params["LS"],
        },
        "algorithm_params": {
            "pop_size": main_pop,
            "global_pop_size": global_pop,
            "local_pop_size": local_pop,
            "max_evaluations": MAX_EVALUATIONS,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "crossover_rate": params["Pc"],
            "os_mutation_rate": params["Pm"],
            "ms_mutation_rate": params["Pm"],
            "tournament_size": 2,
            "gat_improve_window": 5,
            "gat_improve_threshold": 0.005,
            "local_elite_count": ls,
            "local_neighbors_per_elite": 6,
            "local_os_mutation_rate": min(1.0, params["Pm"] + 0.1),
            "local_ms_mutation_rate": min(1.0, params["Pm"] + 0.1),
        }
    }
    return decoded


def build_trial_configs():
    trials = []
    for i, row in enumerate(L16_TABLE, start=1):
        trials.append(decode_trial_row(i, row))
    return trials


# =====================================
# 单次运行
# =====================================

def run_one_trial_seed(instance_path, trial_config, seed):
    _, operations, buffers, _ = load_instance_from_json(instance_path)

    for bid in buffers:
        if "low_wip" not in buffers[bid]:
            buffers[bid]["low_wip"] = 1

    algo_params = deepcopy(trial_config["algorithm_params"])
    algo_params["seed"] = seed

    search = EMTGLocalGAV2(
        operations=operations,
        buffers=buffers,
        **algo_params
    )

    t0 = time.time()
    best = search.run(
        store_stats_init=True,
        store_stats_generations=False,
        verbose=False
    )
    runtime = time.time() - t0

    pareto_front = search.get_pareto_front()

    pareto_solutions = []
    for ind in pareto_front:
        pareto_solutions.append({
            "makespan": ind.makespan,
            "shortage": ind.shortage,
            "OS": ind.OS,
            "MS": ind.MS,
        })

    result = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "instance_name": os.path.splitext(os.path.basename(instance_path))[0],
        "instance_path": instance_path,
        "trial_id": trial_config["trial_id"],
        "seed": seed,
        "factor_levels": trial_config["factor_levels"],
        "algorithm": "EMTGLocalGAV2",
        "algorithm_parameters": algo_params,
        "representative_result": {
            "makespan": best.makespan,
            "shortage": best.shortage,
            "runtime": runtime,
            "n_evaluations": search.n_evaluations,
        },
        "pareto_front": pareto_solutions,
        "igd": None,
    }

    return result


# =====================================
# 保存 / 读取
# =====================================

def save_run_json(result):
    os.makedirs(RUN_DIR, exist_ok=True)
    filename = f"trial{result['trial_id']:02d}_seed{result['seed']}.json"
    path = os.path.join(RUN_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return path


def load_all_run_jsons():
    if not os.path.exists(RUN_DIR):
        return []

    results = []
    for fn in sorted(os.listdir(RUN_DIR)):
        if not fn.endswith(".json"):
            continue
        if fn.startswith("_"):
            continue

        path = os.path.join(RUN_DIR, fn)
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))

    return results


def validate_loaded_results(all_results):
    if not all_results:
        raise RuntimeError(f"RUN_DIR 中没有可分析的 json 文件: {RUN_DIR}")

    required_keys = [
        "instance_name",
        "trial_id",
        "seed",
        "factor_levels",
        "pareto_front",
    ]

    for i, res in enumerate(all_results):
        for key in required_keys:
            if key not in res:
                raise ValueError(f"第 {i} 个结果缺少字段: {key}")

        if not isinstance(res["pareto_front"], list):
            raise ValueError(f"第 {i} 个结果的 pareto_front 不是 list")

        for j, sol in enumerate(res["pareto_front"]):
            if "makespan" not in sol or "shortage" not in sol:
                raise ValueError(f"第 {i} 个结果的 pareto_front[{j}] 缺少 makespan/shortage")


# =====================================
# 汇总分析
# =====================================

def summarize_taguchi_results(all_results):
    all_fronts = [res["pareto_front"] for res in all_results]
    reference_front = build_reference_front(all_fronts)
    bounds = compute_normalization_bounds(reference_front)

    # 给每个 run 计算 IGD
    for res in all_results:
        res["igd"] = compute_igd(res["pareto_front"], reference_front, bounds)

    # trial -> 多个seed的平均IGD
    trial_to_igds = defaultdict(list)
    trial_to_factor_levels = {}

    for res in all_results:
        trial_id = res["trial_id"]
        trial_to_igds[trial_id].append(res["igd"])
        trial_to_factor_levels[trial_id] = res["factor_levels"]

    trial_summary = []
    for trial_id in sorted(trial_to_igds.keys()):
        igds = trial_to_igds[trial_id]
        trial_summary.append({
            "trial_id": trial_id,
            "factor_levels": trial_to_factor_levels[trial_id],
            "mean_igd": statistics.mean(igds),
            "std_igd": statistics.stdev(igds) if len(igds) > 1 else 0.0,
            "all_igd": igds,
        })

    main_effects = {}
    for factor in FACTOR_NAMES:
        level_effect = {}
        for level in FACTOR_LEVELS[factor]:
            vals = [
                rec["mean_igd"]
                for rec in trial_summary
                if rec["factor_levels"][factor] == level
            ]
            level_effect[level] = statistics.mean(vals)
        main_effects[factor] = level_effect

    best_levels = {}
    for factor in FACTOR_NAMES:
        best_level = min(main_effects[factor].items(), key=lambda x: x[1])[0]
        best_levels[factor] = best_level

    return {
        "instance_name": os.path.splitext(os.path.basename(INSTANCE_PATH))[0],
        "reference_front": reference_front,
        "normalization_bounds": bounds,
        "trial_summary": trial_summary,
        "main_effects": main_effects,
        "best_levels": best_levels,
    }


# =====================================
# 画主效应图
# =====================================

def plot_main_effects_connected(main_effects, out_path):
    """
    画成“4段连在一起”的主效应图，并带顶部表头风格。
    """
    factor_order = ["PS", "Pc", "Pm", "LS"]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    x_positions = []
    x_labels = []
    y_values = []
    group_centers = []
    separators = []
    group_ranges = []

    cur = 0
    for i, factor in enumerate(factor_order):
        levels = list(main_effects[factor].keys())
        vals = [main_effects[factor][lv] for lv in levels]

        xs = list(range(cur, cur + len(levels)))

        x_positions.extend(xs)
        x_labels.extend([str(lv) for lv in levels])
        y_values.extend(vals)

        group_centers.append(sum(xs) / len(xs))
        group_ranges.append((xs[0] - 0.5, xs[-1] + 0.5))

        cur += len(levels)
        if i < len(factor_order) - 1:
            separators.append(cur - 0.5)

    fig, ax = plt.subplots(figsize=(12, 5.8))

    ax.plot(x_positions, y_values, marker="o", linewidth=1.6, markersize=6)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("Average IGD", fontsize=13, fontweight="bold")
    ax.set_xlabel("Factor levels", fontsize=12)
    ax.set_title("Main Effects Plot for Average IGD", fontsize=16, fontweight="bold", pad=10)

    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for sep in separators:
        ax.axvline(x=sep, linestyle="--", linewidth=1.0)

    ymin = min(y_values)
    ymax = max(y_values)

    if ymax > ymin:
        yrange = ymax - ymin
    else:
        yrange = max(0.01, abs(ymax) * 0.1 + 0.01)

    plot_ymin = ymin - 0.08 * yrange
    plot_ymax = ymax + 0.22 * yrange
    ax.set_ylim(plot_ymin, plot_ymax)

    header_bottom = ymax + 0.04 * yrange
    header_top = ymax + 0.11 * yrange

    ax.hlines(header_top, group_ranges[0][0], group_ranges[-1][1], colors="0.65", linewidth=1.1)
    ax.hlines(header_bottom, group_ranges[0][0], group_ranges[-1][1], colors="0.65", linewidth=1.1)

    header_boundaries = [group_ranges[0][0]] + separators + [group_ranges[-1][1]]
    for xb in header_boundaries:
        ax.vlines(xb, header_bottom, header_top, colors="0.65", linewidth=1.1)

    for center, factor in zip(group_centers, factor_order):
        ax.text(
            center,
            (header_bottom + header_top) / 2,
            factor,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold"
        )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =====================================
# 保存汇总
# =====================================

def save_summary_json(summary_dict):
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    path = os.path.join(SUMMARY_DIR, "taguchi_summary.json")

    serializable_summary = {
        "instance_name": summary_dict["instance_name"],
        "reference_front": [
            {"makespan": p[0], "shortage": p[1]}
            for p in summary_dict["reference_front"]
        ],
        "normalization_bounds": summary_dict["normalization_bounds"],
        "trial_summary": summary_dict["trial_summary"],
        "main_effects": {
            factor: {str(level): value for level, value in level_dict.items()}
            for factor, level_dict in summary_dict["main_effects"].items()
        },
        "best_levels": summary_dict["best_levels"],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2, ensure_ascii=False)

    return path


# =====================================
# 运行模式：跑实验
# =====================================

def run_all_trials():
    os.makedirs(RUN_DIR, exist_ok=True)

    trial_configs = build_trial_configs()
    total_runs = len(trial_configs) * len(SEEDS)
    run_counter = 0

    print("Instance:", INSTANCE_PATH)
    print(f"Trials: {len(trial_configs)}")
    print(f"Seeds per trial: {len(SEEDS)}")
    print(f"Total runs: {total_runs}")

    for trial_cfg in trial_configs:
        print("\n--------------------------------")
        print(f"Trial {trial_cfg['trial_id']:02d} | factors = {trial_cfg['factor_levels']}")

        for seed in SEEDS:
            result = run_one_trial_seed(INSTANCE_PATH, trial_cfg, seed)
            save_path = save_run_json(result)

            run_counter += 1
            print(
                f"[RUN {run_counter}/{total_runs}] "
                f"trial={trial_cfg['trial_id']:02d} "
                f"seed={seed} "
                f"rep_makespan={result['representative_result']['makespan']} "
                f"rep_shortage={result['representative_result']['shortage']:.2f} "
                f"evals={result['representative_result']['n_evaluations']} "
                f"time={result['representative_result']['runtime']:.2f}s "
                f"saved={save_path}"
            )


# =====================================
# 运行模式：只分析已有结果
# =====================================

def analyze_existing_results():
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    all_results = load_all_run_jsons()
    validate_loaded_results(all_results)

    summary = summarize_taguchi_results(all_results)

    # 把算出来的 igd 回写到每个 run json
    for res in all_results:
        filename = f"trial{res['trial_id']:02d}_seed{res['seed']}.json"
        path = os.path.join(RUN_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    summary_path = save_summary_json(summary)

    fig_path = os.path.join(SUMMARY_DIR, "main_effects_igd.png")
    plot_main_effects_connected(summary["main_effects"], fig_path)

    print("\n================================")
    print("Taguchi analysis finished.")
    print("Best levels:", summary["best_levels"])
    print("Summary JSON:", summary_path)
    print("Main effects plot:", fig_path)


# =====================================
# 主程序
# =====================================

def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    mode = MODE
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()

    if mode == "run":
        run_all_trials()
    elif mode == "analyze":
        analyze_existing_results()
    elif mode == "both":
        run_all_trials()
        analyze_existing_results()
    else:
        raise ValueError(f"未知模式: {mode}")


if __name__ == "__main__":
    main()