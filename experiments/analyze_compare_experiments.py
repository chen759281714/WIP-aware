import os
import json
import math
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt

# =========================================
# 路径配置
# =========================================

RUN_RESULT_DIR = "experiments/results/runs"
ANALYSIS_DIR = "experiments/results/analysis_compare"
SUMMARY_DIR = os.path.join(ANALYSIS_DIR, "summary")
FIG_DIR = os.path.join(ANALYSIS_DIR, "figures")

# 如果你想固定算法显示顺序，可以在这里写
ALGO_ORDER = [
    "EMTGLocalGAV2",
    "BaselineNSGA2",
    "BaselineMOEAD",
]

# 若系统没有 Times New Roman，会自动回退
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# =========================================
# 基础工具
# =========================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_mean(values):
    if not values:
        return None
    return statistics.mean(values)


def safe_std(values):
    if len(values) <= 1:
        return 0.0 if values else None
    return statistics.stdev(values)


# =========================================
# Pareto / GD / IGD 工具
# =========================================

def dominates_point(a, b):
    """
    最小化问题下，点 a 是否支配点 b
    a, b: (makespan, shortage)
    """
    return (
        a[0] <= b[0] and
        a[1] <= b[1] and
        (a[0] < b[0] or a[1] < b[1])
    )


def filter_nondominated(points):
    """
    输入一组二维点，返回非支配点集（最小化）
    """
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


def build_reference_front(all_fronts):
    """
    all_fronts: List[List[{"makespan":..., "shortage":...}, ...]]
    """
    all_points = []

    for front in all_fronts:
        for sol in front:
            try:
                m = float(sol["makespan"])
                s = float(sol["shortage"])
                all_points.append((m, s))
            except Exception:
                continue

    return filter_nondominated(all_points)


def compute_normalization_bounds(reference_front):
    if not reference_front:
        return {
            "m_min": 0.0,
            "m_max": 1.0,
            "s_min": 0.0,
            "s_max": 1.0,
        }

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


def compute_gd(pareto_front, reference_front, bounds):
    """
    归一化 GD：算法前沿 -> 参考前沿
    """
    if not pareto_front or not reference_front:
        return float("inf")

    points = [
        normalize_point((float(sol["makespan"]), float(sol["shortage"])), bounds)
        for sol in pareto_front
    ]
    ref_points = [normalize_point(r, bounds) for r in reference_front]

    distances = []
    for p in points:
        min_dist = min(math.dist(p, r) for r in ref_points)
        distances.append(min_dist)

    return sum(distances) / len(distances)


def compute_igd(pareto_front, reference_front, bounds):
    """
    归一化 IGD：参考前沿 -> 算法前沿
    """
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


# =========================================
# JSON 读取
# =========================================

def load_all_run_jsons(run_root):
    """
    读取 runs/<algorithm>/*.json
    返回列表，每个元素是一条 run 记录
    """
    records = []

    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"运行结果目录不存在: {run_root}")

    for algo_name in os.listdir(run_root):
        algo_dir = os.path.join(run_root, algo_name)

        if not os.path.isdir(algo_dir):
            continue
        if algo_name.startswith("_"):
            continue

        for fn in os.listdir(algo_dir):
            if not fn.endswith(".json"):
                continue

            path = os.path.join(algo_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 补一层保护，防止路径层和内容层不一致
            data["_file_path"] = path
            data["_algo_dir"] = algo_name
            records.append(data)

    return records


def group_records_by_instance(records):
    grouped = defaultdict(list)
    for rec in records:
        instance = rec.get("instance_name", "UNKNOWN_INSTANCE")
        grouped[instance].append(rec)
    return grouped


def group_records_by_algo(records):
    grouped = defaultdict(list)
    for rec in records:
        algo = rec.get("algorithm", rec.get("_algo_dir", "UNKNOWN_ALGO"))
        grouped[algo].append(rec)
    return grouped


# =========================================
# front_history 相关
# =========================================

def get_snapshot_fronts(rec):
    """
    从 run json 中提取快照前沿：
    返回 [(eval_count, pareto_front), ...]
    """
    front_history = rec.get("front_history", {})
    snapshots = front_history.get("snapshots", [])

    result = []
    for snap in snapshots:
        eval_count = snap.get("eval_count", None)
        front = snap.get("pareto_front", [])
        if eval_count is None:
            continue
        result.append((int(eval_count), front))

    return result


def compute_run_convergence_igd(rec, reference_front, bounds):
    """
    对单个 run 的 front_history 中每个 snapshot 计算 IGD
    返回:
    [
        {"eval_count": ..., "igd": ...},
        ...
    ]
    """
    snapshots = get_snapshot_fronts(rec)
    curve = []

    for eval_count, front in snapshots:
        igd = compute_igd(front, reference_front, bounds)
        curve.append({
            "eval_count": eval_count,
            "igd": igd
        })

    return curve


def aggregate_mean_curve(curves):
    """
    输入若干条 run 的曲线，每条形如:
    [
      {"eval_count": 2000, "igd": 0.1},
      {"eval_count": 4000, "igd": 0.08},
      ...
    ]

    按 eval_count 对齐求均值
    """
    bucket = defaultdict(list)

    for curve in curves:
        for pt in curve:
            bucket[int(pt["eval_count"])].append(float(pt["igd"]))

    xs = sorted(bucket.keys())
    ys = [statistics.mean(bucket[x]) for x in xs]

    return xs, ys


# =========================================
# 作图
# =========================================

def get_algo_display_order(algo_names):
    ordered = [a for a in ALGO_ORDER if a in algo_names]
    remaining = sorted([a for a in algo_names if a not in ordered])
    return ordered + remaining


def plot_convergence_curves(instance_name, algo_to_curves, out_path):
    """
    每个算法一条“平均 IGD 收敛曲线”
    横轴：evaluation_count
    纵轴：IGD
    """
    plt.figure(figsize=(8, 5))

    algo_names = get_algo_display_order(list(algo_to_curves.keys()))

    for algo in algo_names:
        curves = algo_to_curves[algo]
        if not curves:
            continue

        xs, ys = aggregate_mean_curve(curves)
        if not xs:
            continue

        plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=algo)

    plt.xlabel("Evaluation Count", fontsize=12)
    plt.ylabel("IGD", fontsize=12)
    plt.title(f"IGD Convergence Curve - {instance_name}", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot(instance_name, algo_to_values, metric_name, out_path):
    """
    画箱线图，metric_name 可为 'GD' 或 'IGD'
    """
    algo_names = get_algo_display_order(list(algo_to_values.keys()))
    data = [algo_to_values[a] for a in algo_names]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=algo_names, showmeans=True)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f"{metric_name} Boxplot - {instance_name}", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================
# 主分析逻辑
# =========================================

def analyze_one_instance(instance_name, records):
    """
    对一个算例下的所有算法、所有seed结果做分析
    """
    # 1) 构造 reference front
    all_fronts = [rec.get("pareto_front", []) for rec in records]
    reference_front = build_reference_front(all_fronts)
    bounds = compute_normalization_bounds(reference_front)

    # 2) 为每个 run 计算 GD / IGD
    for rec in records:
        front = rec.get("pareto_front", [])
        rec["_gd"] = compute_gd(front, reference_front, bounds)
        rec["_igd"] = compute_igd(front, reference_front, bounds)
        rec["_convergence_igd"] = compute_run_convergence_igd(rec, reference_front, bounds)

    # 3) 分算法汇总
    algo_groups = group_records_by_algo(records)

    algo_summary = {}
    algo_to_gd_values = {}
    algo_to_igd_values = {}
    algo_to_curves = {}

    for algo, recs in algo_groups.items():
        gd_values = [r["_gd"] for r in recs]
        igd_values = [r["_igd"] for r in recs]

        algo_to_gd_values[algo] = gd_values
        algo_to_igd_values[algo] = igd_values
        algo_to_curves[algo] = [r["_convergence_igd"] for r in recs if r["_convergence_igd"]]

        rep_makespans = []
        rep_shortages = []
        runtimes = []

        for r in recs:
            rr = r.get("representative_result", {})
            if rr.get("makespan") is not None:
                rep_makespans.append(float(rr["makespan"]))
            if rr.get("shortage") is not None:
                rep_shortages.append(float(rr["shortage"]))
            if rr.get("runtime") is not None:
                runtimes.append(float(rr["runtime"]))

        algo_summary[algo] = {
            "n_runs": len(recs),
            "gd_mean": safe_mean(gd_values),
            "gd_std": safe_std(gd_values),
            "igd_mean": safe_mean(igd_values),
            "igd_std": safe_std(igd_values),
            "rep_makespan_mean": safe_mean(rep_makespans),
            "rep_shortage_mean": safe_mean(rep_shortages),
            "runtime_mean": safe_mean(runtimes),
        }

    return {
        "instance_name": instance_name,
        "reference_front": [
            {"makespan": p[0], "shortage": p[1]}
            for p in reference_front
        ],
        "normalization_bounds": bounds,
        "algorithm_summary": algo_summary,
        "algo_to_gd_values": algo_to_gd_values,
        "algo_to_igd_values": algo_to_igd_values,
        "algo_to_curves": algo_to_curves,
    }


def save_instance_summary(instance_result):
    instance_name = instance_result["instance_name"]
    out_path = os.path.join(SUMMARY_DIR, f"{instance_name}_summary.json")

    save_obj = {
        "instance_name": instance_result["instance_name"],
        "reference_front": instance_result["reference_front"],
        "normalization_bounds": instance_result["normalization_bounds"],
        "algorithm_summary": instance_result["algorithm_summary"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, indent=2, ensure_ascii=False)

    return out_path


def save_global_summary(all_instance_results):
    out_path = os.path.join(SUMMARY_DIR, "_global_summary.json")

    summary = {}
    for res in all_instance_results:
        summary[res["instance_name"]] = res["algorithm_summary"]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return out_path


def generate_figures_for_instance(instance_result):
    instance_name = instance_result["instance_name"]

    # IGD 收敛曲线
    conv_path = os.path.join(FIG_DIR, f"{instance_name}_igd_convergence.png")
    plot_convergence_curves(
        instance_name=instance_name,
        algo_to_curves=instance_result["algo_to_curves"],
        out_path=conv_path
    )

    # GD 箱线图
    gd_box_path = os.path.join(FIG_DIR, f"{instance_name}_gd_boxplot.png")
    plot_boxplot(
        instance_name=instance_name,
        algo_to_values=instance_result["algo_to_gd_values"],
        metric_name="GD",
        out_path=gd_box_path
    )

    # IGD 箱线图
    igd_box_path = os.path.join(FIG_DIR, f"{instance_name}_igd_boxplot.png")
    plot_boxplot(
        instance_name=instance_name,
        algo_to_values=instance_result["algo_to_igd_values"],
        metric_name="IGD",
        out_path=igd_box_path
    )

    return {
        "igd_convergence": conv_path,
        "gd_boxplot": gd_box_path,
        "igd_boxplot": igd_box_path,
    }


# =========================================
# main
# =========================================

def main():
    ensure_dir(ANALYSIS_DIR)
    ensure_dir(SUMMARY_DIR)
    ensure_dir(FIG_DIR)

    all_records = load_all_run_jsons(RUN_RESULT_DIR)
    if not all_records:
        raise RuntimeError("未读取到任何 run JSON，请先运行 run_compare_experiments.py")

    instance_groups = group_records_by_instance(all_records)

    all_instance_results = []

    print("Start analyzing compare experiment results...")
    print(f"Total run files: {len(all_records)}")
    print(f"Total instances: {len(instance_groups)}")

    for instance_name in sorted(instance_groups.keys()):
        print(f"\nAnalyzing instance: {instance_name}")

        instance_result = analyze_one_instance(
            instance_name=instance_name,
            records=instance_groups[instance_name]
        )

        summary_path = save_instance_summary(instance_result)
        fig_paths = generate_figures_for_instance(instance_result)

        print(f"  Summary saved: {summary_path}")
        print(f"  IGD convergence figure: {fig_paths['igd_convergence']}")
        print(f"  GD boxplot: {fig_paths['gd_boxplot']}")
        print(f"  IGD boxplot: {fig_paths['igd_boxplot']}")

        all_instance_results.append(instance_result)

    global_summary_path = save_global_summary(all_instance_results)

    print("\nAnalysis finished.")
    print(f"Global summary saved: {global_summary_path}")


if __name__ == "__main__":
    main()