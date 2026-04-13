# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import json
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
from openpyxl import Workbook

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import load_instance_from_json
from src.solution.encoder import Encoder
from src.solution.decoder import StageBufferWIPScheduler


# =========================================================
# 用户配置区
# =========================================================

ALGO_NAME = "EliteLSGA"
INSTANCE_NAME = "WIP-FMS_05"
SEED = 1

RUN_JSON_PATH = os.path.join(
    PROJECT_ROOT,
    "experiments", "results", "runs", ALGO_NAME, f"{INSTANCE_NAME}_seed{SEED}.json"
)

INSTANCE_JSON_PATH = os.path.join(
    PROJECT_ROOT,
    "data", "instances", "WIP-FMS", f"{INSTANCE_NAME}.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments", "results", "analysis", f"{INSTANCE_NAME}_{ALGO_NAME}_seed{SEED}"
)

# 指定想看的 buffer；若为 None，则自动选第一个 buffer
TARGET_BUFFER_ID = None

# 是否保存图片
SAVE_FIGURES = True

# 是否显示图片
SHOW_FIGURES = True


# =========================================================
# 基础工具函数
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_run_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_solution_has_os_ms(sol: Dict[str, Any]) -> None:
    if "OS" not in sol or "MS" not in sol:
        raise ValueError(
            "Pareto solution 中缺少 OS/MS。\n"
            "请先在 run_compare_experiments.py 保存 pareto_front 时加入:\n"
            '  "OS": ind.OS,\n'
            '  "MS": ind.MS\n'
        )


def select_abc_solutions(pareto_front: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    A: makespan 最优
    B: shortage 最优
    C: 折中解（按 makespan 排序后的中位解）
    """
    if not pareto_front:
        raise ValueError("Pareto front 为空，无法选择 A/B/C 解")

    for sol in pareto_front:
        require_solution_has_os_ms(sol)

    A = min(pareto_front, key=lambda x: (x["makespan"], x["shortage"]))
    B = min(pareto_front, key=lambda x: (x["shortage"], x["makespan"]))

    front_sorted = sorted(pareto_front, key=lambda x: (x["makespan"], x["shortage"]))
    C = front_sorted[len(front_sorted) // 2]

    return {"A": A, "B": B, "C": C}


def build_ms_map_from_solution(
    operations: Dict[str, List[Dict[str, Any]]],
    ms_list: List[str]
) -> Dict[Tuple[str, int], str]:
    encoder = Encoder(operations)
    return encoder.build_ms_map(ms_list)


def evaluate_solution(
    operations: Dict[str, List[Dict[str, Any]]],
    buffers: Dict[str, Dict[str, Any]],
    sol: Dict[str, Any],
) -> Dict[str, Any]:
    scheduler = StageBufferWIPScheduler(operations, buffers)

    os_seq = sol["OS"]
    ms_list = sol["MS"]
    ms_map = build_ms_map_from_solution(operations, ms_list)

    makespan, schedule, buffer_trace = scheduler.decode(os_seq=os_seq, ms_map=ms_map)
    stats = scheduler.analyze(schedule=schedule, buffer_trace=buffer_trace, makespan=makespan)

    return {
        "makespan": makespan,
        "schedule": schedule,
        "buffer_trace": buffer_trace,
        "stats": stats,
        "OS": os_seq,
        "MS": ms_list,
    }


def pick_target_buffer(buffers: Dict[str, Dict[str, Any]], target_buffer_id: str | None) -> str:
    if target_buffer_id is not None:
        if target_buffer_id not in buffers:
            raise ValueError(f"指定的 buffer_id={target_buffer_id} 不存在")
        return target_buffer_id

    bids = sorted(buffers.keys())
    if not bids:
        raise ValueError("实例中没有 buffer")
    return bids[0]


def extract_solution_metrics(
    result: Dict[str, Any],
    buffers: Dict[str, Dict[str, Any]],
    operations: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    stats = result["stats"]

    total_empty_time = sum(stats["buffers"]["per_buffer_empty_time"].values())
    total_empty_ratio = sum(stats["buffers"]["per_buffer_empty_ratio"].values()) / max(
        1, len(stats["buffers"]["per_buffer_empty_ratio"])
    )

    total_below_low_time = stats["shortage"]["total_below_low_time"]
    total_shortage_area = stats["shortage"]["total_shortage_area"]
    total_blocking_time = stats["blocking"]["total_blocking_time"]

    downstream_idle_per_buffer = compute_downstream_idle_per_buffer(
        stats=stats,
        operations=operations,
        buffers=buffers,
    )
    total_downstream_idle = sum(downstream_idle_per_buffer.values())

    return {
        "makespan": result["makespan"],
        "shortage": total_shortage_area,
        "below_low_time": total_below_low_time,
        "blocking": total_blocking_time,
        "avg_empty_ratio": total_empty_ratio,
        "total_empty_time": total_empty_time,
        "total_downstream_idle": total_downstream_idle,
    }


def flatten_per_buffer_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    stats = result["stats"]

    row = {}

    per_below = stats["shortage"]["per_buffer_below_low_time"]
    per_shortage = stats["shortage"]["per_buffer_shortage_area"]
    per_empty = stats["buffers"]["per_buffer_empty_time"]
    per_avg_level = stats["buffers"]["per_buffer_avg_level"]

    for bid, val in per_below.items():
        row[f"{bid}_below_low_time"] = val
    for bid, val in per_shortage.items():
        row[f"{bid}_shortage_area"] = val
    for bid, val in per_empty.items():
        row[f"{bid}_empty_time"] = val
    for bid, val in per_avg_level.items():
        row[f"{bid}_avg_level"] = val

    return row


# =========================================================
# 输出表格
# =========================================================

def compute_downstream_idle_per_buffer(
    stats: Dict[str, Any],
    operations: Dict[str, List[Dict[str, Any]]],
    buffers: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """
    对每个 buffer，统计“使用该 buffer 作为 buffer_in 的下游机器”的总 idle time。
    例如：
      B01 -> 所有其下游工序(buffer_in == B01)可用机器的 idle time 之和
    """
    idle_times = stats["machines"]["per_machine_idle_time"]
    downstream_idle = {}

    for bid in buffers.keys():
        downstream_machines = set()

        for job, ops in operations.items():
            for op in ops:
                if op.get("buffer_in", None) == bid:
                    downstream_machines.update(op["machines"].keys())

        total_idle = sum(idle_times.get(m, 0) for m in downstream_machines)
        downstream_idle[bid] = total_idle

    return downstream_idle

def save_comparison_excel(
    output_path: str,
    abc_results: Dict[str, Dict[str, Any]],
    buffers: Dict[str, Dict[str, Any]],
    operations: Dict[str, List[Dict[str, Any]]],
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "ABC_Comparison"

    base_header = [
        "Solution",
        "makespan",
        "shortage",
        "below_low_time",
        "blocking",
        "avg_empty_ratio",
        "total_empty_time",
        "total_downstream_idle",
    ]

    # 补充所有 per-buffer 列
    extra_cols = []
    if abc_results:
        sample_name = next(iter(abc_results))
        sample_extra = flatten_per_buffer_metrics(abc_results[sample_name])
        extra_cols = list(sample_extra.keys())

    ws.append(base_header + extra_cols)

    for name in ["A", "B", "C"]:
        result = abc_results[name]
        base = extract_solution_metrics(result, buffers, operations)
        extra = flatten_per_buffer_metrics(result)

        row = [
            name,
            base["makespan"],
            base["shortage"],
            base["below_low_time"],
            base["blocking"],
            base["avg_empty_ratio"],
            base["total_empty_time"],
            base["total_downstream_idle"],
        ] + [extra.get(c, "") for c in extra_cols]

        ws.append(row)

    wb.save(output_path)


# =========================================================
# 作图：Pareto front
# =========================================================

def plot_pareto_front(
    pareto_front: List[Dict[str, Any]],
    abc_selected: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    xs = [p["makespan"] for p in pareto_front]
    ys = [p["shortage"] for p in pareto_front]

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, label="Pareto solutions")

    for label, sol in abc_selected.items():
        plt.scatter(
            sol["makespan"],
            sol["shortage"],
            s=100,
            marker="x",
            label=f"{label}"
        )
        plt.annotate(
            label,
            (sol["makespan"], sol["shortage"]),
            textcoords="offset points",
            xytext=(5, 5)
        )

    plt.xlabel("Makespan")
    plt.ylabel("Shortage")
    plt.title("Pareto Front with A/B/C Solutions")
    plt.legend()
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(os.path.join(output_dir, "pareto_front_abc.png"), dpi=200)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


# =========================================================
# 作图：buffer level 曲线
# =========================================================

def events_to_step_xy(events: List[Tuple[int, int, str, Any]], makespan: int):
    """
    将 buffer_trace 事件日志转成 step 图可直接绘制的 x/y
    events: [(t, level, action, job), ...]
    """
    if not events:
        return [0, makespan], [0, 0]

    events_sorted = sorted(events, key=lambda x: x[0])

    xs = []
    ys = []

    # 从第一条事件开始
    first_t = int(events_sorted[0][0])
    first_level = int(events_sorted[0][1])

    if first_t > 0:
        xs.extend([0, first_t])
        ys.extend([first_level, first_level])

    for i in range(len(events_sorted) - 1):
        t_i = int(events_sorted[i][0])
        level_i = int(events_sorted[i][1])
        t_j = int(events_sorted[i + 1][0])

        xs.extend([t_i, t_j])
        ys.extend([level_i, level_i])

    last_t = int(events_sorted[-1][0])
    last_level = int(events_sorted[-1][1])

    if last_t < makespan:
        xs.extend([last_t, makespan])
        ys.extend([last_level, last_level])

    if not xs:
        xs = [0, makespan]
        ys = [first_level, first_level]

    return xs, ys


def plot_buffer_curves(
    abc_results: Dict[str, Dict[str, Any]],
    buffers: Dict[str, Dict[str, Any]],
    target_buffer_id: str,
    output_dir: str
) -> None:
    low_wip = int(buffers[target_buffer_id].get("low_wip", 0))
    cap = int(buffers[target_buffer_id]["capacity"])

    plt.figure(figsize=(8, 5))

    for label in ["A", "B", "C"]:
        result = abc_results[label]
        events = result["buffer_trace"][target_buffer_id]
        makespan = result["makespan"]

        xs, ys = events_to_step_xy(events, makespan)
        plt.plot(xs, ys, label=f"{label}")

    plt.axhline(y=low_wip, linestyle="--", label=f"low_wip={low_wip}")
    plt.axhline(y=cap, linestyle=":", label=f"capacity={cap}")

    plt.xlabel("Time")
    plt.ylabel("Buffer Level")
    plt.title(f"Buffer Level Curve: {target_buffer_id}")
    plt.legend()
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(os.path.join(output_dir, f"buffer_curve_{target_buffer_id}.png"), dpi=200)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


# =========================================================
# 作图：简易甘特图
# =========================================================

def plot_gantt(
    result: Dict[str, Any],
    label: str,
    output_dir: str
) -> None:
    schedule = result["schedule"]

    machines = sorted(list({rec["machine"] for rec in schedule}))
    machine_to_y = {m: i for i, m in enumerate(machines)}

    plt.figure(figsize=(10, 5))

    for rec in schedule:
        y = machine_to_y[rec["machine"]]
        start = rec["start"]
        duration = rec["end"] - rec["start"]
        plt.barh(y=y, width=duration, left=start, height=0.6)
        plt.text(start + duration / 2, y, f"{rec['job']}-{rec['op']}", ha="center", va="center", fontsize=7)

        # 若存在 blocking，可画 release 延迟
        if rec["release"] > rec["end"]:
            block_dur = rec["release"] - rec["end"]
            plt.barh(y=y, width=block_dur, left=rec["end"], height=0.25)

    plt.yticks(range(len(machines)), machines)
    plt.xlabel("Time")
    plt.ylabel("Machine")
    plt.title(f"Gantt Chart - Solution {label}")
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(os.path.join(output_dir, f"gantt_{label}.png"), dpi=200)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


# =========================================================
# 作图：指标柱状图
# =========================================================

def plot_metric_bars(
    abc_results: Dict[str, Dict[str, Any]],
    buffers: Dict[str, Dict[str, Any]],
    operations: Dict[str, List[Dict[str, Any]]],
    output_dir: str
) -> None:
    labels = ["A", "B", "C"]

    makespans = []
    shortages = []
    below_low_times = []

    for name in labels:
        metrics = extract_solution_metrics(abc_results[name], buffers, operations)
        makespans.append(metrics["makespan"])
        shortages.append(metrics["shortage"])
        below_low_times.append(metrics["below_low_time"])

    x = range(len(labels))

    # makespan
    plt.figure(figsize=(6, 4))
    plt.bar(x, makespans)
    plt.xticks(list(x), labels)
    plt.ylabel("Makespan")
    plt.title("Makespan Comparison")
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(os.path.join(output_dir, "bar_makespan.png"), dpi=200)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()

    # shortage
    plt.figure(figsize=(6, 4))
    plt.bar(x, shortages)
    plt.xticks(list(x), labels)
    plt.ylabel("Shortage")
    plt.title("Shortage Comparison")
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(os.path.join(output_dir, "bar_shortage.png"), dpi=200)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()

    # below_low_time
    plt.figure(figsize=(6, 4))
    plt.bar(x, below_low_times)
    plt.xticks(list(x), labels)
    plt.ylabel("Below-Low Time")
    plt.title("Below-Low-Time Comparison")
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(os.path.join(output_dir, "bar_below_low_time.png"), dpi=200)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()

# =========================================================
# 作图：空缓冲与下游机器闲置对比柱状图
# =========================================================
def plot_idle_vs_empty_bar(
    abc_results: Dict[str, Dict[str, Any]],
    buffers: Dict[str, Dict[str, Any]],
    operations: Dict[str, List[Dict[str, Any]]],
    output_dir: str
) -> None:
    labels = ["A", "B", "C"]

    empty_times = []
    downstream_idles = []

    for name in labels:
        stats = abc_results[name]["stats"]
        total_empty_time = sum(stats["buffers"]["per_buffer_empty_time"].values())
        per_downstream_idle = compute_downstream_idle_per_buffer(
            stats=stats,
            operations=operations,
            buffers=buffers,
        )
        total_downstream_idle = sum(per_downstream_idle.values())

        empty_times.append(total_empty_time)
        downstream_idles.append(total_downstream_idle)

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar([i - width / 2 for i in x], empty_times, width=width, label="buffer_empty_time")
    plt.bar([i + width / 2 for i in x], downstream_idles, width=width, label="downstream_idle")

    plt.xticks(list(x), labels)
    plt.ylabel("Time")
    plt.title("Buffer Empty Time vs Downstream Idle")
    plt.legend()
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(os.path.join(output_dir, "bar_empty_vs_downstream_idle.png"), dpi=200)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


# =========================================================
# 打印结果
# =========================================================

def print_abc_summary(
    abc_results: Dict[str, Dict[str, Any]],
    buffers: Dict[str, Dict[str, Any]],
    operations: Dict[str, List[Dict[str, Any]]],
) -> None:
    print("\n===== A/B/C Detailed Comparison =====")
    for name in ["A", "B", "C"]:
        metrics = extract_solution_metrics(abc_results[name], buffers, operations)
        print(f"\n[{name}]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        per_below = abc_results[name]["stats"]["shortage"]["per_buffer_below_low_time"]
        print("  per_buffer_below_low_time:", per_below)

        per_short = abc_results[name]["stats"]["shortage"]["per_buffer_shortage_area"]
        print("  per_buffer_shortage_area :", per_short)

        per_empty = abc_results[name]["stats"]["buffers"]["per_buffer_empty_time"]
        print("  per_buffer_empty_time    :", per_empty)

        per_downstream_idle = compute_downstream_idle_per_buffer(
            stats=abc_results[name]["stats"],
            operations=operations,
            buffers=buffers,
        )
        print("  per_buffer_downstream_idle:", per_downstream_idle)


# =========================================================
# 主程序
# =========================================================

def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.exists(RUN_JSON_PATH):
        raise FileNotFoundError(f"未找到 run json: {RUN_JSON_PATH}")
    if not os.path.exists(INSTANCE_JSON_PATH):
        raise FileNotFoundError(f"未找到 instance json: {INSTANCE_JSON_PATH}")

    # 1) 读取 run json
    run_data = load_run_json(RUN_JSON_PATH)
    pareto_front = run_data.get("pareto_front", [])
    if not pareto_front:
        raise ValueError("run json 中没有 pareto_front")

    # 2) 读取实例
    spec, operations, buffers, _ = load_instance_from_json(INSTANCE_JSON_PATH)

    # 若 instance 中没有 low_wip，则与实验脚本保持一致，默认补 1
    for bid in buffers:
        if "low_wip" not in buffers[bid]:
            buffers[bid]["low_wip"] = 1

    target_buffer_id = pick_target_buffer(buffers, TARGET_BUFFER_ID)

    # 3) 选 A/B/C
    abc_selected = select_abc_solutions(pareto_front)

    print("Selected A/B/C from Pareto front:")
    for name in ["A", "B", "C"]:
        sol = abc_selected[name]
        print(
            f"  {name}: makespan={sol['makespan']}, "
            f"shortage={sol['shortage']}, "
            f"blocking={sol.get('blocking', 'N/A')}"
        )

    # 4) 重新 decode + analyze
    abc_results = {}
    for name in ["A", "B", "C"]:
        abc_results[name] = evaluate_solution(operations, buffers, abc_selected[name])

    # 5) 打印详细信息
    print_abc_summary(abc_results, buffers, operations)

    # 6) 存表
    excel_path = os.path.join(OUTPUT_DIR, "abc_comparison.xlsx")
    save_comparison_excel(excel_path, abc_results, buffers, operations)
    print(f"\nExcel saved to: {excel_path}")

    # 7) 作图
    plot_pareto_front(pareto_front, abc_selected, OUTPUT_DIR)
    plot_buffer_curves(abc_results, buffers, target_buffer_id, OUTPUT_DIR)

    # 甘特图：建议先画 A / B / C
    plot_gantt(abc_results["A"], "A", OUTPUT_DIR)
    plot_gantt(abc_results["B"], "B", OUTPUT_DIR)
    plot_gantt(abc_results["C"], "C", OUTPUT_DIR)

    plot_metric_bars(abc_results, buffers, operations, OUTPUT_DIR)
    plot_idle_vs_empty_bar(abc_results, buffers, operations, OUTPUT_DIR)

    print(f"Analysis outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()