# -*- coding: utf-8 -*-
"""
实例难度分析脚本

功能：
1. 读取 data/instances/WIP-FMS/*.json
2. 调用 StageBufferWIPScheduler 进行 decode + analyze
3. 计算实例难度相关指标
4. 输出 CSV 报告：
   data/instances/WIP-FMS/instance_difficulty_report.csv

运行方式（项目根目录）：
    python experiments/instance_difficulty_analyzer.py
"""

import os
import csv
import glob
from typing import Dict, List, Any
import sys

# ===== 把项目根目录加入 sys.path，解决 ModuleNotFoundError: No module named 'src' =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import load_instance_from_json
from src.solution.decoder import StageBufferWIPScheduler


def compute_lower_bound(
    operations: Dict[str, List[Dict[str, Any]]],
    machines_per_stage: int
) -> float:
    """
    计算一个简单但有意义的理论下界 LB。

    这里取两部分中的最大值：
    1) 单个工件沿工艺路线的最短总加工时间的最大值
    2) 总加工负载 / 总机器数

    注意：
    - 每道工序取其候选机器中的最短加工时间，作为“理想最短时间”
    - 这是一个粗粒度下界，不追求特别紧，但足够做 difficulty 分析
    """
    # 1) 每个 job 的最短总加工时间
    job_lb_list = []
    total_min_processing = 0.0

    for job, ops in operations.items():
        job_sum = 0.0
        for op in ops:
            min_pt = min(op["machines"].values())
            job_sum += min_pt
            total_min_processing += min_pt
        job_lb_list.append(job_sum)

    lb_job = max(job_lb_list) if job_lb_list else 0.0

    # 2) 总加工量 / 总机器数
    # 这里用“所有工序最短加工时间之和 / 每阶段机器数总和的近似”
    # 更准确的做法是分工段负载下界，这里先保持简单
    num_stages = len(next(iter(operations.values()))) if operations else 0
    total_machines = num_stages * machines_per_stage
    lb_load = total_min_processing / max(1, total_machines)

    return max(lb_job, lb_load)


def summarize_machine_utilization(stats: Dict[str, Any]) -> Dict[str, float]:
    """
    从 analyze(stats) 中提取机器利用率摘要。
    """
    util_dict = stats["machines"]["per_machine_utilization"]
    if not util_dict:
        return {
            "avg_machine_util": 0.0,
            "max_machine_util": 0.0,
            "min_machine_util": 0.0,
        }

    vals = list(util_dict.values())
    return {
        "avg_machine_util": sum(vals) / len(vals),
        "max_machine_util": max(vals),
        "min_machine_util": min(vals),
    }


def get_buffer_metric(stats: Dict[str, Any], bid: str, key: str, default=0):
    """
    安全读取 buffer 指标。
    """
    return stats["buffers"].get(key, {}).get(bid, default)


def classify_instance_difficulty(
    blocking_ratio: float,
    gap: float,
    avg_util: float
) -> str:
    """
    给一个非常粗略的难度标签，便于人工浏览。
    这不是严格理论分类，只是帮助你快速筛实例。
    """
    # 先看 blocking 是否明显
    if blocking_ratio < 0.03 and gap < 1.15 and avg_util < 0.65:
        return "easy"

    # 明显存在 WIP 影响，且仍有优化空间
    if 0.05 <= blocking_ratio <= 0.20 and 1.15 <= gap <= 2.00:
        return "good"

    # blocking 很强或系统很紧张
    if blocking_ratio > 0.20 or avg_util > 0.90:
        return "hard"

    return "moderate"


def analyze_instance(json_path: str) -> Dict[str, Any]:
    """
    对单个 JSON 实例做难度分析。
    """
    spec, operations, buffers, os_seq = load_instance_from_json(json_path)

    sch = StageBufferWIPScheduler(operations, buffers)
    makespan, schedule, buffer_trace = sch.decode(os_seq)
    stats = sch.analyze(schedule, buffer_trace, makespan=makespan)

    # 基本指标
    total_blocking = stats["blocking"]["total_blocking_time"]
    blocking_ratio = total_blocking / max(1, makespan)

    # 机器利用率
    util_summary = summarize_machine_utilization(stats)
    avg_util = util_summary["avg_machine_util"]
    max_util = util_summary["max_machine_util"]
    min_util = util_summary["min_machine_util"]

    # 理论下界与 gap
    lb = compute_lower_bound(operations, spec.machines_per_stage)
    gap = makespan / max(1e-9, lb)

    # 缓冲区指标（只重点看前两个，够你当前 3-stage / 多 stage 的初步分析）
    b01_avg = get_buffer_metric(stats, "B01", "per_buffer_avg_level", 0.0)
    b01_full_time = get_buffer_metric(stats, "B01", "per_buffer_full_time", 0)
    b12_avg = get_buffer_metric(stats, "B12", "per_buffer_avg_level", 0.0)
    b12_full_time = get_buffer_metric(stats, "B12", "per_buffer_full_time", 0)

    difficulty = classify_instance_difficulty(blocking_ratio, gap, avg_util)

    return {
        "file": os.path.basename(json_path),
        "num_stages": spec.num_stages,
        "machines_per_stage": spec.machines_per_stage,
        "n_jobs": spec.n_jobs,
        "pt_profile": spec.pt_profile,
        "seed": spec.seed,
        "buffer_caps": str(spec.buffer_caps),

        "makespan": makespan,
        "LB": round(lb, 6),
        "gap_makespan_over_LB": round(gap, 6),

        "total_blocking": total_blocking,
        "blocking_ratio": round(blocking_ratio, 6),

        "avg_machine_util": round(avg_util, 6),
        "max_machine_util": round(max_util, 6),
        "min_machine_util": round(min_util, 6),

        "B01_avg": round(b01_avg, 6),
        "B01_full_time": b01_full_time,
        "B12_avg": round(b12_avg, 6),
        "B12_full_time": b12_full_time,

        "difficulty_label": difficulty,
    }


def main():
    instance_dir = os.path.join("data", "instances", "WIP-FMS")
    out_csv = os.path.join(instance_dir, "instance_difficulty_report.csv")

    json_files = sorted(glob.glob(os.path.join(instance_dir, "*.json")))
    if not json_files:
        print("未找到任何 JSON 实例文件。")
        print("请先运行 experiments/generate_suite.py")
        return

    rows = []
    for fp in json_files:
        row = analyze_instance(fp)
        rows.append(row)
        print(
            f"analyzed: {row['file']} | "
            f"gap={row['gap_makespan_over_LB']:.3f} | "
            f"blocking_ratio={row['blocking_ratio']:.3f} | "
            f"difficulty={row['difficulty_label']}"
        )

    fieldnames = [
        "file",
        "num_stages", "machines_per_stage", "n_jobs",
        "pt_profile", "seed", "buffer_caps",
        "makespan", "LB", "gap_makespan_over_LB",
        "total_blocking", "blocking_ratio",
        "avg_machine_util", "max_machine_util", "min_machine_util",
        "B01_avg", "B01_full_time",
        "B12_avg", "B12_full_time",
        "difficulty_label",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n✅ 实例难度分析完成")
    print("输出文件：", out_csv)


if __name__ == "__main__":
    main()