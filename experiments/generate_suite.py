# -*- coding: utf-8 -*-
"""
生成 20 个 WIP-FMS 测试集（JSON落盘）并输出 suite_manifest.csv

本版本：按用户要求扩大规模差异，并随规模增大并行机数量 K
- Small : S=3,  K=2, N=20  -> 6 instances
- Medium: S=5,  K=3, N=60  -> 7 instances
- Large : S=8,  K=4, N=120 -> 7 instances

核心目标：
- 覆盖不同瓶颈形态（downstream / mid / balanced）
- buffer_caps：
    1) 先用公式给初值（基于 stage_avg_pt & K）
    2) 再用 decode+analyze 的反馈迭代校准，使 blocking_ratio 进入目标区间

blocking_ratio = total_blocking / makespan
默认 sweet spot：
- target_low = 0.05
- target_high = 0.20

输出：
- data/instances/WIP-FMS/*.json
- data/instances/WIP-FMS/suite_manifest.csv
"""

import os
import csv
import math
import sys

# ===== 把项目根目录加入 sys.path，解决 ModuleNotFoundError: No module named 'src' =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import (
    InstanceSpec,
    generate_fms_wip_instance_auto_caps,
    save_instance_to_json,
)
from src.solution.decoder import StageBufferWIPScheduler


# ------------------------------
# 1) caps 缩放（缩小时 floor，放大时 ceil）
# ------------------------------
def scale_caps(caps, factor, cap_min=1, cap_max=120):
    """
    factor < 1：缩小（用 floor，避免取整卡死）
    factor > 1：放大（用 ceil，保证确实变大）
    """
    out = []
    for c in caps:
        if factor < 1.0:
            cc = int(math.floor(c * factor))
        else:
            cc = int(math.ceil(c * factor))
        cc = max(cap_min, min(cap_max, cc))
        out.append(cc)
    return out


def build_buffers_from_caps(num_stages, caps):
    """由 caps 构建 buffers 字典：B01,B12,..."""
    assert len(caps) == num_stages - 1
    return {f"B{i}{i+1}": {"capacity": int(c)} for i, c in enumerate(caps)}


# ------------------------------
# 2) 一次 decode+analyze 得到 quick stats
# ------------------------------
def compute_quick_stats(operations, buffers, os_seq):
    """
    跑一次 decode+analyze，返回 quick stats
    注意：这里的 total_blocking 是 scheduler 统计的“机器释放被 buffer 满阻塞”的时间
    """
    sch = StageBufferWIPScheduler(operations, buffers)
    makespan, schedule, buffer_trace = sch.decode(os_seq)
    stats = sch.analyze(schedule, buffer_trace, makespan=makespan)

    def _buf(bid, key, default=0):
        return stats["buffers"].get(key, {}).get(bid, default)

    total_blocking = stats["blocking"]["total_blocking_time"]
    blocking_ratio = total_blocking / max(1, makespan)

    out = {
        "makespan": makespan,
        "total_blocking": total_blocking,
        "blocking_ratio": blocking_ratio,

        "B01_avg": _buf("B01", "per_buffer_avg_level", 0.0),
        "B01_full_time": _buf("B01", "per_buffer_full_time", 0),
        "B01_empty_ratio": _buf("B01", "per_buffer_empty_ratio", 0.0),

        "B12_avg": _buf("B12", "per_buffer_avg_level", 0.0),
        "B12_full_time": _buf("B12", "per_buffer_full_time", 0),
        "B12_empty_ratio": _buf("B12", "per_buffer_empty_ratio", 0.0),
    }
    return out


# ------------------------------
# 3) 反馈校准 caps 到 sweet spot
# ------------------------------
def tune_caps_to_sweet_spot(
    operations,
    os_seq,
    num_stages,
    initial_caps,
    target_low=0.05,
    target_high=0.20,
    max_iters=14,
    cap_min=1,
    cap_max=120,
):
    """
    用 blocking_ratio 反馈迭代调整 caps。

    - ratio < target_low ：caps 缩小（更容易满载 → 更易 blocking）
    - ratio > target_high：caps 放大（降低满载频率 → 减少 blocking）
    - 进入区间即停止
    - 若缩放后 caps 不再变化（到达取整/边界），提前停止
    """
    caps = list(initial_caps)
    last_quick = None

    for it in range(1, max_iters + 1):
        buffers = build_buffers_from_caps(num_stages, caps)
        quick = compute_quick_stats(operations, buffers, os_seq)
        last_quick = quick

        ratio = quick["blocking_ratio"]
        tb = quick["total_blocking"]

        # 命中 sweet spot
        if target_low <= ratio <= target_high:
            return caps, quick, it, "hit"

        old_caps = list(caps)

        # blocking 太弱：缩小 caps
        if ratio < target_low:
            factor = 0.70 if tb == 0 else 0.85
            caps = scale_caps(caps, factor, cap_min=cap_min, cap_max=cap_max)

        # blocking 太强：放大 caps
        elif ratio > target_high:
            caps = scale_caps(caps, 1.15, cap_min=cap_min, cap_max=cap_max)

        # caps 不变，说明被取整/边界卡住
        if caps == old_caps:
            return caps, last_quick, it, "stuck"

    return caps, last_quick, max_iters, "max_iter"


# ------------------------------
# 4) 主流程：生成 20 个实例并落盘
# ------------------------------
def main():
    out_dir = os.path.join("data", "instances", "WIP-FMS")
    os.makedirs(out_dir, exist_ok=True)

    # 三档规模：S、K、N 拉开；os_repeat 保持较大，保证调参统计稳定
    tiers = [
        {"tag": "small",  "num_stages": 3, "machines_per_stage": 2, "n_jobs": 20,  "os_repeat": 120},
        {"tag": "mid",    "num_stages": 5, "machines_per_stage": 3, "n_jobs": 60,  "os_repeat": 120},
        {"tag": "large",  "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "os_repeat": 120},
    ]

    # 共 20 个实例（6 + 7 + 7）
    plan = {
        "small": [
            ("downstream_bottleneck", [11, 22]),
            ("mid_bottleneck",        [33, 44]),
            ("balanced",              [55, 66]),
        ],
        "mid": [
            ("downstream_bottleneck", [11, 22, 33]),
            ("mid_bottleneck",        [44, 55]),
            ("balanced",              [66, 77]),
        ],
        "large": [
            ("downstream_bottleneck", [11, 22, 33]),
            ("mid_bottleneck",        [44, 55]),
            ("balanced",              [66, 77]),
        ],
    }

    # sweet spot 区间
    target_low = 0.05
    target_high = 0.20

    # caps 上下限：K 变大，cap_max 放宽
    cap_min = 1
    cap_max = 120

    manifest_path = os.path.join(out_dir, "suite_manifest.csv")
    fieldnames = [
        "file",
        "tier",
        "num_stages", "machines_per_stage", "n_jobs",
        "pt_profile", "seed",
        "pt_low", "pt_high", "os_repeat",
        "stage_avg_pt",
        "buffer_caps",
        "makespan", "total_blocking", "blocking_ratio",
        "B01_avg", "B01_full_time", "B01_empty_ratio",
        "B12_avg", "B12_full_time", "B12_empty_ratio",
        "tune_iters", "tune_status",
    ]

    rows = []

    for tier in tiers:
        tier_tag = tier["tag"]

        for prof, seeds in plan[tier_tag]:
            for sd in seeds:
                # 先给一个“占位 caps”（长度必须 S-1），auto_caps 会覆盖成初值
                placeholder_caps = [1] * (tier["num_stages"] - 1)

                spec = InstanceSpec(
                    num_stages=tier["num_stages"],
                    machines_per_stage=tier["machines_per_stage"],
                    n_jobs=tier["n_jobs"],
                    buffer_caps=placeholder_caps,
                    pt_profile=prof,
                    pt_low=1,
                    pt_high=10,
                    seed=sd,
                    os_repeat=tier["os_repeat"],
                )

                # 1) 公式生成初值 caps + operations
                base_spec, operations, _, os_seq, stage_avg = generate_fms_wip_instance_auto_caps(
                    spec,
                    alpha=2.5, beta=0.5,
                    min_mult=2.0, max_mult=4.0,
                )

                # 2) 反馈校准 caps 到 sweet spot
                tuned_caps, quick, iters_used, status = tune_caps_to_sweet_spot(
                    operations=operations,
                    os_seq=os_seq,
                    num_stages=base_spec.num_stages,
                    initial_caps=base_spec.buffer_caps,
                    target_low=target_low,
                    target_high=target_high,
                    max_iters=14,
                    cap_min=cap_min,
                    cap_max=cap_max,
                )

                buffers = build_buffers_from_caps(base_spec.num_stages, tuned_caps)

                # 3) final spec 写回 tuned_caps
                final_spec = InstanceSpec(
                    num_stages=base_spec.num_stages,
                    machines_per_stage=base_spec.machines_per_stage,
                    n_jobs=base_spec.n_jobs,
                    buffer_caps=tuned_caps,
                    pt_profile=base_spec.pt_profile,
                    pt_low=base_spec.pt_low,
                    pt_high=base_spec.pt_high,
                    seed=base_spec.seed,
                    os_repeat=base_spec.os_repeat,
                )

                # 4) 保存 JSON
                fname = (
                    f"WIP-FMS_{tier_tag}_S{final_spec.num_stages}_K{final_spec.machines_per_stage}_"
                    f"N{final_spec.n_jobs}_{prof}_seed{sd}.json"
                )
                fpath = os.path.join(out_dir, fname)
                save_instance_to_json(fpath, final_spec, operations, buffers)

                # 5) manifest
                rows.append({
                    "file": fname,
                    "tier": tier_tag,
                    "num_stages": final_spec.num_stages,
                    "machines_per_stage": final_spec.machines_per_stage,
                    "n_jobs": final_spec.n_jobs,
                    "pt_profile": final_spec.pt_profile,
                    "seed": final_spec.seed,
                    "pt_low": final_spec.pt_low,
                    "pt_high": final_spec.pt_high,
                    "os_repeat": final_spec.os_repeat,
                    "stage_avg_pt": str([round(x, 3) for x in stage_avg]),
                    "buffer_caps": str(final_spec.buffer_caps),
                    "makespan": quick["makespan"],
                    "total_blocking": quick["total_blocking"],
                    "blocking_ratio": f"{quick['blocking_ratio']:.6f}",
                    "B01_avg": f"{quick['B01_avg']:.6f}",
                    "B01_full_time": quick["B01_full_time"],
                    "B01_empty_ratio": f"{quick['B01_empty_ratio']:.6f}",
                    "B12_avg": f"{quick['B12_avg']:.6f}",
                    "B12_full_time": quick["B12_full_time"],
                    "B12_empty_ratio": f"{quick['B12_empty_ratio']:.6f}",
                    "tune_iters": iters_used,
                    "tune_status": status,
                })

                print(
                    "generated:", fname,
                    "| K:", final_spec.machines_per_stage,
                    "| caps:", tuned_caps,
                    "| makespan:", quick["makespan"],
                    "| blocking:", quick["total_blocking"],
                    "| ratio:", f"{quick['blocking_ratio']:.3f}",
                    "| iters:", iters_used,
                    "| status:", status
                )

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n✅ suite generated (20 instances, tuned caps)")
    print("instances dir:", out_dir)
    print("manifest:", manifest_path)
    print(f"sweet spot: blocking_ratio in [{target_low}, {target_high}]")


if __name__ == "__main__":
    main()