# -*- coding: utf-8 -*-
"""
生成分层版 WIP-FMS benchmark suite（20 个实例）

特性：
1. 实例文件统一命名为 WIP-FMS_01 ~ WIP-FMS_20
2. 采用 level-based benchmark 分层：
   - easy
   - moderate
   - hard
   - very_hard
3. 自动调节 buffer_caps 到 sweet spot
4. 生成 instance_description.xlsx，记录每个实例的关键信息
5. 支持显式控制：
   - bottleneck strength
   - machine heterogeneity
   - buffer tightness
6. 支持按 level + variant 设置不同的 blocking 接受区间
"""

import os
import math
import sys
import random
from openpyxl import Workbook

# ===== 把项目根目录加入 sys.path，解决 ModuleNotFoundError: No module named 'src' =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import (
    InstanceSpec,
    generate_fms_wip_instance_auto_caps,
    save_instance_to_json,
    build_stage_weights,
)
from src.solution.decoder import StageBufferWIPScheduler


# ------------------------------
# 1) caps 缩放（缩小时 floor，放大时 ceil）
# ------------------------------
def scale_caps(caps, factor, cap_min=1, cap_max=40):
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
    assert len(caps) == num_stages - 1
    return {f"B{i}{i+1}": {"capacity": int(c)} for i, c in enumerate(caps)}


# ------------------------------
# 2) 一次 decode+analyze 得到 quick stats
# ------------------------------
def compute_quick_stats(operations, buffers, os_seq):
    sch = StageBufferWIPScheduler(operations, buffers)
    makespan, schedule, buffer_trace = sch.decode(os_seq)
    stats = sch.analyze(schedule, buffer_trace, makespan=makespan)

    total_blocking = stats["blocking"]["total_blocking_time"]
    blocking_ratio = total_blocking / max(1, makespan)

    out = {
        "makespan": makespan,
        "total_blocking": total_blocking,
        "blocking_ratio": blocking_ratio,
    }
    return out


def compute_multi_os_quick_stats(operations, buffers, os_repeat, n_samples=3, seed=0):
    """
    对同一个 instance，使用多个随机 OS 进行 quick evaluation，
    返回平均统计量，避免只依赖某一个 OS 导致误判。
    """
    rng = random.Random(seed)

    jobs = sorted(list(operations.keys()), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

    makespans = []
    total_blockings = []
    blocking_ratios = []

    for _ in range(n_samples):
        os_seq = jobs * int(os_repeat)
        rng.shuffle(os_seq)

        sch = StageBufferWIPScheduler(operations, buffers)
        makespan, schedule, buffer_trace = sch.decode(os_seq)
        stats = sch.analyze(schedule, buffer_trace, makespan=makespan)

        total_blocking = stats["blocking"]["total_blocking_time"]
        blocking_ratio = total_blocking / max(1, makespan)

        makespans.append(makespan)
        total_blockings.append(total_blocking)
        blocking_ratios.append(blocking_ratio)

    return {
        "avg_makespan": sum(makespans) / len(makespans),
        "avg_total_blocking": sum(total_blockings) / len(total_blockings),
        "avg_blocking_ratio": sum(blocking_ratios) / len(blocking_ratios),
    }


# ------------------------------
# 3) 反馈校准 caps 到 sweet spot
# ------------------------------
def tune_caps_to_sweet_spot(
    operations,
    os_seq,
    num_stages,
    initial_caps,
    target_low=0.05,
    target_high=0.15,
    max_iters=14,
    cap_min=1,
    cap_max=40,
):
    caps = list(initial_caps)
    last_quick = None

    for it in range(1, max_iters + 1):
        buffers = build_buffers_from_caps(num_stages, caps)
        quick = compute_quick_stats(operations, buffers, os_seq)
        last_quick = quick

        ratio = quick["blocking_ratio"]
        tb = quick["total_blocking"]

        if target_low <= ratio <= target_high:
            return caps, quick, it, "hit"

        old_caps = list(caps)

        if ratio < target_low:
            factor = 0.70 if tb == 0 else 0.85
            caps = scale_caps(caps, factor, cap_min=cap_min, cap_max=cap_max)

        elif ratio > target_high:
            caps = scale_caps(caps, 1.15, cap_min=cap_min, cap_max=cap_max)

        if caps == old_caps:
            return caps, last_quick, it, "stuck"

    return caps, last_quick, max_iters, "max_iter"


# ------------------------------
# 4) buffer tightness 参数映射
# ------------------------------
def get_buffer_mults(level: str):
    if level == "tight":
        return 0.8, 1.4
    if level == "loose":
        return 1.4, 2.5
    # default: medium
    return 1.0, 2.0


# ------------------------------
# 5) level + variant 配置映射
# ------------------------------
def get_level_config(level: str, scale: str, variant: int):
    """
    根据难度层级 + 变体编号返回实例控制参数。
    设计原则：
    1. scale 主尺寸固定（S, K, N 在 plan 中固定）
    2. 同一 level 内通过 variant 形成不同结构模板
    3. 每个 level 仍对应一个相对稳定的 blocking 区间
    """

    if level == "easy":
        # easy: WIP 作用较弱，但不能完全消失
        if variant == 1:
            return {
                "scenario": "balanced",
                "strength": "mild",
                "heterogeneity": "medium",
                "buffer_tightness": "medium",
                "accept_low": 0.06,
                "accept_high": 0.11,
            }
        else:
            return {
                "scenario": "balanced",
                "strength": "moderate",
                "heterogeneity": "low",
                "buffer_tightness": "medium",
                "accept_low": 0.06,
                "accept_high": 0.11,
            }

    if level == "moderate":
        # moderate: 标准 WIP 层
        if variant == 1:
            return {
                "scenario": "balanced" if scale == "small" else "mid_bottleneck",
                "strength": "moderate",
                "heterogeneity": "medium",
                "buffer_tightness": "medium",
                "accept_low": 0.10,
                "accept_high": 0.15,
            }
        else:
            return {
                "scenario": "mid_bottleneck",
                "strength": "mild",
                "heterogeneity": "high",
                "buffer_tightness": "medium",
                "accept_low": 0.10,
                "accept_high": 0.15,
            }

    if level == "hard":
        # hard: blocking 明显，但不至于太极端
        if variant == 1:
            return {
                "scenario": "mid_bottleneck",
                "strength": "moderate",
                "heterogeneity": "high",
                "buffer_tightness": "tight",
                "accept_low": 0.12,
                "accept_high": 0.18,
            }
        elif variant == 2:
            return {
                "scenario": "downstream_bottleneck",
                "strength": "moderate",
                "heterogeneity": "medium",
                "buffer_tightness": "medium",
                "accept_low": 0.12,
                "accept_high": 0.18,
            }
        else:
            return {
                "scenario": "mid_bottleneck",
                "strength": "strong",
                "heterogeneity": "medium",
                "buffer_tightness": "medium",
                "accept_low": 0.12,
                "accept_high": 0.2,
            }

    if level == "very_hard":
        # very_hard: 强 WIP / 强瓶颈，但内部要避免同质化
        if variant == 1:
            return {
                "scenario": "downstream_bottleneck",
                "strength": "strong",
                "heterogeneity": "high",
                "buffer_tightness": "tight",
                "accept_low": 0.14,
                "accept_high": 0.21,
            }
        elif variant == 2:
            return {
                "scenario": "downstream_bottleneck",
                "strength": "strong",
                "heterogeneity": "medium",
                "buffer_tightness": "medium",
                "accept_low": 0.14,
                "accept_high": 0.21,
            }
        elif variant == 3:
            return {
                "scenario": "downstream_bottleneck",
                "strength": "strong",
                "heterogeneity": "medium",
                "buffer_tightness": "tight",
                "accept_low": 0.14,
                "accept_high": 0.21,
            }
        else:
            return {
                "scenario": "downstream_bottleneck",
                "strength": "moderate",
                "heterogeneity": "high",
                "buffer_tightness": "tight",
                "accept_low": 0.14,
                "accept_high": 0.21,
            }

    raise ValueError(f"unknown level: {level}")


# ------------------------------
# 6) 分层 benchmark 计划
# ------------------------------
def build_level_plan():
    """
    构建分层 benchmark 计划。

    设计原则：
    1. scale 主尺寸固定
       - small : S=3, K=2, N=20
       - medium: S=5, K=3, N=60
       - large : S=8, K=4, N=120

    2. 同一 level 内通过 variant 区分不同结构模板
    3. 不再通过改变 stage/job 数量来制造“假层次”
    """
    return [
        # ===== small =====
        {"instance": "WIP-FMS_01", "scale": "small", "level": "easy",      "variant": 1, "num_stages": 3, "machines_per_stage": 2, "n_jobs": 20,  "seed": 55, "os_repeat": 120},
        {"instance": "WIP-FMS_02", "scale": "small", "level": "moderate",  "variant": 1, "num_stages": 3, "machines_per_stage": 2, "n_jobs": 20,  "seed": 33, "os_repeat": 120},
        {"instance": "WIP-FMS_03", "scale": "small", "level": "hard",      "variant": 1, "num_stages": 3, "machines_per_stage": 2, "n_jobs": 20,  "seed": 11, "os_repeat": 120},

        # ===== medium =====
        {"instance": "WIP-FMS_04", "scale": "medium", "level": "easy",       "variant": 1, "num_stages": 5, "machines_per_stage": 3, "n_jobs": 60, "seed": 66, "os_repeat": 120},
        {"instance": "WIP-FMS_05", "scale": "medium", "level": "moderate",   "variant": 1, "num_stages": 5, "machines_per_stage": 3, "n_jobs": 60, "seed": 77, "os_repeat": 120},
        {"instance": "WIP-FMS_06", "scale": "medium", "level": "moderate",   "variant": 2, "num_stages": 5, "machines_per_stage": 3, "n_jobs": 60, "seed": 44, "os_repeat": 120},
        {"instance": "WIP-FMS_07", "scale": "medium", "level": "hard",       "variant": 1, "num_stages": 5, "machines_per_stage": 3, "n_jobs": 60, "seed": 55, "os_repeat": 120},
        {"instance": "WIP-FMS_08", "scale": "medium", "level": "hard",       "variant": 2, "num_stages": 5, "machines_per_stage": 3, "n_jobs": 60, "seed": 22, "os_repeat": 120},
        {"instance": "WIP-FMS_09", "scale": "medium", "level": "very_hard",  "variant": 1, "num_stages": 5, "machines_per_stage": 3, "n_jobs": 60, "seed": 33, "os_repeat": 120},
        {"instance": "WIP-FMS_10", "scale": "medium", "level": "very_hard",  "variant": 2, "num_stages": 5, "machines_per_stage": 3, "n_jobs": 60, "seed": 44, "os_repeat": 120},

        # ===== large =====
        {"instance": "WIP-FMS_11", "scale": "large", "level": "easy",       "variant": 1, "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "seed": 88, "os_repeat": 120},
        {"instance": "WIP-FMS_12", "scale": "large", "level": "moderate",   "variant": 1, "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "seed": 99, "os_repeat": 120},
        {"instance": "WIP-FMS_13", "scale": "large", "level": "moderate",   "variant": 2, "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "seed": 66, "os_repeat": 120},
        {"instance": "WIP-FMS_14", "scale": "large", "level": "hard",       "variant": 1, "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "seed": 77, "os_repeat": 120},
        {"instance": "WIP-FMS_15", "scale": "large", "level": "hard",       "variant": 2, "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "seed": 88, "os_repeat": 120},
        {"instance": "WIP-FMS_16", "scale": "large", "level": "hard",       "variant": 3, "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "seed": 55, "os_repeat": 120},
        {"instance": "WIP-FMS_17", "scale": "large", "level": "very_hard",  "variant": 1, "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "seed": 66, "os_repeat": 120},
        {"instance": "WIP-FMS_18", "scale": "large", "level": "very_hard",  "variant": 2, "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "seed": 77, "os_repeat": 120},
        {"instance": "WIP-FMS_19", "scale": "large", "level": "very_hard",  "variant": 3, "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "seed": 88, "os_repeat": 120},
        {"instance": "WIP-FMS_20", "scale": "large", "level": "very_hard",  "variant": 4, "num_stages": 8, "machines_per_stage": 4, "n_jobs": 120, "seed": 99, "os_repeat": 120},
    ]


# ------------------------------
# 7) 保存 benchmark 描述 Excel
# ------------------------------
def save_instance_description_excel(out_dir, rows):
    path = os.path.join(out_dir, "instance_description.xlsx")

    wb = Workbook()
    ws = wb.active
    ws.title = "WIP-FMS Suite"

    header = [
        "Instance",
        "Scale",
        "Level",
        "Variant",
        "Scenario",
        "Strength",
        "Heterogeneity",
        "BufferTightness",
        "Seed",
        "Jobs",
        "Stages",
        "Machines/Stage",
        "pt_low",
        "pt_high",
        "os_repeat",
        "StageAvgPT",
        "BufferCaps",
        "QuickMakespan",
        "QuickBlocking",
        "QuickBlockingRatio",
        "MultiAvgMakespan",
        "MultiAvgBlocking",
        "MultiAvgBlockingRatio",
        "TuneIters",
        "TuneStatus",
    ]
    ws.append(header)

    for row in rows:
        ws.append([
            row["instance"],
            row["scale"],
            row["level"],
            row["variant"],
            row["scenario"],
            row["strength"],
            row["heterogeneity"],
            row["buffer_tightness"],
            row["seed"],
            row["jobs"],
            row["stages"],
            row["machines_per_stage"],
            row["pt_low"],
            row["pt_high"],
            row["os_repeat"],
            row["stage_avg_pt"],
            row["buffer_caps"],
            row["quick_makespan"],
            row["quick_blocking"],
            row["quick_blocking_ratio"],
            row["multi_avg_makespan"],
            row["multi_avg_blocking"],
            row["multi_avg_blocking_ratio"],
            row["tune_iters"],
            row["tune_status"],
        ])

    wb.save(path)
    print("instance description saved to:", path)


# ------------------------------
# 8) 主流程：生成 20 个实例并落盘
# ------------------------------
def main():
    out_dir = os.path.join("data", "instances", "WIP-FMS")
    os.makedirs(out_dir, exist_ok=True)

    plan = build_level_plan()

    # quick tuning 统一一套区间
    target_low = 0.05
    target_high = 0.15
    cap_min = 1
    cap_max = 40
    max_regen_tries = 30

    description_rows = []

    for item in plan:

        success = False

        cfg = get_level_config(item["level"], item["scale"], item["variant"])

        for regen_try in range(1, max_regen_tries + 1):

            current_seed = item["seed"] + regen_try - 1

            placeholder_caps = [1] * (item["num_stages"] - 1)

            spec = InstanceSpec(
                num_stages=item["num_stages"],
                machines_per_stage=item["machines_per_stage"],
                n_jobs=item["n_jobs"],
                buffer_caps=placeholder_caps,
                pt_profile=cfg["scenario"],
                pt_low=1,
                pt_high=10,
                seed=current_seed,
                os_repeat=item["os_repeat"],
            )

            stage_weights = build_stage_weights(
                cfg["scenario"],
                cfg["strength"],
                item["num_stages"]
            )

            min_mult, max_mult = get_buffer_mults(cfg["buffer_tightness"])

            # 1) 公式生成初值 caps + operations
            base_spec, operations, _, os_seq, stage_avg = generate_fms_wip_instance_auto_caps(
                spec,
                alpha=1.8,
                beta=0.6,
                min_mult=min_mult,
                max_mult=max_mult,
                stage_weights_override=stage_weights,
                heterogeneity_level=cfg["heterogeneity"],
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

            multi_quick = compute_multi_os_quick_stats(
                operations=operations,
                buffers=buffers,
                os_repeat=base_spec.os_repeat,
                n_samples=3,
                seed=base_spec.seed + 999
            )

            accept_multi_low = cfg["accept_low"]
            accept_multi_high = cfg["accept_high"]

            multi_ratio = multi_quick["avg_blocking_ratio"]

            if multi_ratio < accept_multi_low or multi_ratio > accept_multi_high:
                print(
                    f"REJECT: {item['instance']} "
                    f"level={item['level']} "
                    f"variant={item['variant']} "
                    f"try={regen_try} "
                    f"seed={current_seed} "
                    f"multi_ratio={multi_ratio:.4f} "
                    f"(target: [{accept_multi_low:.2f}, {accept_multi_high:.2f}])"
                )
                continue

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

            # 4) 保存 JSON（统一命名）
            fname = f"{item['instance']}.json"
            fpath = os.path.join(out_dir, fname)
            save_instance_to_json(fpath, final_spec, operations, buffers)

            # 5) 记录 Excel 描述信息
            description_rows.append({
                "instance": item["instance"],
                "scale": item["scale"],
                "level": item["level"],
                "variant": item["variant"],
                "scenario": cfg["scenario"],
                "strength": cfg["strength"],
                "heterogeneity": cfg["heterogeneity"],
                "buffer_tightness": cfg["buffer_tightness"],
                "seed": final_spec.seed,
                "jobs": final_spec.n_jobs,
                "stages": final_spec.num_stages,
                "machines_per_stage": final_spec.machines_per_stage,
                "pt_low": final_spec.pt_low,
                "pt_high": final_spec.pt_high,
                "os_repeat": final_spec.os_repeat,
                "stage_avg_pt": str([round(x, 3) for x in stage_avg]),
                "buffer_caps": str(final_spec.buffer_caps),
                "quick_makespan": quick["makespan"],
                "quick_blocking": quick["total_blocking"],
                "quick_blocking_ratio": round(quick["blocking_ratio"], 6),
                "multi_avg_makespan": round(multi_quick["avg_makespan"], 3),
                "multi_avg_blocking": round(multi_quick["avg_total_blocking"], 3),
                "multi_avg_blocking_ratio": round(multi_quick["avg_blocking_ratio"], 6),
                "tune_iters": iters_used,
                "tune_status": status,
            })

            print(
                "generated:", fname,
                "| scale:", item["scale"],
                "| level:", item["level"],
                "| variant:", item["variant"],
                "| scenario:", cfg["scenario"],
                "| strength:", cfg["strength"],
                "| heterogeneity:", cfg["heterogeneity"],
                "| buffer:", cfg["buffer_tightness"],
                "| K:", final_spec.machines_per_stage,
                "| caps:", tuned_caps,
                "| makespan:", quick["makespan"],
                "| blocking:", quick["total_blocking"],
                "| ratio:", f"{quick['blocking_ratio']:.3f}",
                "| multi_ratio:", f"{multi_ratio:.3f}",
                "| iters:", iters_used,
                "| status:", status
            )

            success = True
            break

        if not success:
            print(f"FAILED: {item['instance']} could not generate a valid instance after {max_regen_tries} tries")

    save_instance_description_excel(out_dir, description_rows)

    print("\n✅ new benchmark suite generated")
    print("instances dir:", out_dir)
    print(f"quick sweet spot: blocking_ratio in [{target_low}, {target_high}]")


if __name__ == "__main__":
    main()