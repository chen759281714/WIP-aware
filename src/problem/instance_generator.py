"""
实例生成器：Flexible Multi-Stage Scheduling with Finite Buffers (WIP)
- 多工段串联（固定工艺顺序）
- 每工段并行机（阶段内柔性）
- 工段间有限缓冲（blocking/starving 由 scheduler 实现）
输出结构与 StageBufferWIPScheduler 兼容：
- operations: Dict[job, List[op_dict]]
- buffers: Dict[buffer_id, {"capacity": int}]
- os_seq: List[str]  (OS 方案A：重复job列表，保证足够长)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import random
import json
import os
import sys
import math

# ===== 把项目根目录加入 sys.path，解决 ModuleNotFoundError: No module named 'src' =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


@dataclass(frozen=True)
class InstanceSpec:
    """生成实例的配置参数（建议写进日志/文件名，保证可复现）"""
    num_stages: int                 # 工段数 S
    machines_per_stage: int         # 每工段并行机数量 K（统一）
    n_jobs: int                     # 工件数 N
    buffer_caps: List[int]          # 缓冲容量列表，长度 S-1，对应 B01, B12, ...
    pt_profile: str                 # "downstream_bottleneck" | "mid_bottleneck" | "balanced"
    pt_low: int = 1                 # 加工时间下界
    pt_high: int = 10               # 加工时间上界
    seed: int = 0                   # 随机种子
    os_repeat: int = 50             # OS 序列重复轮数（方案A）


def _validate_spec(spec: InstanceSpec) -> None:
    if spec.num_stages < 2:
        raise ValueError("num_stages 必须 >= 2")
    if spec.machines_per_stage < 1:
        raise ValueError("machines_per_stage 必须 >= 1")
    if spec.n_jobs < 1:
        raise ValueError("n_jobs 必须 >= 1")
    if len(spec.buffer_caps) != spec.num_stages - 1:
        raise ValueError("buffer_caps 长度必须等于 num_stages-1")
    if any(c < 0 for c in spec.buffer_caps):
        raise ValueError("buffer_caps 中容量必须 >= 0（0 表示无缓冲，相当于严格阻塞传递，慎用）")
    if spec.pt_low < 1 or spec.pt_high < spec.pt_low:
        raise ValueError("pt_low/pt_high 设置不合法")
    if spec.pt_profile not in {"downstream_bottleneck", "mid_bottleneck", "balanced"}:
        raise ValueError("pt_profile 只能是 downstream_bottleneck / mid_bottleneck / balanced")
    if spec.os_repeat < 1:
        raise ValueError("os_repeat 必须 >= 1")


def _stage_weight(num_stages: int, profile: str) -> List[float]:
    """
    给每个工段一个“慢/快权重”，用于控制瓶颈位置。
    返回长度 S 的权重，权重越大 => 期望加工时间越大（越慢）。
    """
    if profile == "balanced":
        return [1.0] * num_stages

    if profile == "downstream_bottleneck":
        # 越靠后越慢：线性递增
        # e.g. S=5 -> [0.8, 1.0, 1.2, 1.4, 1.6]
        base = 0.8
        step = 0.8 / max(1, num_stages - 1)
        return [base + step * i for i in range(num_stages)]

    # mid_bottleneck：中间最慢，两端更快（“山峰形”）
    mid = (num_stages - 1) / 2.0
    w = []
    for s in range(num_stages):
        # 距离中点越近权重越大
        dist = abs(s - mid)
        # dist=0 -> 1.6, dist大 -> 0.9 左右
        w.append(1.6 - 0.7 * (dist / max(1.0, mid)))
    return w


def _clip_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(x))))


def generate_fms_wip_instance(spec: InstanceSpec) -> Tuple[Dict[str, List[Dict[str, Any]]],
                                                          Dict[str, Dict[str, int]],
                                                          List[str]]:
    """
    生成一个“多工段并行机 + 有限缓冲”的实例（结构对接 scheduler）。
    返回：
      operations, buffers, os_seq
    """
    _validate_spec(spec)
    rng = random.Random(spec.seed)

    # 工件集合
    jobs = [f"J{i}" for i in range(1, spec.n_jobs + 1)]

    # 缓冲区定义：B01, B12, ...
    buffers: Dict[str, Dict[str, int]] = {}
    for i, cap in enumerate(spec.buffer_caps):
        bid = f"B{i}{i+1}"  # B01, B12...
        buffers[bid] = {"capacity": int(cap)}

    # 每工段机器集合：S0: M0a,M0b..., S1: M1a,M1b...
    stage_machines: List[List[str]] = []
    for s in range(spec.num_stages):
        ms = [f"M{s}{chr(ord('a') + k)}" for k in range(spec.machines_per_stage)]
        stage_machines.append(ms)

    # 工段权重（控制瓶颈）
    weights = _stage_weight(spec.num_stages, spec.pt_profile)

    # operations 结构
    operations: Dict[str, List[Dict[str, Any]]] = {}
    for j in jobs:
        ops_j: List[Dict[str, Any]] = []
        for s in range(spec.num_stages):
            # buffer_in/out
            buffer_in = None if s == 0 else f"B{s-1}{s}"
            buffer_out = None if s == spec.num_stages - 1 else f"B{s}{s+1}"

            # 为该工段内每台机生成加工时间
            # 先从[pt_low, pt_high]抽一个基准，再乘以工段权重，再加一点机器间扰动
            base = rng.randint(spec.pt_low, spec.pt_high)
            machines_dict: Dict[str, int] = {}
            for m in stage_machines[s]:
                # 机器扰动：±10%
                jitter = 1.0 + rng.uniform(-0.10, 0.10)
                pt = _clip_int(base * weights[s] * jitter, spec.pt_low, spec.pt_high * 3)
                machines_dict[m] = int(pt)

            ops_j.append({
                "machines": machines_dict,
                "buffer_in": buffer_in,
                "buffer_out": buffer_out,
            })
        operations[j] = ops_j

    # OS 方案A：重复 job 列表，保证长度足够（解码器只会取到所有工序排完为止）
    os_seq = jobs * int(spec.os_repeat)

    return operations, buffers, os_seq


def describe_instance(spec: InstanceSpec,
                      operations: Dict[str, List[Dict[str, Any]]],
                      buffers: Dict[str, Dict[str, int]]) -> str:
    """
    生成一个便于日志打印的摘要（非必须，但调试很方便）。
    """
    s = []
    s.append(f"InstanceSpec(num_stages={spec.num_stages}, machines_per_stage={spec.machines_per_stage}, "
             f"n_jobs={spec.n_jobs}, buffer_caps={spec.buffer_caps}, pt_profile='{spec.pt_profile}', "
             f"seed={spec.seed})")
    # 简要看每工段平均加工时间（对所有job取平均，再对机器取平均）
    stage_avg = []
    for st in range(spec.num_stages):
        vals = []
        for j, ops in operations.items():
            md = ops[st]["machines"]
            vals.extend(list(md.values()))
        stage_avg.append(sum(vals) / max(1, len(vals)))
    s.append("stage_avg_pt=" + ", ".join([f"S{idx}:{v:.2f}" for idx, v in enumerate(stage_avg)]))
    s.append("buffers=" + ", ".join([f"{bid}(cap={info['capacity']})" for bid, info in buffers.items()]))
    return "\n".join(s)

def save_instance_to_json(
    filepath: str,
    spec: InstanceSpec,
    operations: Dict[str, List[Dict[str, Any]]],
    buffers: Dict[str, Dict[str, int]],
) -> None:
    """
    保存实例到 JSON（保证可复现）。
    注意：os_seq 不保存（方案A可运行时生成），避免文件过大。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = {
        "spec": {
            "num_stages": spec.num_stages,
            "machines_per_stage": spec.machines_per_stage,
            "n_jobs": spec.n_jobs,
            "buffer_caps": spec.buffer_caps,
            "pt_profile": spec.pt_profile,
            "pt_low": spec.pt_low,
            "pt_high": spec.pt_high,
            "seed": spec.seed,
            "os_repeat": spec.os_repeat,
        },
        "buffers": buffers,
        "operations": operations,
        "meta": {
            "format": "WIP-FMS-JSON",
            "version": 1,
        }
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_instance_from_json(filepath: str) -> Tuple[InstanceSpec,
                                                   Dict[str, List[Dict[str, Any]]],
                                                   Dict[str, Dict[str, int]],
                                                   List[str]]:
    """
    从 JSON 加载实例，并自动生成 os_seq（方案A）。
    返回：spec, operations, buffers, os_seq
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    s = data["spec"]
    spec = InstanceSpec(
        num_stages=int(s["num_stages"]),
        machines_per_stage=int(s["machines_per_stage"]),
        n_jobs=int(s["n_jobs"]),
        buffer_caps=list(s["buffer_caps"]),
        pt_profile=str(s["pt_profile"]),
        pt_low=int(s.get("pt_low", 1)),
        pt_high=int(s.get("pt_high", 10)),
        seed=int(s.get("seed", 0)),
        os_repeat=int(s.get("os_repeat", 50)),
    )

    operations = data["operations"]
    buffers = data["buffers"]

    # OS 方案A：重复 job 列表
    jobs = sorted(list(operations.keys()), key=lambda x: int(x[1:]) if x.startswith("J") and x[1:].isdigit() else x)
    os_seq = jobs * int(spec.os_repeat)

    return spec, operations, buffers, os_seq

import math


def compute_stage_avg_pt(operations: Dict[str, List[Dict[str, Any]]]) -> List[float]:
    """
    计算每个工段的平均加工时间（对所有 job、所有候选机器取平均）。
    返回长度 S 的列表：avg_pt[s]
    """
    # 工段数 S
    any_job = next(iter(operations))
    S = len(operations[any_job])

    avg = []
    for s in range(S):
        vals = []
        for j, ops in operations.items():
            md = ops[s]["machines"]
            vals.extend(list(md.values()))
        avg.append(sum(vals) / max(1, len(vals)))
    return avg


def auto_buffer_caps_from_stage_avg(
    stage_avg_pt: List[float],
    machines_per_stage: int,
    alpha: float = 2.5,
    beta: float = 0.5,
    min_mult: float = 2.0,
    max_mult: float = 4.0,
) -> List[int]:
    """
    根据“sweet spot”规则自动计算每个 buffer 的容量。

    核心思想：
    - 基准容量 ~ alpha * K（K=并行机数）
    - 根据相邻工段产能比 r = avg_pt_down / avg_pt_up 做修正
      r>1：下游更慢（更易拥塞）=> cap 略增，避免长期顶满
      r<1：下游更快（更易空）  => cap 略减，让 WIP 仍能产生影响
    - 最终 cap 限制在 [min_mult*K, max_mult*K]，确保不会过紧/过松

    返回长度 S-1 的 caps，对应 B01, B12, ...
    """
    K = machines_per_stage
    caps = []
    for s in range(len(stage_avg_pt) - 1):
        up = stage_avg_pt[s]
        dn = stage_avg_pt[s + 1]
        r = dn / max(1e-9, up)

        # 系数：alpha + beta*(r-1)
        coeff = alpha + beta * (r - 1.0)
        coeff = max(min_mult, min(max_mult, coeff))  # 先把系数限制一下

        cap = int(math.ceil(coeff * K))
        cap_min = int(math.ceil(min_mult * K))
        cap_max = int(math.ceil(max_mult * K))
        cap = max(cap_min, min(cap_max, cap))

        caps.append(cap)

    return caps


def generate_fms_wip_instance_auto_caps(
    spec: InstanceSpec,
    alpha: float = 2.5,
    beta: float = 0.5,
    min_mult: float = 2.0,
    max_mult: float = 4.0,
) -> Tuple[InstanceSpec, Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, int]], List[str], List[float]]:
    """
    生成实例（operations先生成），然后根据 stage_avg_pt 自动计算 buffer_caps，
    再构建 buffers，并返回最终 spec（buffer_caps 被写入）。

    返回：
      final_spec, operations, buffers, os_seq, stage_avg_pt
    """
    # 1) 先生成 operations（暂时不依赖 buffer_caps）
    # 这里复用你原来的 generate_fms_wip_instance(spec)，但它会校验 buffer_caps 长度。
    # 所以我们要求传入的 spec.buffer_caps 只是占位，长度必须是 S-1。
    operations, _, os_seq = generate_fms_wip_instance(spec)

    # 2) 计算每工段平均加工时间
    stage_avg = compute_stage_avg_pt(operations)

    # 3) 自动算 caps
    caps = auto_buffer_caps_from_stage_avg(
        stage_avg_pt=stage_avg,
        machines_per_stage=spec.machines_per_stage,
        alpha=alpha,
        beta=beta,
        min_mult=min_mult,
        max_mult=max_mult,
    )

    # 4) 构建 buffers
    buffers = {}
    for i, cap in enumerate(caps):
        bid = f"B{i}{i+1}"  # B01, B12...
        buffers[bid] = {"capacity": int(cap)}

    # 5) 返回一个“最终 spec”（把 buffer_caps 写进去）
    final_spec = InstanceSpec(
        num_stages=spec.num_stages,
        machines_per_stage=spec.machines_per_stage,
        n_jobs=spec.n_jobs,
        buffer_caps=caps,
        pt_profile=spec.pt_profile,
        pt_low=spec.pt_low,
        pt_high=spec.pt_high,
        seed=spec.seed,
        os_repeat=spec.os_repeat,
    )

    return final_spec, operations, buffers, os_seq, stage_avg