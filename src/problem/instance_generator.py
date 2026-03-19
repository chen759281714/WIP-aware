"""
实例生成器：Flexible Multi-Stage Scheduling with Finite Buffers (WIP)
- 多工段串联（固定工艺顺序）
- 每工段并行机（阶段内柔性）
- 工段间有限缓冲（blocking/starving 由 scheduler 实现）

新增功能（完全向后兼容）：
- stage_weights_override：允许外部指定工段权重
- heterogeneity_level：控制机器异构程度（low / medium / high）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import random
import json
import os
import sys
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


@dataclass(frozen=True)
class InstanceSpec:
    num_stages: int
    machines_per_stage: int
    n_jobs: int
    buffer_caps: List[int]
    pt_profile: str
    pt_low: int = 1
    pt_high: int = 10
    seed: int = 0
    os_repeat: int = 50


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
        raise ValueError("buffer_caps 中容量必须 >= 0")
    if spec.pt_low < 1 or spec.pt_high < spec.pt_low:
        raise ValueError("pt_low/pt_high 设置不合法")
    if spec.pt_profile not in {"downstream_bottleneck", "mid_bottleneck", "balanced"}:
        raise ValueError("pt_profile 不合法")
    if spec.os_repeat < 1:
        raise ValueError("os_repeat 必须 >= 1")


def _stage_weight(num_stages: int, profile: str) -> List[float]:

    if profile == "balanced":
        return [1.0] * num_stages

    if profile == "downstream_bottleneck":

        if num_stages == 2:
            return [0.7, 1.8]

        vals = []
        for s in range(num_stages):
            ratio = s / (num_stages - 1)
            vals.append(0.6 + 1.4 * ratio)

        return vals

    mid = (num_stages - 1) / 2.0

    vals = []

    for s in range(num_stages):

        dist = abs(s - mid) / max(1.0, mid)

        vals.append(2.0 - 1.3 * dist)

    return vals

def build_stage_weights(profile: str, strength: str, num_stages: int) -> List[float]:
    """
    根据 profile + strength 显式生成工段权重。
    供 generate_suite.py 调用。

    profile:
        - balanced
        - mid_bottleneck
        - downstream_bottleneck

    strength:
        - mild
        - moderate
        - strong
    """
    if profile == "balanced":
        if strength == "mild":
            return [1.0] * num_stages
        elif strength == "moderate":
            return [1.0 + 0.05 * ((i % 2) * 2 - 1) for i in range(num_stages)]
        else:
            return [1.0 + 0.10 * ((i % 2) * 2 - 1) for i in range(num_stages)]

    if profile == "downstream_bottleneck":
        if num_stages == 2:
            if strength == "mild":
                return [0.9, 1.4]
            elif strength == "moderate":
                return [0.7, 1.8]
            else:
                return [0.5, 2.2]

        vals = []
        if strength == "mild":
            lo, hi = 0.8, 1.6
        elif strength == "moderate":
            lo, hi = 0.6, 2.0
        else:
            lo, hi = 0.4, 2.4

        for s in range(num_stages):
            ratio = s / (num_stages - 1)
            vals.append(lo + (hi - lo) * ratio)
        return vals

    # mid_bottleneck
    mid = (num_stages - 1) / 2.0
    vals = []

    if strength == "mild":
        peak = 1.6
        edge = 0.9
    elif strength == "moderate":
        peak = 2.0
        edge = 0.7
    else:
        peak = 2.5
        edge = 0.5

    for s in range(num_stages):
        dist = abs(s - mid) / max(1.0, mid)
        vals.append(peak - (peak - edge) * dist)

    return vals


def _clip_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(x))))


def _machine_multipliers(k: int) -> List[float]:

    if k == 1:
        return [1.0]

    if k == 2:
        return [0.85, 1.25]

    if k == 3:
        return [0.75, 1.00, 1.35]

    if k == 4:
        return [0.70, 0.95, 1.20, 1.50]

    base = [0.7, 0.9, 1.1, 1.3, 1.5]

    if k <= len(base):
        return base[:k]

    step = (1.6 - 0.7) / (k - 1)

    return [0.7 + i * step for i in range(k)]


def _apply_heterogeneity(base_mult: List[float], level: str) -> List[float]:

    if level == "low":
        return [1 + (m - 1) * 0.4 for m in base_mult]

    if level == "high":
        return [1 + (m - 1) * 1.8 for m in base_mult]

    return base_mult


def generate_fms_wip_instance(
    spec: InstanceSpec,
    stage_weights_override: Optional[List[float]] = None,
    heterogeneity_level: str = "medium"
) -> Tuple[Dict[str, List[Dict[str, Any]]],
           Dict[str, Dict[str, int]],
           List[str]]:

    _validate_spec(spec)

    rng = random.Random(spec.seed)

    jobs = [f"J{i}" for i in range(1, spec.n_jobs + 1)]

    buffers: Dict[str, Dict[str, int]] = {}

    for i, cap in enumerate(spec.buffer_caps):
        bid = f"B{i}{i+1}"
        buffers[bid] = {"capacity": int(cap)}

    stage_machines: List[List[str]] = []

    for s in range(spec.num_stages):
        ms = [f"M{s}{chr(ord('a') + k)}" for k in range(spec.machines_per_stage)]
        stage_machines.append(ms)

    if stage_weights_override is not None:
        weights = stage_weights_override
    else:
        weights = build_stage_weights(spec.pt_profile, "moderate", spec.num_stages)

    operations: Dict[str, List[Dict[str, Any]]] = {}

    for j in jobs:

        ops_j: List[Dict[str, Any]] = []

        for s in range(spec.num_stages):

            buffer_in = None if s == 0 else f"B{s-1}{s}"
            buffer_out = None if s == spec.num_stages - 1 else f"B{s}{s+1}"

            base = rng.randint(spec.pt_low, spec.pt_high)

            machines_dict: Dict[str, int] = {}

            base_mult = _machine_multipliers(spec.machines_per_stage)

            multipliers = _apply_heterogeneity(base_mult, heterogeneity_level)

            for m, mult in zip(stage_machines[s], multipliers):

                jitter = 1.0 + rng.uniform(-0.05, 0.05)

                pt = _clip_int(
                    base * weights[s] * mult * jitter,
                    spec.pt_low,
                    spec.pt_high * 4
                )

                machines_dict[m] = int(pt)

            ops_j.append({
                "machines": machines_dict,
                "buffer_in": buffer_in,
                "buffer_out": buffer_out,
            })

        operations[j] = ops_j

    os_seq = jobs * int(spec.os_repeat)

    return operations, buffers, os_seq


def compute_stage_avg_pt(operations: Dict[str, List[Dict[str, Any]]]) -> List[float]:

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

    K = machines_per_stage

    caps = []

    for s in range(len(stage_avg_pt) - 1):

        up = stage_avg_pt[s]

        dn = stage_avg_pt[s + 1]

        r = dn / max(1e-9, up)

        coeff = alpha + beta * (r - 1.0)

        coeff = max(min_mult, min(max_mult, coeff))

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
    stage_weights_override=None,
    heterogeneity_level="medium"
) -> Tuple[InstanceSpec,
           Dict[str, List[Dict[str, Any]]],
           Dict[str, Dict[str, int]],
           List[str],
           List[float]]:

    operations, _, os_seq = generate_fms_wip_instance(
        spec,
        stage_weights_override=stage_weights_override,
        heterogeneity_level=heterogeneity_level
    )

    stage_avg = compute_stage_avg_pt(operations)

    caps = auto_buffer_caps_from_stage_avg(
        stage_avg,
        spec.machines_per_stage,
        alpha,
        beta,
        min_mult,
        max_mult,
    )

    buffers = {}

    for i, cap in enumerate(caps):
        bid = f"B{i}{i+1}"
        buffers[bid] = {"capacity": int(cap)}

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


def describe_instance(spec: InstanceSpec,
                      operations: Dict[str, List[Dict[str, Any]]],
                      buffers: Dict[str, Dict[str, int]]) -> str:

    s = []

    s.append(f"InstanceSpec(num_stages={spec.num_stages}, machines_per_stage={spec.machines_per_stage}, "
             f"n_jobs={spec.n_jobs}, buffer_caps={spec.buffer_caps}, pt_profile='{spec.pt_profile}', "
             f"seed={spec.seed})")

    stage_avg = compute_stage_avg_pt(operations)

    s.append("stage_avg_pt=" + ", ".join([f"S{idx}:{v:.2f}" for idx, v in enumerate(stage_avg)]))

    s.append("buffers=" + ", ".join([f"{bid}(cap={info['capacity']})" for bid, info in buffers.items()]))

    return "\n".join(s)


def save_instance_to_json(
    filepath: str,
    spec: InstanceSpec,
    operations: Dict[str, List[Dict[str, Any]]],
    buffers: Dict[str, Dict[str, int]],
) -> None:

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

    jobs = sorted(list(operations.keys()), key=lambda x: int(x[1:]) if x.startswith("J") and x[1:].isdigit() else x)

    os_seq = jobs * int(spec.os_repeat)

    return spec, operations, buffers, os_seq