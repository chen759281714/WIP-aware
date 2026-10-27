"""统一正式算法对比实验；分析阶段不导入优化器，也不执行解码。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==================== 用户配置区 ====================
MODE = "run"  # 可选模式：run（运行）、analyze（分析）、all（先运行再分析）
SELECTED_ALGORITHMS = ["full", "wo_mgs", "wo_hlns", "wo_ac"]
ANALYSIS_ALGORITHMS = None  # None 表示使用 SELECTED_ALGORITHMS；也可纳入此前批次已完成的算法
INSTANCE_DIR = "data/final_benchmark/instances"
SELECTED_INSTANCES = None  # 可指定不含扩展名的实例名称，或以 .json 结尾的文件名；None 表示全部实例
EXPECTED_INSTANCE_COUNT = 96
SEEDS = list(range(10, 20))
FE_MAX = 40000
# 从算法模块已冻结的默认配置中解析参数，不重复定义参数值。
PROPOSED_PARAMS = None  # None 表示沿用 get_default_full_parameters() 对应的默认配置；也可显式指定覆盖参数字典
USE_MULTIPROCESSING = True
MAX_WORKERS = 16
RESULTS_DIR = "experiments/results/formal_comparison"
SKIP_COMPLETED = True
HV_REFERENCE_POINT = (1.1, 1.1)
SAVE_CONVERGENCE = True
CONVERGENCE_INTERVAL_FE = 2000
PLOT_INSTANCES = ["WIPHFSP_S1_P1_B1"]
MAKE_FIGURES = True
# ================== 用户配置区结束 ==================

SCHEMA = "wip-formal-v1"
TOL = 1e-12  # 目标值跨度的绝对容差，两个指标使用同一规则
HASH_SEED = "0"
FIELDS = ("pm_specialized_kind", "ps_specialized_kind",
          "pm_archive_guidance_enabled", "ps_archive_guidance_enabled")
# 外部算法扩展时需注册模块、类、参数、身份和源码文件，
# 并提供满足相同运行、回调与档案接口约定的适配器。
REGISTRY = {
    "full": ("wip_graph_dual_population", "WIPGraphDualPopulation", "HLNS_FULL", ("MGS", "HLNS", True, False)),
    "wo_mgs": ("wip_graph_dual_population_wo_mgs", "WIPGraphDualPopulationWithoutMGS", "HLNS_WO_MGS", ("BasicMutation", "HLNS", True, False)),
    "wo_hlns": ("wip_graph_dual_population_wo_hlns", "WIPGraphDualPopulationWithoutHLNS", "HLNS_WO_HLNS", ("MGS", "BasicMutation", True, False)),
    "wo_ac": ("wip_graph_dual_population_wo_ac", "WIPGraphDualPopulationWithoutAC", "HLNS_WO_AC", ("MGS", "HLNS", False, False)),
}


# 启动时只读保存源码哈希，防止长期运行的工作进程将磁盘上的新哈希
# 错配给已导入的旧优化器代码；此处不包含分析代码。
STARTUP_SOURCE_HASHES = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for directory in (ROOT/"src/algorithms", ROOT/"src/solution")
    for p in directory.glob("*.py")
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    return (ROOT / path).resolve()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def dimensions(name):
    m = re.fullmatch(r"WIPHFSP_(S[1-8])_(P[1-4])_(B[1-3])", name)
    if not m:
        raise ValueError(f"Invalid instance name: {name}")
    return dict(zip(("size", "profile", "buffer"), m.groups()))


def points(values, nonnegative=False):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Empty or invalid objective archive")
    out = []
    for p in values:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("Objectives must be pairs")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in p):
            raise ValueError("Nonfinite/nonnumeric objectives")
        if nonnegative and min(p) < 0:
            raise ValueError("Negative physical objective")
        out.append(tuple(p))
    return out


def nondominated(values):
    ordered = sorted(set(points(values)))
    front, best = [], math.inf
    for x, y in ordered:
        if y < best:
            front.append((x, y))
            best = y
    return front


def compute_hv(values, reference=HV_REFERENCE_POINT):
    """计算归一化目标空间中二维矩形并集的精确面积，不再除以 1.21。"""
    values = points(values)
    rx, ry = points([reference])[0]
    eligible = [p for p in values if p[0] < rx and p[1] < ry]
    if not eligible:
        return 0.0
    area, previous = 0.0, ry
    for x, y in nondominated(eligible):
        area += (rx-x)*(previous-y)
        previous = y
    if not math.isfinite(area):
        raise ValueError("Nonfinite HV")
    return area


def compute_igd(values, reference):
    a, r = points(values), points(reference)
    result = st.mean(min(math.dist(z, p) for p in a) for z in r)
    if not math.isfinite(result):
        raise ValueError("Nonfinite IGD")
    return result


def bounds_for(values):
    a = points(values, True)
    lo = [min(p[j] for p in a) for j in range(2)]
    hi = [max(p[j] for p in a) for j in range(2)]
    return {"min": lo, "max": hi, "degenerate": [hi[j]-lo[j] <= TOL for j in range(2)],
            "absolute_span_tolerance": TOL}


def normalize(values, bounds):
    # 不裁剪归一化结果；退化维度使用分母 1，保留中间快照的目标偏移，
    # 同时使最终档案中的恒定目标值映射到零。
    return [tuple((v-bounds["min"][j]) / (1.0 if bounds["degenerate"][j]
                    else bounds["max"][j]-bounds["min"][j]) for j, v in enumerate(p))
            for p in points(values, True)]


def validate_instance(path):
    data = json.loads(path.read_text())
    operations, buffers = data["operations"], data["buffers"]
    if not isinstance(operations, dict) or not operations or not isinstance(buffers, dict):
        raise ValueError(f"Invalid instance: {path}")
    for b in buffers.values():
        if type(b.get("capacity")) is not int or b["capacity"] < 1:
            raise ValueError(f"Invalid buffer capacity: {path}")
        if not 0 <= b.get("low_wip", 0) <= b["capacity"]:
            raise ValueError(f"Invalid low_wip: {path}")
    for ops in operations.values():
        if not isinstance(ops, list) or not ops:
            raise ValueError(f"Invalid operations: {path}")
        for i, op in enumerate(ops):
            if not isinstance(op.get("machines"), dict) or not op["machines"]:
                raise ValueError(f"Missing machines: {path}")
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in op["machines"].values()):
                raise ValueError(f"Invalid processing time: {path}")
            for key in ("buffer_in", "buffer_out"):
                if key not in op or (op[key] is not None and op[key] not in buffers):
                    raise ValueError(f"Invalid {key}: {path}")
            if op["buffer_in"] != (None if i == 0 else ops[i-1]["buffer_out"]):
                raise ValueError(f"Broken buffer chain: {path}")
    return data


def unique(values, name):
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be nonempty and unique")


def checkpoints(fe, interval):
    return sorted(set(range(interval, fe+1, interval)) | {fe})


def setup():
    run_algs = list(SELECTED_ALGORITHMS)
    analysis_algs = list(ANALYSIS_ALGORITHMS if ANALYSIS_ALGORITHMS is not None else run_algs)
    for values in (run_algs, analysis_algs):
        unique(values, "algorithms")
        if any(a not in REGISTRY for a in values):
            raise ValueError("Unknown algorithm registry key")
    unique(SEEDS, "seeds")
    if any(type(s) is not int or s < 0 for s in SEEDS):
        raise ValueError("Seeds must be nonnegative integers")
    if type(MAX_WORKERS) is not int or MAX_WORKERS < 1:
        raise ValueError("MAX_WORKERS must be positive")
    if tuple(HV_REFERENCE_POINT) != (1.1, 1.1):
        raise ValueError("Formal HV reference must be (1.1,1.1)")
    directory, output = resolve(INSTANCE_DIR), resolve(RESULTS_DIR)
    allowed = ROOT / "experiments/results/formal_comparison"
    # 测试可使用仓库外的临时目录，但不得写入仓库中的历史实验结果目录。
    if output == ROOT or ROOT in output.parents:
        if output != allowed and allowed not in output.parents:
            raise ValueError("Repository output must be inside formal_comparison")
    available = sorted(directory.glob("*.json"))
    if SELECTED_INSTANCES is None:
        if len(available) != EXPECTED_INSTANCE_COUNT:
            raise ValueError(f"Expected {EXPECTED_INSTANCE_COUNT} instances, got {len(available)}")
        names = [p.stem for p in available]
        expected = {f"WIPHFSP_S{s}_P{p}_B{b}" for s,p,b in itertools.product(range(1,9),range(1,5),range(1,4))}
        if set(names) != expected:
            raise ValueError("Full benchmark must cover all 8x4x3 names")
    else:
        names = sorted(n.removesuffix(".json") for n in SELECTED_INSTANCES)
        unique(names, "instances")
    for n in names:
        dimensions(n)
        validate_instance(directory/f"{n}.json")
    # 解析分析配置时也不导入优化器，
    # 仅提取已冻结的字面量配置字典，避免重复手写参数值。
    import ast
    tree = ast.parse((ROOT/"src/algorithms/wip_graph_dual_population.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "DEFAULT_FULL_PARAMETERS" for t in n.targets))
    params = ast.literal_eval(node.value.args[0])
    params.update(PROPOSED_PARAMS or {})
    if params.get("N_A") is None:
        params["N_A"] = 2*params["N"]
    if set(params) != {"N", "N_A", "p_mut", "rho", "T_coop"}:
        raise ValueError("Unexpected proposed parameters")
    if any(type(v) is not int for v in (params["N"],params["N_A"],params["T_coop"],FE_MAX,CONVERGENCE_INTERVAL_FE)):
        raise ValueError("Integer population/FE/interval required")
    if params["N"] < 1 or params["N_A"] < 2 or params["T_coop"] < 1 or FE_MAX < 2*params["N"]:
        raise ValueError("Invalid population/FE/cooperation parameters")
    if not 0 <= params["p_mut"] <= 1 or not 0 < params["rho"] <= 1 or CONVERGENCE_INTERVAL_FE < 1:
        raise ValueError("Invalid probability or interval")
    checks = checkpoints(FE_MAX, CONVERGENCE_INTERVAL_FE) if SAVE_CONVERGENCE else []
    if any(k < 2*params["N"] or (k != FE_MAX and k % (2*params["N"])) for k in checks):
        raise ValueError("Checkpoint must coincide with a real archive update (multiple of 2*N, or final FE)")
    return {"instance_dir": str(directory), "output": str(output), "instances": names,
            "subset": SELECTED_INSTANCES is not None, "seeds": list(SEEDS), "parameters": params,
            "fe_max": FE_MAX, "checkpoints": checks, "run_algorithms": run_algs,
            "analysis_algorithms": analysis_algs, "hash_seed": HASH_SEED,
            "execution": {"multiprocessing": USE_MULTIPROCESSING, "max_workers": MAX_WORKERS},
            "registry": {a: REGISTRY[a] for a in set(run_algs+analysis_algs)}}


def tasks(config, algorithms):
    return [{"algorithm": a, "instance": n, "seed": s} for n in config["instances"]
            for a in algorithms for s in config["seeds"]]


def source_identity(task, config):
    module, cls, variant, mechanisms = config["registry"][task["algorithm"]]
    sources = {"src/algorithms/wip_graph_dual_population.py", f"src/algorithms/{module}.py"}
    sources.update(str(p.relative_to(ROOT)) for p in (ROOT/"src/solution").glob("*.py"))
    hashes = {p: file_hash(ROOT/p) for p in sorted(sources)}
    if any(STARTUP_SOURCE_HASHES.get(p) != h for p,h in hashes.items()):
        raise ValueError("Optimizer source changed after runner startup; restart before proceeding")
    return {"schema": SCHEMA, **task, "class": f"src.algorithms.{module}.{cls}",
            "variant": variant, "mechanisms": dict(zip(FIELDS, mechanisms)),
            "parameters": {**config["parameters"], "FE_max": config["fe_max"], "seed": task["seed"]},
            "checkpoints": config["checkpoints"], "hash_seed": config["hash_seed"],
            "sources": hashes,
            "instance_sha256": file_hash(Path(config["instance_dir"])/f"{task['instance']}.json")}


def raw_path(task, config):
    return Path(config["output"])/"raw"/task["instance"]/task["algorithm"]/f"seed_{task['seed']}.json"


def validate_record(record, identity):
    payload = {k:v for k,v in record.items() if k != "payload_sha256"}
    if record.get("payload_sha256") != digest(payload):
        raise ValueError("Result payload checksum mismatch")
    if record["identity"] != identity or record["fingerprint"] != digest(identity):
        raise ValueError("Task/source/parameter identity conflict; preserve old results and use a new directory")
    if type(record["actual_fe"]) is not int or record["actual_fe"] != identity["parameters"]["FE_max"]:
        raise ValueError("Actual FE mismatch")
    if record["actual_parameters"] != identity["parameters"]:
        raise ValueError("Actual algorithm parameters mismatch")
    final = points(record["final_archive"], True)
    if set(nondominated(final)) != set(final) or record["archive_size"] != len(final) or len(final) > identity["parameters"]["N_A"]:
        raise ValueError("Invalid final archive")
    if not math.isfinite(record["runtime_seconds"]) or record["runtime_seconds"] < 0 or not record["completed_utc"]:
        raise ValueError("Invalid runtime/completion metadata")
    if record["mechanisms"] != identity["mechanisms"] or record["algorithm_variant"] != identity["variant"]:
        raise ValueError("Mechanism identity mismatch")
    for key, value in identity["mechanisms"].items():
        if record["diagnostics"].get(key) != value:
            raise ValueError("Diagnostic mechanism mismatch")
    if record["diagnostics"].get("algorithm_variant") != identity["variant"]:
        raise ValueError("Diagnostic variant mismatch")
    d = record["diagnostics"]
    generated = 0
    for side in ("pm_moves", "ps_moves"):
        for counts in d[side].values():
            if any(type(counts[k]) is not int or counts[k] < 0 for k in ("generated", "accepted")) or counts["accepted"] > counts["generated"]:
                raise ValueError("Invalid move statistics")
            generated += counts["generated"]
    if generated != record["actual_fe"] - 2*identity["parameters"]["N"]:
        raise ValueError("Diagnostic evaluation count mismatch")
    if identity["algorithm"] == "wo_ac" and d["cooperation"]["pm_attempts"]:
        raise ValueError("Disabled AC used")
    if identity["algorithm"] == "wo_mgs" and any(d["pm_moves"][k]["generated"] for k in ("machine_priority", "machine_reassign")):
        raise ValueError("Disabled MGS used")
    if identity["algorithm"] == "wo_hlns" and (d["ps_moves"]["hlns_joint"]["generated"] or d["hlns"]["hlns_calls"]):
        raise ValueError("Disabled H-LNS used")
    snapshots = record["convergence"]
    if [s["fe"] for s in snapshots] != identity["checkpoints"]:
        raise ValueError("Missing/misaligned convergence checkpoints")
    for snap in snapshots:
        a = points(snap["archive"], True)
        if type(snap["fe"]) is not int or snap["archive_size"] != len(a) or set(nondominated(a)) != set(a):
            raise ValueError("Invalid convergence archive")
    if snapshots and snapshots[-1]["archive"] != record["final_archive"]:
        raise ValueError("Final checkpoint/archive mismatch")


def load_record(task, config):
    record = json.loads(raw_path(task, config).read_text())
    validate_record(record, source_identity(task, config))
    return record


def run_task(task, config):
    """工作进程将错误转为可序列化的普通数据返回，不向主进程传播自定义异常。"""
    try:
        if os.environ.get("PYTHONHASHSEED") != config["hash_seed"]:
            raise RuntimeError("Restart interpreter with PYTHONHASHSEED=0")
        path = raw_path(task, config)
        if path.exists():
            raise FileExistsError(f"Refusing overwrite: {path}")
        identity = source_identity(task, config)
        module, clsname, variant, _ = config["registry"][task["algorithm"]]
        cls = getattr(importlib.import_module("src.algorithms."+module), clsname)
        data = validate_instance(Path(config["instance_dir"])/f"{task['instance']}.json")
        kwargs = dict(identity["parameters"])
        if task["algorithm"] == "full":
            kwargs["variant"] = variant
        started = time.perf_counter()
        engine = cls(data["operations"], data["buffers"], **kwargs)
        if engine.algorithm_variant != variant or any(getattr(engine,k) != v for k,v in identity["mechanisms"].items()):
            raise ValueError("Constructed wrong mechanism/class")
        actual_parameters = {k: getattr(engine,k) for k in identity["parameters"] if k != "seed"}
        actual_parameters["seed"] = task["seed"]  # 私有随机数生成器由该构造参数初始化
        snapshots = []
        checks = set(identity["checkpoints"])
        def observe(state):
            if state.n_evaluations in checks:
                snapshots.append({"fe": state.n_evaluations, "archive_size": len(state.A),
                                  "archive": [[x.makespan,x.shortage] for x in state.A]})
        archive = engine.run(diagnostic_callback=observe if checks else None)
        if identity != source_identity(task, config):
            raise ValueError("Source changed during optimization")
        diagnostics = {k: engine.diagnostics[k] for k in ("algorithm_variant", *FIELDS,
                        "pm_moves", "ps_moves", "cooperation", "shortage_diagnosis", "source_generated")}
        diagnostics["hlns"] = engine.hlns_statistics
        record = {"identity": identity, "fingerprint": digest(identity), "actual_fe": engine.n_evaluations,
                  "algorithm_variant": engine.algorithm_variant, "mechanisms": identity["mechanisms"],
                  "actual_parameters": actual_parameters,
                  "runtime_seconds": time.perf_counter()-started, "completed_utc": utc(),
                  "final_archive": [[x.makespan,x.shortage] for x in archive], "archive_size": len(archive),
                  "convergence": snapshots, "diagnostics": diagnostics,
                  "runtime_environment": {"python": sys.version, "platform": sys.platform},
                  "runner_source_sha256": file_hash(__file__)}
        record["payload_sha256"] = digest(record)
        validate_record(record, identity)
        atomic_json(path, record)
        return {**task, "status": "completed"}
    except Exception:
        return {**task, "status": "failed", "error": traceback.format_exc()}


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root/".session.lock"
    try:
        with lock.open("x") as f:
            import socket
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started_utc": utc()}, f)
    except FileExistsError:
        raise RuntimeError(f"Directory locked: {lock}. Read its PID/host and verify old processes exited before moving the lock.")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_mode(config):
    root = Path(config["output"])
    with run_lock(root):
        batch = tasks(config, config["run_algorithms"])
        pending, statuses = [], []
        # 启动任何优化任务前，先检查全部已有结果文件。
        for task in batch:
            if raw_path(task, config).exists():
                load_record(task, config)
                if not SKIP_COMPLETED:
                    raise FileExistsError("SKIP_COMPLETED=False cannot overwrite existing data; choose a new output directory")
                statuses.append({**task, "status": "skipped"})
            else:
                pending.append(task)
        # 各批次配置仅追加保存，config.json 则记录最近一次批次的配置。
        atomic_json(root/"config.json", config)
        atomic_json(root/"configs"/f"{digest(config)}.json", config)
        def progress():
            counts = Counter(x["status"] for x in statuses)
            state = {"updated_utc": utc(), "expected": len(batch), "remaining": len(batch)-len(statuses),
                     "counts": dict(counts), "tasks": statuses, "batch": config}
            atomic_json(root/"manifest.json", state)
            print(f"Completed {len(statuses)}/{len(batch)} | completed={counts['completed']} "
                  f"skipped={counts['skipped']} failed={counts['failed']} remaining={state['remaining']}", flush=True)
        print(f"{'SUBSET' if config['subset'] else 'FULL 96-instance'} batch: {len(batch)} tasks, "
              f"planned FE={len(batch)*config['fe_max']}", flush=True)
        progress()
        if USE_MULTIPROCESSING and pending:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=get_context("spawn")) as pool:
                futures = {pool.submit(run_task,t,config): t for t in pending}
                try:
                    for f in as_completed(futures):
                        statuses.append(f.result())
                        progress()
                except BaseException:
                    for f in futures:
                        f.cancel()
                    atomic_json(root/"pool_failure.json", {"error": traceback.format_exc(), "utc": utc()})
                    raise
        else:
            for task in pending:
                statuses.append(run_task(task,config))
                progress()
        atomic_json(root/"manifests"/f"{time.time_ns()}.json", {"batch": config, "tasks": statuses})
        if any(s["status"] == "failed" for s in statuses):
            raise RuntimeError("Tasks failed; see manifest.json. Valid results preserved.")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values):
    return st.stdev(values) if len(values) > 1 else None


def summarize(rows, keys, metrics, level):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        row = dict(zip(keys,key))
        row.update(n=len(group), sd_level=level)
        for metric in metrics:
            values = [r[metric] for r in group]
            row.update({f"mean_{metric}": st.mean(values), f"sample_sd_{metric}": sample_sd(values),
                        f"median_{metric}": st.median(values)})
        output.append(row)
    return output


def analyze_mode(config):
    records, invalid = [], []
    for task in tasks(config, config["analysis_algorithms"]):
        try:
            records.append((task, load_record(task, config)))
        except Exception as exc:
            invalid.append({**task,"error": str(exc)})
    if invalid:
        # 数据不完整时，不输出部分任务的正式评价指标。
        atomic_json(Path(config["output"])/"analysis_validation_errors.json", {"invalid_or_missing": invalid})
        raise ValueError(f"Analysis incomplete: {len(invalid)} tasks. See analysis_validation_errors.json")
    metadata = {"schema": SCHEMA, "algorithms": config["analysis_algorithms"],
                "instances": config["instances"], "seeds": config["seeds"], "subset": config["subset"],
                "run_fingerprints": [r["fingerprint"] for _,r in records],
                "raw_content_hashes": [file_hash(raw_path(t,config)) for t,_ in records],
                "reference_point": HV_REFERENCE_POINT, "tolerance": TOL,
                "degenerate_rule": "unit denominator; no clipping", "analysis_source": file_hash(__file__),
                "igd_reference": "empirical union nondominated final archives, not true optimal front"}
    evaluation_id = digest(metadata)[:20]
    folder = Path(config["output"])/"analysis"/evaluation_id
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/"evaluation_config.json", metadata)
    bounds, references, rows, convergence = {}, {}, [], []
    for name in config["instances"]:
        subset = [(t,r) for t,r in records if t["instance"] == name]
        union = [p for _,r in subset for p in r["final_archive"]]
        b = bounds_for(union)
        ref = nondominated(union)
        nr = normalize(ref,b)
        bounds[name] = b
        references[name] = {"raw": ref,"normalized": nr}
        for task, record in subset:
            a = record["final_archive"]
            norm = normalize(a,b)
            rows.append({**task, **dimensions(name), "hv": compute_hv(norm), "igd": compute_igd(norm,nr),
                         "best_makespan": min(p[0] for p in a), "best_shortage": min(p[1] for p in a),
                         "archive_size": len(a), "actual_fe": record["actual_fe"], "runtime_seconds": record["runtime_seconds"]})
            for snap in record["convergence"]:
                norm = normalize(snap["archive"],b)
                convergence.append({**task,"fe": snap["fe"], "hv": compute_hv(norm), "igd": compute_igd(norm,nr),
                                    "archive_size": snap["archive_size"],
                                    "outside_unit_points": sum(any(v < 0 or v > 1 for v in p) for p in norm),
                                    "outside_hv_reference_points": sum(any(v > 1.1 for v in p) for p in norm)})
    instance = summarize(rows,["instance","algorithm"],["hv","igd"],"seeds within instance")
    means = [{"algorithm": r["algorithm"],"instance": r["instance"],**dimensions(r["instance"]),
              "hv": r["mean_hv"],"igd": r["mean_igd"]} for r in instance]
    overall = summarize(means,["algorithm"],["hv","igd"],"instance means")
    groups = []
    for dim in ("size","profile","buffer"):
        for row in summarize(means,["algorithm",dim],["hv","igd"],"instance means"):
            value = row.pop(dim)
            groups.append({"dimension": dim,"group":value,**row})
    cs = summarize(convergence,["instance","algorithm","fe"],["hv","igd"],"seeds within instance")
    atomic_json(folder/"normalization_bounds.json", bounds)
    atomic_json(folder/"reference_fronts.json", references)
    for filename,data in (("run_metrics",rows),("instance_summary",instance),("overall_summary",overall),
                          ("group_summary",groups),("convergence_runs",convergence),("convergence_summary",cs)):
        write_csv(folder/f"{filename}.csv",data)
    if MAKE_FIGURES:
        plot_results(config,evaluation_id,records,rows,cs,overall)
    atomic_json(Path(config["output"])/"analysis"/"latest.json", {"evaluation_id": evaluation_id,"directory":str(folder)})
    print(f"Analysis ({'SUBSET' if config['subset'] else '96 instances'}) saved: {folder}")
    return folder


def plot_results(config, evaluation_id, records, rows, convergence, overall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(config["output"])/"figures"/evaluation_id
    for sub in ("comparisons","pareto_fronts","convergence"):
        (folder/sub).mkdir(parents=True,exist_ok=True)
    def save(fig,path):
        fig.tight_layout()
        fig.savefig(path.with_suffix(".png"),dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    for metric in ("hv","igd"):
        fig,ax = plt.subplots()
        ax.bar([r["algorithm"] for r in overall],[r["mean_"+metric] for r in overall])
        ax.set_ylabel(metric.upper()+" (equal instance weight)")
        save(fig,folder/"comparisons"/metric)
    for name in PLOT_INSTANCES:
        if name not in config["instances"]:
            continue
        fig,ax = plt.subplots()
        for alg in config["analysis_algorithms"]:
            a = nondominated([p for t,r in records if t["instance"]==name and t["algorithm"]==alg for p in r["final_archive"]])
            ax.plot([p[0] for p in a],[p[1] for p in a],"o-",label=alg)
        ax.set(xlabel="Makespan",ylabel="Shortage",title=name+" (union of seeds)")
        ax.legend()
        save(fig,folder/"pareto_fronts"/name)
        for metric in ("hv","igd"):
            fig,ax = plt.subplots()
            for alg in config["analysis_algorithms"]:
                a = sorted((r for r in convergence if r["instance"]==name and r["algorithm"]==alg),key=lambda r:r["fe"])
                if not a:
                    continue
                x=[r["fe"] for r in a]; y=[r["mean_"+metric] for r in a]
                sd=[r["sample_sd_"+metric] or 0 for r in a]
                ax.plot(x,y,label=alg)
                ax.fill_between(x,[v-s for v,s in zip(y,sd)],[v+s for v,s in zip(y,sd)],alpha=.15)
            ax.set(xlabel="FE",ylabel=metric.upper(),title=name+" (mean ± sample SD)")
            if ax.lines:
                ax.legend()
            save(fig,folder/"convergence"/f"{name}_{metric}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("run","analyze","all"),default=MODE)
    args = parser.parse_args()
    if args.mode != "analyze" and os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        env = dict(os.environ,PYTHONHASHSEED=HASH_SEED)
        return subprocess.call([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env=env)
    config = setup()
    if args.mode in ("run","all"):
        run_mode(config)
    if args.mode in ("analyze","all"):
        analyze_mode(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
