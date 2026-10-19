"""Frozen event-time derivative and computational timing-graph validation.

Run with PYTHONHASHSEED=0 python3 -m experiments.diagnostics.shortage_sensitivity.
All decoding is offline. Integer time-grid refinement is an explicitly tested
continuous extension, not a derivative of the literal int(p) staircase.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from experiments import run_pre_ablation as r
from experiments import sis_active_lever as active
from experiments import sis_local_viability as local
from experiments import sis_role_direction as role
from experiments import sis_selection_aware as selection
from experiments.validate_shortage_objective import average_ranks
from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation
from src.solution.decoder import StageBufferWIPScheduler

ROOT = r.PROJECT_ROOT / "experiments/results/shortage_sensitivity_validation"
SCALE = 1000
EPSILON = 1 / SCALE
TOL = 1e-5
OPS_PER_STATE = 8
REGISTRATION = {
    "scale": SCALE, "epsilon": EPSILON, "random_ops_per_state": OPS_PER_STATE,
    "direct_tolerance": TOL, "g_case_1_and_2": "Both unit deficit increments have g=-1, not different derivatives",
    "integer_semantics": "Integer grid refinement; unmodified decoder; not FD of literal int(p) on unscaled floats",
    "boundary": "Last take moves active_end: g=deficit_before. active_start is a fixed instance boundary for timeline FD.",
    "parameter_boundary": "Exclude p inputs on any fastest-route active_start minimum: Lambda_C alone omits that parameter path",
    "smooth_path": "Exclude every root reaching a tied/unknown timing dependency or nonsmooth objective event, including potential tied edges",
    "gate_A": "Every smooth FD matches within atol=1e-5 and rtol=1e-7; any failure STOP",
    "gate_B": "Nonzero-case sign accuracy > .5 and Spearman > 0 in both instances; >=4/6 groups positive; propagation-only nonzero FD correct > .5; exact forward/reverse",
    "gate_3A": "Topology-preserved smooth nonzero-child sign accuracy > .5 and Spearman > 0 in both instances; no eligible sample => NOT TESTED, STOP",
    "gate_C": "Graph oracle beats root-direct on COMMON actual-nonzero sign and Spearman, top-quartile lift >1; within-state mean rank >0 in both instances and >=4/6 groups; global graph metrics also positive in both instances and >=4/6 groups",
    "gate_D": "LambdaECT strictly improves either top-1 improvement or within-group rank vs ECT-only in both instances, >=4/6 groups nonnegative; no eligible comparisons => NOT TESTED",
    "small_sample": "Missing/constant/nonzero-empty primary data is insufficient evidence, not fabricated zero correlation",
    "mild_topology_change": "<=5% changed stable graph/sequence/order tokens, descriptive only",
    "calibration_bins": "Strong: abs(prediction)>=1% S_seed; mild: tolerance<abs(prediction)<1% S_seed; near-zero abs(prediction)<=tolerance",
    "benefit_sign": "positive=improvement; B_actual=S_seed-S_child",
}


def identity(state):
    return {k: state[k] for k in ("instance", "seed", "fe_index")}


def state_path(state):
    return ROOT / "cache" / state["instance"] / f"seed_{state['seed']}" / f"fe_{state['fe_index']}.json"


def sign(value):
    return 1 if value > TOL else -1 if value < -TOL else 0


def corr(x, y, rank=True):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    if rank:
        x, y = average_ranks(x), average_ranks(y)
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def metric_pairs(rows, pred="prediction", actual="actual"):
    x, y = [z[pred] for z in rows], [z[actual] for z in rows]
    nonzero = [(a, b) for a, b in zip(x, y) if sign(a) or sign(b)]
    actual_nonzero = [(a, b) for a, b in zip(x, y) if sign(b)]
    zeros = [b for a, b in zip(x, y) if not sign(a)]
    return {
        "cases": len(rows), "nonzero_cases": len(nonzero),
        "sign_accuracy": statistics.mean(sign(a) == sign(b) for a, b in zip(x, y)) if rows else None,
        "nonzero_sign_accuracy": statistics.mean(sign(a) == sign(b) for a, b in nonzero) if nonzero else None,
        "actual_nonzero_sign_accuracy": statistics.mean(sign(a) == sign(b) for a, b in actual_nonzero) if actual_nonzero else None,
        "spearman": corr(x, y), "pearson_secondary": corr(x, y, False),
        "mean_absolute_error": statistics.mean(abs(a - b) for a, b in zip(x, y)) if rows else None,
        "normalized_absolute_error": statistics.mean(abs(a-b) / max(1, abs(b)) for a, b in zip(x, y)) if rows else None,
        "max_error": max((abs(a-b) for a, b in zip(x, y)), default=None),
        "zero_prediction_specificity": statistics.mean(not sign(b) for b in zeros) if zeros else None,
    }


def timeline_area(events, low, start, end):
    if end <= start:
        return 0.0
    ordered = sorted(events, key=lambda e: (e["time"], e["id"]))
    level, left, result = 0, 0.0, 0.0
    for event in ordered:
        right = event["time"]
        result += max(0, low-level) * max(0, min(end, right)-max(start, left))
        level, left = event["after"], right
    return result + max(0, low-level) * max(0, end-max(start, left))


def objective(timelines, windows, offset_node=None, offset=0):
    result = 0.0
    for bid, events in timelines.items():
        copied = [{**e, "time": e["time"] + (offset if e["cause"] == offset_node else 0)} for e in events]
        window = windows[bid]
        takes = [e["time"] for e in copied if e["action"] == "take"]
        end = max(takes) if takes else window["start"]
        result += timeline_area(copied, window["low"], window["start"], end)
    return result


def event_geometry(timelines, windows):
    g, nonsmooth, rows = defaultdict(float), defaultdict(set), []
    for bid, events in timelines.items():
        window = windows[bid]
        start, end, low = window["start"], window["end"], window["low"]
        changing = [e for e in events if e["action"] != "init" and e["before"] != e["after"]]
        # Even a surplus-level transition with g=0 can exchange logical order
        # with a shortage-changing event when only one timestamp is perturbed.
        time_counts = Counter(e["time"] for e in changing)
        for e in changing:
            node, time = e["cause"], e["time"]
            if node is None:
                raise AssertionError("State-changing event has no factual cause")
            before, after = max(0, low-e["before"]), max(0, low-e["after"])
            relevant = start <= time <= end and end > start
            correction = after if e["action"] == "take" and time == end else 0
            contribution = before - after + correction if relevant else 0.0
            reasons = []
            if relevant and time_counts[time] > 1:
                reasons.append("same_time_buffer_state_changes")
            if relevant and time == start:
                reasons.append("fixed_active_start_boundary")
            for reason in reasons:
                nonsmooth[node].add(reason)
            g[node] += contribution
            rows.append({"buffer_id": bid, "event_id": e["id"], "node": node,
                         "event_time": time, "action": e["action"], "deficit_before": before,
                         "deficit_after": after, "raw_g": before-after,
                         "moving_end_correction": correction if relevant else 0,
                         "g_buffer": contribution, "relevant": relevant,
                         "nonsmooth": bool(reasons), "reasons": reasons})
    return dict(g), nonsmooth, rows


def choose_binding(sources, actual_time):
    """Merge aliases of the SAME variable before detecting a max tie."""
    grouped = defaultdict(list)
    for node, time, label in sources:
        grouped[(node, time)].append(label)
    maximum = max((time for node, time in grouped), default=0)
    winners = {node for node, time in grouped if time == maximum and node is not None}
    unknown = abs(maximum - actual_time) > 1e-9
    tied = len(winners) > 1
    edges = []
    for (node, time), labels in grouped.items():
        if node is not None:
            binding = time == maximum
            edges.append({"src": node, "alpha": int(binding and not tied and not unknown),
                          "kind": "start_" + "_".join(sorted(labels)), "binding": binding,
                          "potential": binding, "tie": tied and binding})
    return edges, tied, unknown


def release_dependency(complete, release, unblocking):
    if release > complete:
        return "strict_blocked", [("completion", 0), ("unblocking", 1)], unblocking is None
    if unblocking is not None:
        return "completion_unblocking_tie", [("completion", 0), ("unblocking", 0)], True
    return "unblocked", [("completion", 1)], False


def graph_from_decode(decoded, operations, buffers, propagate=False):
    prov = decoded["provenance"]
    nodes = {i: {"id": i, "time": ev.time, "logical_id": ev.event_id,
                 "uid": [ev.job, ev.op_idx, ev.kind], "kind": ev.kind}
             for i, ev in prov.events.items()}
    timelines = {bid: [{"id": e.event_id, "time": e.time, "action": e.action,
                       "job": e.job, "before": e.level_before, "after": e.level_after,
                       "cause": e.cause_event_id} for e in events] for bid, events in prov.buffer_events.items()}
    stats = decoded["stats"]["shortage"]
    windows = {bid: {"start": stats["per_buffer_active_start"][bid],
                     "end": stats["per_buffer_active_end"][bid],
                     "low": stats["per_buffer_low_wip"][bid]} for bid in buffers}
    g, nonsmooth, event_rows = event_geometry(timelines, windows)
    edges, release_types = [], {}
    records = {(z["job"], z["op"]): z for z in decoded["schedule"]}
    by_machine = defaultdict(list)
    for key, rec in records.items():
        by_machine[rec["machine"]].append(key)
    predecessor = {}
    sequences = {}
    for machine, keys in by_machine.items():
        keys.sort(key=lambda k: (records[k]["start"], prov.op_events[k]["start"]))
        sequences[machine] = [list(k) for k in keys]
        predecessor.update({b: a for a, b in zip(keys, keys[1:])})
    def edge(src, dst, alpha, kind, potential=None, tie=False):
        if src is not None:
            edges.append({"src": src, "dst": dst, "alpha": alpha, "kind": kind,
                          "potential": bool(alpha) if potential is None else potential, "tie": tie})
    unblocking_by_release = defaultdict(list)
    for e in prov.active_dependencies:
        if e.kind == "unblocking":
            unblocking_by_release[e.dst_event_id].append(e.src_event_id)
    for key, rec in records.items():
        ids = prov.op_events[key]
        s, c, rel = (ids[k] for k in ("start", "complete", "release"))
        edge(s, c, 1, "processing")
        triggers = list(dict.fromkeys(unblocking_by_release.get(rel, [])))
        trigger = triggers[0] if len(triggers) == 1 else None
        kind, relation, bad = release_dependency(rec["end"], rec["release"], trigger)
        release_types[str(rel)] = {"type": kind, "trigger": trigger}
        if bad or len(triggers) > 1:
            nonsmooth[rel].add(kind if bad else "multiple_unblocking_triggers")
        for label, alpha in relation:
            src = c if label == "completion" else trigger
            edge(src, rel, alpha, "release_" + label,
                 potential=bool(alpha) or kind == "completion_unblocking_tie",
                 tie=kind == "completion_unblocking_tie")
        sources = []
        if key[1]:
            previous = prov.op_events[(key[0], key[1]-1)]["release"]
            sources.append((previous, nodes[previous]["time"], "job"))
        bid = operations[key[0]][key[1]].get("buffer_in")
        if bid is not None:
            take = next(e for e in timelines[bid] if e["action"] == "take" and e["cause"] == s)
            puts = [e for e in timelines[bid] if e["action"] == "put" and e["job"] == key[0] and e["id"] < take["id"]]
            if puts:
                supply = max(puts, key=lambda e: e["id"])["cause"]
                sources.append((supply, nodes[supply]["time"], "wip"))
            elif key[0] in (buffers[bid].get("init_content") or []):
                sources.append((None, 0, "initial_wip"))
            else:
                nonsmooth[s].add("missing_exact_wip_source")
        if key in predecessor:
            previous = prov.op_events[predecessor[key]]["release"]
            sources.append((previous, nodes[previous]["time"], "machine"))
        binding, tied, unknown = choose_binding(sources, rec["start"])
        if tied:
            nonsmooth[s].add("distinct_max_readiness_tie")
        if unknown:
            nonsmooth[s].add("start_not_equal_readiness_max")
        for entry in binding:
            edge(entry["src"], s, entry["alpha"], entry["kind"], entry["potential"], entry["tie"])
    order = sorted(nodes, key=lambda i: (nodes[i]["time"], nodes[i]["logical_id"]))
    position = {node: i for i, node in enumerate(order)}
    if any(position[e["src"]] >= position[e["dst"]] for e in edges if e["potential"]):
        raise AssertionError("Timing graph violates (time, logical_id) DAG")
    graph = {"nodes": list(nodes.values()), "order": order, "edges": edges, "g": g,
             "nonsmooth": {i: sorted(v) for i, v in nonsmooth.items()},
             "timelines": timelines, "windows": windows, "event_rows": event_rows,
             "machine_sequences": sequences, "release_types": release_types,
             "op_nodes": [[list(k), dict(ids)] for k, ids in prov.op_events.items()]}
    if propagate:
        graph["lambda"] = reverse(graph)
    graph["topology"] = topology_tokens(graph)
    return graph


def reverse(graph):
    values = {node: graph["g"].get(node, 0.0) for node in graph["order"]}
    outgoing = defaultdict(list)
    for e in graph["edges"]:
        if e["alpha"]:
            outgoing[e["src"]].append(e)
    for node in reversed(graph["order"]):
        values[node] += sum(e["alpha"] * values[e["dst"]] for e in outgoing[node])
    return values


def forward(graph, roots):
    incoming = defaultdict(list)
    for e in graph["edges"]:
        if e["alpha"]:
            incoming[e["dst"]].append(e)
    values = {}
    for node in graph["order"]:
        values[node] = roots.get(node, 0) + sum(e["alpha"] * values[e["src"]] for e in incoming[node])
    return values


def invalid_path(graph, root):
    if "_valid_paths" in graph:
        return not graph["_valid_paths"][root]
    adjacency = defaultdict(list)
    for e in graph["edges"]:
        if e["potential"]:
            adjacency[e["src"]].append(e["dst"])
    valid = {}
    for node in reversed(graph["order"]):
        valid[node] = node not in graph["nonsmooth"] and all(valid[child] for child in adjacency[node])
    graph["_valid_paths"] = valid
    return not valid[root]


def topology_tokens(graph):
    uid = {n["id"]: tuple(n["uid"]) for n in graph["nodes"]}
    tokens = set()
    for m, keys in graph["machine_sequences"].items():
        tokens.add(json.dumps(["machine", m, keys], separators=(",", ":")))
    for e in graph["edges"]:
        tokens.add(json.dumps(["edge", uid[e["src"]], uid[e["dst"]], e["kind"], e["alpha"], e["potential"], e["tie"]], separators=(",", ":")))
    for node, definition in graph["release_types"].items():
        tokens.add(json.dumps(["release", uid[int(node)], definition["type"], uid.get(definition["trigger"])], separators=(",", ":")))
    for bid, events in graph["timelines"].items():
        changes = [e for e in sorted(events, key=lambda e: (e["time"], e["id"])) if e["action"] != "init"]
        tokens.add(json.dumps(["buffer_order", bid, [[uid[e["cause"]], e["action"], e["before"], e["after"]] for e in changes]], separators=(",", ":")))
    return sorted(tokens)


def restore_graph(graph):
    for field in ("g", "lambda", "nonsmooth", "_valid_paths"):
        if field in graph:
            graph[field] = {int(k): value for k, value in graph[field].items()}
    return graph


def scaled_operations(operations, key=None, machine=None, step=0):
    result = copy.deepcopy(operations)
    for job, ops in result.items():
        for index, operation in enumerate(ops):
            for m, duration in operation["machines"].items():
                if int(duration) != duration:
                    raise AssertionError("Frozen inputs are not integer durations")
                operation["machines"][m] = int(duration) * SCALE + (step if (job, index) == key and m == machine else 0)
    return result


def normalized_decode(operations, buffers, ind, scaled=False):
    decoded = role.decode_events(StageBufferWIPScheduler(operations, buffers), ind)
    if not scaled:
        return decoded
    decoded["makespan"] /= SCALE
    decoded["shortage"] /= SCALE
    decoded["buffer_shortage"] = {b: v / SCALE for b, v in decoded["buffer_shortage"].items()}
    for rec in decoded["schedule"]:
        for k in ("start", "end", "release"):
            rec[k] /= SCALE
    decoded["trace"] = {b: [(t / SCALE, level, action, job) for t, level, action, job in es] for b, es in decoded["trace"].items()}
    prov = decoded["provenance"]
    prov.events = {i: dataclasses.replace(e, time=e.time / SCALE) for i, e in prov.events.items()}
    prov.buffer_events = {b: [dataclasses.replace(e, time=e.time / SCALE) for e in es] for b, es in prov.buffer_events.items()}
    stats = decoded["stats"]["shortage"]
    stats["total_shortage_area"] /= SCALE
    for k in ("per_buffer_shortage_area", "per_buffer_active_start", "per_buffer_active_end", "per_buffer_active_horizon"):
        stats[k] = {b: v / SCALE for b, v in stats[k].items()}
    return decoded


def parameter_boundary_keys(operations, buffers):
    excluded = set()
    for bid, buffer in buffers.items():
        if buffer.get("init_content"):
            continue
        arrivals = [(sum(min(op["machines"].values()) for op in ops[:index+1]), job, index)
                    for job, ops in operations.items() for index, op in enumerate(ops) if op.get("buffer_out") == bid]
        minimum = min(t for t, job, index in arrivals)
        for time, job, index in arrivals:
            if time == minimum:
                for k in range(index+1):
                    fastest = min(operations[job][k]["machines"].values())
                    excluded.update(((job, k), m) for m, p in operations[job][k]["machines"].items() if p == fastest)
    return excluded


def direct_cases(state, graph):
    original = objective(graph["timelines"], graph["windows"])
    if abs(original-state["S_seed"]) > 1e-9:
        raise AssertionError("Independent event integral differs from formal objective")
    rows = []
    by_node = defaultdict(list)
    for row in graph["event_rows"]:
        by_node[row["node"]].append(row)
    for node, events in by_node.items():
        relevant = any(e["relevant"] for e in events)
        if not relevant:
            continue
        times = []
        time = events[0]["event_time"]
        for e in events:
            bid = e["buffer_id"]
            times.extend(v["time"] for v in graph["timelines"][bid] if v["time"] != time)
            times.append(graph["windows"][bid]["start"])
        gaps = [abs(t-time) for t in times if t != time]
        epsilon = min([EPSILON] + [gap * .1 for gap in gaps])
        nonsmooth = any(e["nonsmooth"] for e in events)
        prediction = graph["g"].get(node, 0)
        fd = None if nonsmooth else (objective(graph["timelines"], graph["windows"], node, epsilon)-original) / epsilon
        rows.append({**identity(state), "node": node, "buffers": [e["buffer_id"] for e in events],
                     "event_time": time, "prediction": prediction, "actual": fd,
                     "epsilon": epsilon, "nonsmooth": nonsmooth, "event_rows": events,
                     "passed": None if nonsmooth else math.isclose(prediction, fd, rel_tol=1e-7, abs_tol=TOL)})
    return rows


def freeze():
    source = json.loads((active.OUTPUT_ROOT / "source_manifest.json").read_text())
    for relative, expected in {**source["checked_sources"], **source["capture_sha256"]}.items():
        if role.file_hash(r.PROJECT_ROOT / relative) != expected:
            raise AssertionError(f"Frozen source changed: {relative}")
    if role.file_hash(Path(active.__file__)) != source["active_diagnostic_source"]:
        raise AssertionError("Frozen AMP generator changed")
    old = json.loads((selection.OUTPUT_ROOT / "fidelity_gate.json").read_text())
    if not old["passed"] or old["states"] != 240 or not all(c["rng_off_on_exact"] for c in old["cases"]):
        raise AssertionError("Historical replay certificate missing")
    old_registration = json.loads((selection.OUTPUT_ROOT / "preregistration.json").read_text())
    for relative, expected in old_registration["files"].items():
        if role.file_hash(r.PROJECT_ROOT / relative) != expected:
            raise AssertionError(f"Pinned replay/candidate source changed: {relative}")
    files = {str(p.relative_to(r.PROJECT_ROOT)): role.file_hash(p) for p in
             (Path(__file__), selection.OUTPUT_ROOT / "fidelity_gate.json",
              selection.OUTPUT_ROOT / "preregistration.json",
              active.OUTPUT_ROOT / "raw/active_lever_actions.csv",
              selection.OUTPUT_ROOT / "raw/proposal_trials.csv")}
    payload = {"formal_source": source, "files": files, "rules": REGISTRATION}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = ROOT / "preregistration.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise AssertionError("Preregistration changed; refusing silent rule change")
    r.write_json_atomic(path, payload)
    return fingerprint, source


def prepare_task(task, source, fingerprint):
    states = local.load_capture(task, local.local_settings(), source["original_capture_fingerprint"])["states"]
    data = json.loads((r.INSTANCE_DIR / f"{task.instance}.json").read_text())
    engine = WIPGraphDualPopulation(data["operations"], data["buffers"], N=3, N_A=12, FE_max=6, seed=0)
    signature = local.formal_state_signature(engine)
    direct, edges, info = [], [], []
    excluded = parameter_boundary_keys(data["operations"], data["buffers"])
    for state in states:
        ind = r.prefix_individual(state["genome"])
        decoded = normalized_decode(data["operations"], data["buffers"], ind)
        role.verify_seed(state, decoded, WIPGraphDualPopulation(data["operations"], data["buffers"], N=3, N_A=12, FE_max=6, seed=0))
        graph = graph_from_decode(decoded, data["operations"], data["buffers"])
        scaled = normalized_decode(scaled_operations(data["operations"]), data["buffers"], ind, True)
        scaled_graph = graph_from_decode(scaled, data["operations"], data["buffers"])
        if (decoded["schedule"] != scaled["schedule"] or decoded["trace"] != scaled["trace"]
                or decoded["shortage"] != scaled["shortage"] or graph["topology"] != scaled_graph["topology"]):
            raise AssertionError("Uniform integer-grid rescaling changed seed semantics")
        cases = direct_cases(state, graph)
        direct.extend(cases)
        op_nodes = {tuple(k): ids for k, ids in graph["op_nodes"]}
        eligible, invalid, boundary = [], 0, 0
        for key, ids in op_nodes.items():
            c = ids["complete"]
            if (key, ind.ms_map[key]) in excluded:
                boundary += 1
            elif invalid_path(graph, c):
                invalid += 1
            else:
                eligible.append(key)
        rng = random.Random(int.from_bytes(hashlib.sha256(json.dumps([*identity(state).values(), "sensitivity_ops"]).encode()).digest(), "big"))
        sampled = rng.sample(eligible, min(OPS_PER_STATE, len(eligible)))
        compact = {k: state[k] for k in ("instance", "seed", "fe_index", "genome", "S_seed", "Cmax_seed")}
        r.write_json_atomic(state_path(state), {"fingerprint": fingerprint, "state": compact, "graph": graph,
                                               "sampled_operations": [list(k) for k in sampled]})
        edges.extend({**identity(state), **e} for e in graph["edges"])
        info.append({**identity(state), "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
                     "nonsmooth_nodes": len(graph["nonsmooth"]), "smooth_eligible_ops": len(eligible),
                     "invalid_path_ops": invalid, "active_start_parameter_ops": boundary,
                     "sampled_ops": len(sampled), "direct_smooth": sum(not z["nonsmooth"] for z in cases),
                     "direct_nonsmooth": sum(z["nonsmooth"] for z in cases), "DAG_passed": True})
    if local.formal_state_signature(engine) != signature:
        raise AssertionError("Seed reconstruction changed formal state")
    result = {"fingerprint": fingerprint, "direct": direct, "edges": edges, "info": info,
              "seed_decodes": len(states), "scale_equivalence_decodes": len(states), "fidelity_passed": True}
    r.write_json_atomic(ROOT / "cache/batches" / task.instance / f"seed_{task.seed}.json", result)
    print(f"Stage 0/1 captured: {task.instance} seed={task.seed}; eligible ops={sum(z['smooth_eligible_ops'] for z in info)}", flush=True)
    return result


def graph_fd_task(task, fingerprint):
    data = json.loads((r.INSTANCE_DIR / f"{task.instance}.json").read_text())
    rows = []
    for path in sorted((ROOT / "cache" / task.instance / f"seed_{task.seed}").glob("fe_*.json")):
        cached = json.loads(path.read_text())
        if cached["fingerprint"] != fingerprint:
            raise AssertionError("State cache fingerprint mismatch")
        state, graph = cached["state"], restore_graph(cached["graph"])
        graph["lambda"] = reverse(graph)
        cached["graph"] = graph
        r.write_json_atomic(path, cached)
        ind = r.prefix_individual(state["genome"])
        ids = {tuple(k): v for k, v in graph["op_nodes"]}
        for key_list in cached["sampled_operations"]:
            key = tuple(key_list)
            c = ids[key]["complete"]
            root_forward = forward(graph, {c: EPSILON})
            forward_value = sum(graph["g"].get(n, 0) * v for n, v in root_forward.items())
            reverse_value = graph["lambda"][c] * EPSILON
            if abs(forward_value-reverse_value) > 1e-9:
                raise AssertionError("Forward/reverse disagreement")
            decodes = [normalized_decode(scaled_operations(data["operations"], key, ind.ms_map[key], delta), data["buffers"], ind, True)
                       for delta in (1, -1)]
            gs = [graph_from_decode(d, data["operations"], data["buffers"]) for d in decodes]
            boundary_change = any(g["windows"] != graph["windows"] and
                                  any(g["windows"][b]["start"] != graph["windows"][b]["start"] for b in graph["windows"]) for g in gs)
            preserved = all(g["topology"] == graph["topology"] for g in gs) and not boundary_change
            fd = (decodes[0]["shortage"]-decodes[1]["shortage"]) / (2*EPSILON)
            prediction = graph["lambda"][c]
            rows.append({**identity(state), "operation": list(key), "complete_node": c,
                         "g_C": graph["g"].get(c, 0), "prediction": prediction, "actual": fd,
                         "topology_preserved": preserved, "active_start_changed": boundary_change,
                         "subgroup": "propagation_only" if sign(prediction) and not sign(graph["g"].get(c, 0)) else
                                     "direct_source" if sign(graph["g"].get(c, 0)) else "predicted_zero",
                         "forward_reverse_error": abs(forward_value-reverse_value),
                         "S_plus": decodes[0]["shortage"], "S_minus": decodes[1]["shortage"],
                         "epsilon": EPSILON, "offline_decodes": 2})
    r.write_json_atomic(ROOT / "cache/fd_batches" / task.instance / f"seed_{task.seed}.json", rows)
    print(f"Stage 2: {task.instance} seed={task.seed}, cases={len(rows)}, preserved={sum(z['topology_preserved'] for z in rows)}", flush=True)
    return rows


def group_metrics(rows, pred="prediction", actual="actual"):
    result = [{"group": "overall", **metric_pairs(rows, pred, actual)}]
    for name in local.INSTANCES:
        subset = [z for z in rows if z["instance"] == name]
        result.append({"group": name, **metric_pairs(subset, pred, actual)})
        for seed in local.SEEDS:
            result.append({"group": f"{name}/seed{seed}", **metric_pairs([z for z in subset if z["seed"] == seed], pred, actual)})
    return result


def direction_gate(groups):
    instances = [g for g in groups if g["group"] in local.INSTANCES]
    seeds = [g for g in groups if "/seed" in g["group"]]
    good = lambda g: g["nonzero_sign_accuracy"] is not None and g["nonzero_sign_accuracy"] > .5 and g["spearman"] is not None and g["spearman"] > 0
    return len(instances) == 2 and all(good(g) for g in instances) and sum(good(g) for g in seeds) >= 4


def existing_actions(state, engine, active_rows, proposal_rows):
    ind = r.prefix_individual(state["genome"])
    found = []
    for row in active_rows:
        if row["family"] == "supply_AMP" and row["policy"] == "ActivePriority":
            family = "SupplyAMP"
        elif row["family"] == "AMR" and row["policy"] in ("ECT_AMR", "RandomMS", "FastestProcessingMS"):
            family = {"ECT_AMR": "ECT-AMR", "RandomMS": "RandomMS", "FastestProcessingMS": "FastestProcessingMS"}[row["policy"]]
        else:
            continue
        candidate = json.loads(row["candidate"])
        child = active.construct_child(engine, ind, row["family"], candidate)
        found.append((family, child, float(row["S_child"]), candidate))
    for row in proposal_rows:
        if row["policy"] != "BasicMutation":
            continue
        child = selection.exact_basic(engine, ind, selection.rng_for(state, int(row["trial_id"]), "BasicMutation"))
        if selection.digest(engine.genotype_key(child)) != row["child_genotype_id"]:
            raise AssertionError("Historical BasicMutation genotype reconstruction mismatch")
        family = "BasicMutation-OS" if child.os_seq != ind.os_seq else "BasicMutation-MS"
        found.append((family, child, float(row["S_child"]), {}))
    return found


def child_task(task, fingerprint):
    data = json.loads((r.INSTANCE_DIR / f"{task.instance}.json").read_text())
    engine = WIPGraphDualPopulation(data["operations"], data["buffers"], N=3, N_A=12, FE_max=6, seed=0)
    before = local.formal_state_signature(engine)
    def load_rows(path):
        with path.open() as f:
            return [z for z in csv.DictReader(f) if z["instance"] == task.instance and int(z["seed"]) == task.seed]
    active_rows = load_rows(active.OUTPUT_ROOT / "raw/active_lever_actions.csv")
    proposal_rows = load_rows(selection.OUTPUT_ROOT / "raw/proposal_trials.csv")
    rows, contexts, decodes = [], [], 0
    for path in sorted((ROOT / "cache" / task.instance / f"seed_{task.seed}").glob("fe_*.json")):
        cached = json.loads(path.read_text())
        state, graph = cached["state"], restore_graph(cached["graph"])
        candidates = existing_actions(state, engine,
                     [z for z in active_rows if int(z["fe_index"]) == state["fe_index"]],
                     [z for z in proposal_rows if int(z["fe_index"]) == state["fe_index"]])
        unique = {}
        for family, child, shortage, candidate in candidates:
            key = engine.genotype_key(child)
            if key not in unique:
                decoded = normalized_decode(data["operations"], data["buffers"], child)
                if decoded["shortage"] != shortage:
                    raise AssertionError("Historical child objective mismatch")
                child_graph = graph_from_decode(decoded, data["operations"], data["buffers"])
                times_by_uid = {tuple(n["uid"]): n["time"] for n in child_graph["nodes"]}
                changes = {n["id"]: times_by_uid[tuple(n["uid"])]-n["time"] for n in graph["nodes"]}
                changed = {n for n, dt in changes.items() if sign(dt)}
                nonsmooth = any(invalid_path(graph, node) for node in changed)
                preserved = graph["topology"] == child_graph["topology"]
                ratio = len(set(graph["topology"]) ^ set(child_graph["topology"])) / max(1, len(graph["topology"]))
                group = "nonsmooth_seed" if nonsmooth else "topology_preserved" if preserved else "mild_topology_change" if ratio <= .05 else "strong_topology_change"
                benefit = state["S_seed"] - shortage
                full = -sum(graph["g"].get(n, 0) * delta for n, delta in changes.items())
                unique[key] = {"actual": benefit, "full_prediction": full, "topology_group": group,
                               "topology_changed_fraction": ratio, "changes": changes}
                decodes += 1
            result = unique[key]
            row = {**identity(state), "family": family, "child_id": selection.digest(key),
                   "S_seed": state["S_seed"],
                   "actual": result["actual"], "full_prediction": result["full_prediction"],
                   "topology_group": result["topology_group"], "topology_changed_fraction": result["topology_changed_fraction"]}
            # Same policy/genotype counted once per state; no action quality filtering.
            if any(z["family"] == family and z["child_id"] == row["child_id"] and z["fe_index"] == row["fe_index"] for z in rows):
                continue
            rows.append(row)
            contexts.append({"row": row, "changes": result["changes"], "candidate": candidate})
        print(f"Stage 3A children: {task.instance} seed={task.seed} FE={state['fe_index']}", flush=True)
    if local.formal_state_signature(engine) != before:
        raise AssertionError("Retrospective decoding changed formal state")
    r.write_json_atomic(ROOT / "cache/child_batches" / task.instance / f"seed_{task.seed}.json",
                        {"fingerprint": fingerprint, "contexts": contexts, "unique_decodes": decodes})
    return rows, decodes


def graph_oracles(context, graph):
    changes = {int(n): dt for n, dt in context["changes"].items()}
    changed = {n for n, dt in changes.items() if sign(dt)}
    parents = defaultdict(list)
    for e in graph["edges"]:
        if e["alpha"]:
            parents[e["dst"]].append(e["src"])
    roots = {n: changes[n] for n in changed if not any(p in changed for p in parents[n])}
    reconstructed = forward(graph, roots)
    graph_pred = -sum(graph["lambda"][n] * dt for n, dt in roots.items())
    direct_pred = -sum(graph["g"].get(n, 0) * dt for n, dt in roots.items())
    forward_pred = -sum(graph["g"].get(n, 0) * dt for n, dt in reconstructed.items())
    if abs(graph_pred-forward_pred) > 1e-7:
        raise AssertionError("Oracle forward/reverse inconsistency")
    node_kinds = {n["id"]: n["kind"] for n in graph["nodes"]}
    return {**context["row"], "graph_prediction": graph_pred, "root_direct_prediction": direct_pred,
            "root_count": len(roots), "roots": [{"node": n, "kind": node_kinds[n], "delta_t": dt} for n, dt in roots.items()],
            "timing_MAE": statistics.mean(abs(reconstructed[n]-changes[n]) for n in changes),
            "timing_sign_accuracy": statistics.mean(sign(reconstructed[n]) == sign(changes[n]) for n in changes),
            "timing_spearman": corr(list(reconstructed.values()), [changes[n] for n in reconstructed]),
            "forward_reverse_error": abs(graph_pred-forward_pred)}


def ranking_metrics(rows, pred):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["instance"], row["seed"], row["fe_index"], row["family"])].append(row)
    result = []
    for key, values in groups.items():
        if len(values) < 2:
            continue
        coefficient = corr([z[pred] for z in values], [z["actual"] for z in values])
        best = max(values, key=lambda z: (z[pred], z["child_id"]))
        random_rate = statistics.mean(z["actual"] > TOL for z in values)
        result.append({"instance": key[0], "seed": key[1], "fe_index": key[2], "family": key[3],
                       "predictor": pred, "actions": len(values), "rank_spearman": coefficient,
                       "predicted_best_improves": best["actual"] > TOL,
                       "predicted_best_actual": best["actual"], "random_candidate_improve_rate": random_rate})
    return result


def prediction_metrics(rows, pred):
    metric = metric_pairs(rows, pred, "actual")
    metric["predictor"] = pred
    improve = [z for z in rows if z["actual"] > TOL]
    top = sorted(rows, key=lambda z: (-z[pred], z["child_id"]))[:math.ceil(len(rows)/4)]
    base_rate = len(improve) / len(rows) if rows else None
    top_rate = statistics.mean(z["actual"] > TOL for z in top) if top else None
    metric.update(top_quartile_improvement_rate=top_rate,
                  top_quartile_lift=top_rate/base_rate if base_rate else None,
                  classification_accuracy=statistics.mean((z[pred] > TOL) == (z["actual"] > TOL) for z in rows) if rows else None)
    return metric


def calibration_rows(rows, pred):
    grouped = defaultdict(list)
    for row in rows:
        value = row[pred]
        boundary = .01 * row["S_seed"]
        label = ("near_zero" if abs(value) <= TOL else "strongly_harmful" if value <= -boundary else
                 "mildly_harmful" if value < 0 else "strongly_beneficial" if value >= boundary else "mildly_beneficial")
        grouped[(row["topology_group"], row["family"], label)].append(row)
    return [{"predictor": pred, "topology_group": group, "family": family, "bin": label,
             "cases": len(values), "mean_prediction": statistics.mean(z[pred] for z in values),
             "mean_actual": statistics.mean(z["actual"] for z in values),
             "actual_improve_rate": statistics.mean(z["actual"] > TOL for z in values)}
            for (group, family, label), values in sorted(grouped.items())]


def pressure_test(contexts):
    """Rank only previously decoded alternatives; no new machine portfolio."""
    groups = defaultdict(dict)
    for context in contexts:
        row = context["row"]
        if row["family"] not in ("ECT-AMR", "RandomMS", "FastestProcessingMS"):
            continue
        candidate = context["candidate"]
        target = tuple(candidate["target_op"])
        groups[(row["instance"], row["seed"], row["fe_index"], target)][candidate["new_machine"]] = context
    scores, chosen, offline_decodes = [], [], 0
    for (name, seed, fe, target), alternatives in groups.items():
        if len(alternatives) < 2:
            continue
        cached = json.loads(state_path({"instance": name, "seed": seed, "fe_index": fe}).read_text())
        graph, state = restore_graph(cached["graph"]), cached["state"]
        data = json.loads((r.INSTANCE_DIR / f"{name}.json").read_text())
        engine = WIPGraphDualPopulation(data["operations"], data["buffers"], N=3, N_A=12, FE_max=6, seed=0)
        ind = r.prefix_individual(state["genome"])
        decoded = normalized_decode(data["operations"], data["buffers"], ind)
        offline_decodes += 1
        ready, _ = active.readiness(decoded, data["operations"], data["buffers"], target)
        release = decoded["provenance"].op_events[target]["release"]
        if invalid_path(graph, release):
            continue
        observed = []
        for machine, context in alternatives.items():
            duration = data["operations"][target[0]][target[1]]["machines"][machine]
            # Frozen prior ECT surrogate predicts completion, not blocked release.
            prediction = active.earliest_slot(decoded["schedule"], target, machine, ready, duration) + duration
            gain = decoded["provenance"].events[release].time-prediction
            observed.append({**identity(state), "target": list(target), "machine": machine,
                             "ECT_only": gain, "LambdaECT": graph["lambda"][release]*gain,
                             "processing_time": duration, "actual": context["row"]["actual"]})
        scores.extend(observed)
        rng = selection.rng_for(state, 0, "sensitivity_pressure_random")
        for policy in ("LambdaECT", "ECT_only", "RandomMS", "FastestProcessingMS"):
            best = rng.choice(observed) if policy == "RandomMS" else min(observed, key=lambda z: (z["processing_time"], z["machine"])) if policy == "FastestProcessingMS" else min(observed, key=lambda z: (-z[policy], z["machine"]))
            chosen.append({**identity(state), "target": list(target), "policy": policy,
                           "chosen_machine": best["machine"], "actual": best["actual"],
                           "rank_spearman": corr([z[policy] for z in observed], [z["actual"] for z in observed]) if policy in ("LambdaECT", "ECT_only") else None})
    return scores, chosen, offline_decodes


def parallel(tasks, function, *args):
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(function, task, *args) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def save(name, rows, folder="analysis"):
    path = ROOT / folder / name
    if rows:
        local.save_csv(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("status,reason\nNOT_TESTED,Earlier gate stopped this stage\n")


def finish(summary, stage2=None, children=None):
    summary["formal_FE"] = 0
    summary["source_changes"] = "Diagnostic module/tests only; Full/SIS/encoder/decoder/defaults unchanged"
    summary["recommendation"] = "GO" if all(summary["stages"].get(k) == "PASS" for k in ("direct", "graph", "discrete", "predecode")) else "NO-GO"
    r.write_json_atomic(ROOT / "analysis/sensitivity_validation_summary.json", summary)
    lines = ["# Shortage Sensitivity Validation", "", "## A. Fidelity", json.dumps(summary["fidelity"], indent=2),
             "240 historical states are reused. No Full trajectory run is needed: seed provenance is reconstructed offline and verified against pinned captures. Prior selection-aware off/on replay certificates are reused; historic RNG raw does not exist.",
             "Formal FE=0; offline seed/grid/perturbation/child calls are reported separately. No diagnostic RNG is attached to the formal search.",
             "", "## B. Direct sensitivity", json.dumps(summary.get("direct", {}), indent=2),
             "Unit deficit increments 0->1 and 1->2 both have g=-1. Isolated last take has g=deficit_before because active_end moves with it; an internal-event slope cannot be blindly used at a boundary.",
             "", "## C. Computational timing graph", json.dumps(summary.get("graph_coverage", {}), indent=2),
             "Processing and uniquely binding readiness/release relations only. Job/WIP aliases of one release are merged; priority competition is discrete, never a derivative edge. Nonbinding completion of a blocked release has alpha=0. Ties are flagged, not arbitrarily differentiated.",
             "", "## D. Processing-time FD", json.dumps(summary.get("graph_fd", {}), indent=2),
             "Formal int(p) and int(event_time) preclude literal subinteger FD. Multiply integer processing times by 1000, perturb by +/-1, divide output times/areas by 1000. Every scaled seed schedule/trace/objective/topology must match the unscaled seed. This tests a continuous timing extension on a refined integer grid, not the derivative of the literal unscaled quantizer.",
             "Parameters affecting theoretical active_start are excluded from Lambda_C-only validation and counted explicitly; their derivative has an additional window-boundary parameter path.",
             "", "## E. Propagation-only", json.dumps(summary.get("propagation_only", {}), indent=2),
             "Direct C sensitivities are zero in this S/C/R representation: buffer put is attributed to R, take to S. Nonzero Lambda_C therefore tests genuine downstream propagation.",
             "", "## F. Full-time child oracle", json.dumps(summary.get("full_oracle", {}), indent=2),
             "", "## G. Graph oracle", json.dumps(summary.get("graph_oracle", {}), indent=2),
             "Oracle predictions use actual child times and are not deployable. Families and within-state ranking are separated; topology changes and nonsmooth paths are never pooled into primary results.",
             "", "## H. AMR pressure test", json.dumps(summary.get("pressure", {}), indent=2),
             "If reached, only previously saved MS alternatives are ranked with seed graph and frozen ECT slot computation. No child objective participates in scoring. Completion-as-release is only the old surrogate, not a new blocker predictor.",
             "", "## I. Failure localization", summary.get("failure", "No failure gate"),
             "", "## J. Decision", f"**{summary['recommendation']}**", json.dumps(summary["stages"], indent=2),
             "NO-GO means no justified progression to formal event-graph directional guidance. Insufficient smooth coverage is not proof that a mathematical derivative is wrong; failed forward/reverse or FD tests have a different meaning.",
             "No Full/SIS/defaults changed, no new P_S operator, no formal pre-ablation rerun."]
    (ROOT / "analysis/sensitivity_validation_interpretation.md").write_text("\n".join(lines) + "\n")
    make_figures(stage2 or [], children or [])
    print(f"Sensitivity validation: {summary['recommendation']}; stages={summary['stages']}; offline calls={summary['offline_decodes']}", flush=True)
    return summary


def make_figures(stage2, children):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = ROOT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    if stage2:
        rows = [z for z in stage2 if z["topology_preserved"]]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter([z["prediction"] for z in rows], [z["actual"] for z in rows], s=9, alpha=.4)
        ax.set(xlabel="Lambda_C", ylabel="Processing-time FD", title="Topology-preserved random sample")
        fig.tight_layout(); fig.savefig(folder / "graph_fd.png", dpi=160); plt.close(fig)
    if children:
        for pred, filename in (("full_prediction", "full_oracle.png"), ("graph_prediction", "graph_oracle.png")):
            rows = [z for z in children if pred in z and z["topology_group"] == "topology_preserved"]
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter([z[pred] for z in rows], [z["actual"] for z in rows], s=10, alpha=.4)
            ax.set(xlabel=pred, ylabel="Actual benefit")
            fig.tight_layout(); fig.savefig(folder / filename, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Set PYTHONHASHSEED=0")
    fingerprint, source = freeze()
    tasks = [r.Task(name, "full", seed) for name in local.INSTANCES for seed in local.SEEDS]
    summary = {"fingerprint": fingerprint, "stages": {k: "NOT TESTED" for k in ("direct", "graph", "discrete", "predecode")}, "offline_decodes": 0}
    batches = parallel(tasks, prepare_task, source, fingerprint)
    direct = [z for batch in batches for z in batch["direct"]]
    edges = [z for batch in batches for z in batch["edges"]]
    info = [z for batch in batches for z in batch["info"]]
    if len(info) != 240 or not all(batch["fidelity_passed"] for batch in batches):
        raise AssertionError("Incomplete 240-state fidelity gate")
    summary["offline_decodes"] = 480
    summary["fidelity"] = {"reused_states": 240, "new_trajectory_replays": 0, "source_pinned": True,
                           "historic_seed_genes_objectives_intervals_phi_H_dependencies": "matched", "scale_equivalence": "240/240", "prior_archive_counter_child_checkpoint_RNG_certificate": "reused pinned reconstruction/off-on exact"}
    smooth = [z for z in direct if not z["nonsmooth"]]
    summary["direct"] = {**metric_pairs(smooth), "smooth_events": len(smooth), "nonsmooth_events": len(direct)-len(smooth)}
    summary["graph_coverage"] = {"nodes": sum(z["nodes"] for z in info), "edges": len(edges),
        "edge_types": dict(Counter(e["kind"] for e in edges)), "alpha_one_edges": sum(e["alpha"] == 1 for e in edges),
        "tied_edges": sum(e["tie"] for e in edges), "nonsmooth_nodes": sum(z["nonsmooth_nodes"] for z in info),
        "invalid_path_ops": sum(z["invalid_path_ops"] for z in info), "active_start_parameter_ops": sum(z["active_start_parameter_ops"] for z in info), "DAG_states_passed": 240}
    save("seed_event_sensitivity.csv", direct, "raw")
    save("timing_dependency_edges.csv", edges, "raw")
    save("direct_sensitivity_summary.csv", group_metrics(smooth))
    save("topology_summary.csv", info)
    for name in ("perturbation_cases.csv", "existing_child_predictions.csv", "amr_predecode_scores.csv"):
        save(name, [], "raw")
    for name in ("graph_fd_summary.csv", "propagation_only_summary.csv", "child_prediction_summary.csv", "within_state_ranking.csv", "amr_pressure_test.csv", "instance_seed_summary.csv"):
        save(name, [])
    if not smooth or not all(z["passed"] for z in smooth):
        summary["stages"]["direct"] = "FAIL" if smooth else "NOT TESTED"
        summary["failure"] = "Gate A: direct derivative mismatch or no smooth objective events. Graph/child stages withheld."
        return finish(summary)
    summary["stages"]["direct"] = "PASS"
    fd = [z for batch in parallel(tasks, graph_fd_task, fingerprint) for z in batch]
    summary["offline_decodes"] += 2 * len(fd)
    primary = [z for z in fd if z["topology_preserved"]]
    grouped_events = defaultdict(list)
    for row in direct:
        grouped_events[(row["instance"], row["seed"], row["fe_index"])].append(row)
    for (name, seed, fe), events in grouped_events.items():
        cached_graph = restore_graph(json.loads(state_path({"instance": name, "seed": seed, "fe_index": fe}).read_text())["graph"])
        node_info = {n["id"]: n for n in cached_graph["nodes"]}
        for row in events:
            row["lambda"] = cached_graph["lambda"][row["node"]]
            row["sensitivity_path_valid"] = not invalid_path(cached_graph, row["node"])
            row["event_uid"] = node_info[row["node"]]["uid"]
    save("seed_event_sensitivity.csv", direct, "raw")
    groups = group_metrics(primary)
    propagation = [z for z in primary if z["subgroup"] == "propagation_only"]
    prop_groups = group_metrics(propagation)
    summary["graph_fd"] = {"sampled_cases": len(fd), "topology_preserved": len(primary), "topology_preserved_rate": len(primary)/len(fd) if fd else None, "groups": groups}
    summary["propagation_only"] = {"cases": len(propagation), "nonzero_FD": sum(bool(sign(z["actual"])) for z in propagation), "groups": prop_groups}
    save("perturbation_cases.csv", fd, "raw")
    save("graph_fd_summary.csv", groups)
    save("propagation_only_summary.csv", prop_groups)
    save("instance_seed_summary.csv", groups)
    prop_instances = [g for g in prop_groups if g["group"] in local.INSTANCES]
    propagation_pass = (bool(propagation) and
                        all(g["nonzero_sign_accuracy"] is not None and g["nonzero_sign_accuracy"] > .5
                            for g in prop_instances) and
                        all(any(sign(z["actual"]) for z in propagation if z["instance"] == name)
                            for name in local.INSTANCES))
    if not direction_gate(groups) or not propagation_pass:
        has_information = bool(primary) and any(sign(z["actual"]) for z in primary)
        summary["stages"]["graph"] = "FAIL" if has_information else "NOT TESTED"
        summary["failure"] = "Gate B: computational graph FD, propagation-only information, or cross-instance/group stability did not pass. No existing-child/pressure analysis performed."
        return finish(summary, fd)
    summary["stages"]["graph"] = "PASS"
    child_batches = parallel(tasks, child_task, fingerprint)
    children = [z for rows, count in child_batches for z in rows]
    summary["offline_decodes"] += sum(count for rows, count in child_batches)
    primary_children = [z for z in children if z["topology_group"] == "topology_preserved"]
    full_groups = group_metrics(primary_children, "full_prediction")
    summary["full_oracle"] = {"cases": len(children), "topology_groups": dict(Counter(z["topology_group"] for z in children)), "primary": full_groups}
    child_summaries = [{"topology_group": group, "family": family, **prediction_metrics([z for z in children if z["topology_group"] == group and z["family"] == family], "full_prediction")}
                       for group in sorted({z["topology_group"] for z in children}) for family in sorted({z["family"] for z in children})]
    save("existing_child_predictions.csv", children, "raw")
    save("child_prediction_summary.csv", child_summaries)
    save("calibration_bins.csv", calibration_rows(children, "full_prediction"))
    if not direction_gate(full_groups):
        has_information = bool(primary_children) and any(sign(z["actual"]) for z in primary_children)
        summary["stages"]["discrete"] = "FAIL" if has_information else "NOT TESTED"
        summary["failure"] = ("Gate 3A: primary smooth topology-preserved full-time oracle accuracy failed. "
                              if has_information else "Gate 3A is not identifiable: primary smooth topology-preserved existing children have no nonzero shortage changes. Zero-case agreement is not explanatory power. ")
        summary["failure"] += "Graph oracle and AMR ranking withheld. Fixed complete computational topology plus unchanged processing inputs can make OS-only timing shifts impossible; MS reassignment necessarily changes machine-sequence tokens."
        return finish(summary, fd, children)
    contexts, oracles = [], []
    for task in tasks:
        cached = json.loads((ROOT / "cache/child_batches" / task.instance / f"seed_{task.seed}.json").read_text())
        contexts.extend(cached["contexts"])
        for context in cached["contexts"]:
            graph = restore_graph(json.loads(state_path(context["row"]).read_text())["graph"])
            oracles.append(graph_oracles(context, graph))
    primary_oracles = [z for z in oracles if z["topology_group"] == "topology_preserved"]
    rankings = ranking_metrics(primary_oracles, "graph_prediction") + ranking_metrics(primary_oracles, "root_direct_prediction")
    m_graph, m_direct = (prediction_metrics(primary_oracles, p) for p in ("graph_prediction", "root_direct_prediction"))
    ranks = [z["rank_spearman"] for z in rankings if z["predictor"] == "graph_prediction" and z["rank_spearman"] is not None]
    lifts_ok = m_graph["top_quartile_lift"] is not None and m_graph["top_quartile_lift"] > 1
    graph_ranks = [z for z in rankings if z["predictor"] == "graph_prediction" and z["rank_spearman"] is not None]
    def rank_mean(name, seed=None):
        values = [z["rank_spearman"] for z in graph_ranks if z["instance"] == name and (seed is None or z["seed"] == seed)]
        return statistics.mean(values) if values else None
    rank_ok = (all((rank_mean(name) or 0) > 0 for name in local.INSTANCES) and
               sum((rank_mean(name, seed) or 0) > 0 for name in local.INSTANCES for seed in local.SEEDS) >= 4)
    superior = (m_graph["actual_nonzero_sign_accuracy"] or 0) > (m_direct["actual_nonzero_sign_accuracy"] or 0) and (m_graph["spearman"] or 0) > (m_direct["spearman"] or 0)
    summary["graph_oracle"] = {"graph": m_graph, "root_direct": m_direct, "within_state_mean_rank": statistics.mean(ranks) if ranks else None,
                               "groups": group_metrics(primary_oracles, "graph_prediction")}
    save("existing_child_predictions.csv", oracles, "raw")
    save("within_state_ranking.csv", rankings)
    save("calibration_bins.csv", [z for p in ("full_prediction", "graph_prediction", "root_direct_prediction") for z in calibration_rows(oracles, p)])
    if not superior or not lifts_ok or not rank_ok or not direction_gate(summary["graph_oracle"]["groups"]):
        summary["stages"]["discrete"] = "FAIL"
        summary["failure"] = "Gate C: graph oracle lacks incremental value over root-direct and/or stable within-state action ranking. No predecode test."
        return finish(summary, fd, oracles)
    summary["stages"]["discrete"] = "PASS"
    scores, chosen, pressure_decodes = pressure_test(contexts)
    summary["offline_decodes"] += pressure_decodes
    save("amr_predecode_scores.csv", scores, "raw")
    save("amr_pressure_test.csv", chosen)
    differences = []
    for name in local.INSTANCES:
        groups_to_check = [None, *local.SEEDS]
        for seed in groups_to_check:
            rows = [z for z in chosen if z["instance"] == name and (seed is None or z["seed"] == seed)]
            rates = {policy: statistics.mean(z["actual"] > TOL for z in rows if z["policy"] == policy) for policy in ("LambdaECT", "ECT_only")} if rows else {}
            rank = {policy: statistics.mean(z["rank_spearman"] for z in rows if z["policy"] == policy and z["rank_spearman"] is not None) if any(z["policy"] == policy and z["rank_spearman"] is not None for z in rows) else None for policy in ("LambdaECT", "ECT_only")}
            differences.append({"instance": name, "seed": seed, "top1_difference": rates.get("LambdaECT", 0)-rates.get("ECT_only", 0),
                                "rank_difference": rank["LambdaECT"]-rank["ECT_only"] if all(v is not None for v in rank.values()) else None,
                                "groups": len(rows)//4})
    primary_groups = [g for g in differences if g["seed"] is None]
    pressure_pass = all(g["groups"] and (g["top1_difference"] > 0 or (g["rank_difference"] or 0) > 0) for g in primary_groups) and sum(g["top1_difference"] >= 0 for g in differences if g["seed"] is not None) >= 4
    summary["pressure"] = {"groups": differences, "candidate_scores": len(scores), "comparisons": len(chosen)//4}
    summary["stages"]["predecode"] = "PASS" if pressure_pass else "FAIL" if chosen else "NOT TESTED"
    summary["failure"] = "No failure gate" if pressure_pass else "Gate D: seed Lambda + ECT does not establish stable predecode action-ranking advantage."
    return finish(summary, fd, oracles)


if __name__ == "__main__":
    main()
