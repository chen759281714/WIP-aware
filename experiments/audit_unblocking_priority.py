"""LEGACY / DEPRECATED DIAGNOSTIC SCRIPT.

The direct unblocking_priority operator was removed from the formal algorithm.
This module retains historical record parsing helpers only; it does not run
as part of the current algorithm or validation workflow.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithms.wip_graph_dual_population import Individual, Move
from src.solution.decoder import ActiveDependency, OpKey

def _records_by_op(ind: Individual) -> Dict[OpKey, dict]:
    return {(rec["job"], rec["op"]): rec for rec in ind.schedule}


def _schedule_signature(ind: Individual) -> Dict[OpKey, tuple]:
    return {key: (rec["machine"], rec["start"], rec["end"], rec.get("release", rec["end"]))
            for key, rec in _records_by_op(ind).items()}


def _v_limiting_cause(child: Individual, v_op: OpKey) -> str:
    start_id = child.provenance.op_events[v_op]["start"]
    incoming = {edge.kind for edge in child.provenance.active_dependencies
                if edge.dst_event_id == start_id}
    machine = "machine" in incoming
    wip = "wip" in incoming
    if machine and wip:
        return "both_machine_and_wip_bound"
    if machine:
        return "machine_bound"
    if wip:
        return "wip_bound"
    return "unknown_or_other"


def _u_trigger_status(child: Individual, u_op: OpKey, v_op: OpKey,
                      blocking_u_after: float) -> str:
    release_id = child.provenance.op_events[u_op]["release"]
    triggers = [edge for edge in child.provenance.active_dependencies
                if edge.kind == "unblocking" and edge.dst_event_id == release_id]
    for edge in triggers:
        event = child.provenance.events[edge.src_event_id]
        if (event.job, event.op_idx) == v_op:
            return "same_trigger_still_active"
    if triggers:
        return "different_trigger"
    if blocking_u_after <= 0:
        return "no_longer_blocked"
    return "release_unchanged_other_reason"


def build_unblocking_audit_record(parent: Individual, child: Individual,
                                  edge: ActiveDependency, move: Move,
                                  accepted: bool, fe_index: int) -> Dict[str, Any]:
    source = parent.provenance.events[edge.src_event_id]
    target = parent.provenance.events[edge.dst_event_id]
    assert edge.kind == "unblocking" and source.kind == "start" and target.kind == "release"
    v_op, u_op = (source.job, source.op_idx), (target.job, target.op_idx)
    before, after = _records_by_op(parent), _records_by_op(child)
    result: Dict[str, Any] = {
        "fe_index": fe_index,
        "v_op_key": list(v_op),
        "u_op_key": list(u_op),
        "buffer_id": edge.buffer_id,
        "move_span": abs(child.os_seq.index(move.key) - parent.os_seq.index(move.key)),
        "move_depth": move.depth,
        "priority_rank_v_before": parent.os_seq.index(v_op),
        "priority_rank_v_after": child.os_seq.index(v_op),
        "priority_rank_u_before": parent.os_seq.index(u_op),
        "priority_rank_u_after": child.os_seq.index(u_op),
        "genotype_changed": (parent.os_seq != child.os_seq or parent.ms_map != child.ms_map),
        "schedule_changed": _schedule_signature(parent) != _schedule_signature(child),
        "accepted_into_PM": bool(accepted),
    }
    for op_label, op in (("v", v_op), ("u", u_op)):
        for when, rec in (("before", before[op]), ("after", after[op])):
            result[f"start_{op_label}_{when}"] = rec["start"]
            result[f"complete_{op_label}_{when}"] = rec["end"]
            result[f"release_{op_label}_{when}"] = rec.get("release", rec["end"])
    result["blocking_u_before"] = result["release_u_before"] - result["complete_u_before"]
    result["blocking_u_after"] = result["release_u_after"] - result["complete_u_after"]
    result["makespan_before"] = parent.makespan
    result["makespan_after"] = child.makespan
    result["delta_start_v"] = result["start_v_after"] - result["start_v_before"]
    result["delta_release_u"] = result["release_u_after"] - result["release_u_before"]
    result["delta_blocking_u"] = result["blocking_u_after"] - result["blocking_u_before"]
    result["delta_makespan"] = child.makespan - parent.makespan
    dv, du = result["delta_start_v"], result["delta_release_u"]
    result["causal_success"] = dv < 0 and du < 0
    result["start_only"] = dv < 0 and du == 0
    result["release_without_start"] = dv == 0 and du < 0
    result["no_timing_effect"] = dv == 0 and du == 0
    result["adverse_effect"] = dv > 0 or du > 0
    result["genotype_changed_but_schedule_same"] = (result["genotype_changed"]
                                                     and not result["schedule_changed"])
    result["v_limiting_cause_after"] = (_v_limiting_cause(child, v_op) if dv == 0 else None)
    result["u_trigger_status_after"] = (_u_trigger_status(child, u_op, v_op,
                                                            result["blocking_u_after"])
                                        if dv < 0 and du == 0 else None)
    return result


def _numeric_summary(values: List[float]) -> Dict[str, Any]:
    return {"count": len(values),
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None}


def summarize(records: List[dict]) -> Dict[str, Any]:
    total = len(records)
    def counts_for(field: str) -> Dict[str, Any]:
        count = sum(bool(rec[field]) for rec in records)
        return {"count": count, "rate": count / total if total else None}

    def sign_counts(field: str) -> Dict[str, Any]:
        values = [rec[field] for rec in records]
        return {name: {"count": sum(test(v) for v in values),
                       "rate": sum(test(v) for v in values) / total if total else None}
                for name, test in (("negative", lambda v: v < 0),
                                   ("zero", lambda v: v == 0),
                                   ("positive", lambda v: v > 0))}

    v_same = [rec for rec in records if rec["delta_start_v"] == 0]
    start_only = [rec for rec in records if rec["delta_start_v"] < 0
                  and rec["delta_release_u"] == 0]
    return {
        "selected_moves": total,
        "v_start": sign_counts("delta_start_v"),
        "u_release": sign_counts("delta_release_u"),
        "blocking_duration": sign_counts("delta_blocking_u"),
        "makespan": sign_counts("delta_makespan"),
        "categories": {name: counts_for(name) for name in
                       ("causal_success", "start_only", "release_without_start",
                        "no_timing_effect", "adverse_effect", "schedule_changed",
                        "genotype_changed_but_schedule_same", "accepted_into_PM")},
        "deltas": {name: _numeric_summary([rec[name] for rec in records]) for name in
                   ("delta_start_v", "delta_release_u", "delta_blocking_u", "delta_makespan")},
        "v_same_limiting_cause": {"denominator": len(v_same),
                                  "counts": dict(Counter(rec["v_limiting_cause_after"] for rec in v_same))},
        "start_only_trigger_status": {"denominator": len(start_only),
                                       "counts": dict(Counter(rec["u_trigger_status_after"]
                                                              for rec in start_only))},
    }


def main() -> None:
    print("LEGACY: this script documents the retired direct unblocking_priority "
          "operator and is not part of the current algorithm workflow.")


if __name__ == "__main__":
    main()
