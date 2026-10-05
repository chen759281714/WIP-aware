"""Dual specialized populations using factual WIP event provenance.

OS genes are explicit operations; every variation preserves job precedence.
The archive is nondominated memory, not a third breeding population.
P_M acts on machine and processing structures; unblocking and WIP edges
only propagate the backward trace toward those structures.
"""

from __future__ import annotations

import math
import random
import warnings
from bisect import bisect_right, insort
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.solution.decoder import DecodeProvenance, OpKey, StageBufferWIPScheduler
from src.solution.encoder import Encoder


@dataclass
class Individual:
    os_seq: List[OpKey]
    ms_map: Dict[OpKey, str]
    makespan: Optional[int] = None
    shortage: Optional[float] = None
    schedule: Optional[List[Dict[str, Any]]] = None
    buffer_trace: Optional[Dict[str, list]] = None
    stats: Optional[Dict[str, Any]] = None
    provenance: Optional[DecodeProvenance] = None
    operation_order: Optional[Tuple[OpKey, ...]] = None

    @property
    def OS(self) -> List[OpKey]:
        return self.os_seq

    @property
    def MS(self) -> List[str]:
        if self.operation_order is None:
            raise ValueError("MS list requires an explicit operation_order")
        return [self.ms_map[key] for key in self.operation_order]

    def copy(self) -> "Individual":
        return Individual(self.os_seq[:], self.ms_map.copy(), self.makespan,
                          self.shortage, self.schedule, self.buffer_trace,
                          self.stats, self.provenance, self.operation_order)


@dataclass(frozen=True)
class Move:
    kind: str
    key: OpKey
    other: Optional[OpKey] = None
    machine: Optional[str] = None
    depth: int = 0


@dataclass(frozen=True)
class RelinkUnit:
    key: OpKey
    kind: str


@dataclass(frozen=True)
class OSRelinkAction:
    key: OpKey
    target_index: int
    d_before: int
    d_after: int
    displacement: int


@dataclass
class ShortageDiagnosis:
    intervals: List[Tuple[str, int, int, float, Set[OpKey]]]
    phi: Dict[OpKey, float]
    influence_events: Set[int]
    influence_dependencies: list
    unanchored_intervals: int = 0


class WIPGraphDualPopulation:
    """Makespan/shortage steady-state search with a shared Pareto archive."""

    algorithm_variant = "full"
    pm_specialized_kind = "MGS"
    ps_specialized_kind = "SIS"
    pm_archive_guidance_enabled = True
    ps_archive_guidance_enabled = True

    def __init__(self, operations: Dict[str, list], buffers: Dict[str, dict],
                 N: int = 50, N_A: int = 100, rho: float = 0.8,
                 eta: float = 0.3, p_mut: float = 0.1, T_coop: int = 10,
                 FE_max: int = 10000,
                 seed: Optional[int] = None):
        if N < 1 or N_A < 2 or FE_max < 2 * N or T_coop < 1:
            raise ValueError("Require N>=1, N_A>=2, FE_max>=2N, T_coop>=1")
        if not (0 < rho <= 1 and 0 < eta <= 1 and 0 <= p_mut <= 1):
            raise ValueError("Invalid rho, eta, or p_mut")
        self.operations, self.buffers = operations, buffers
        self.N, self.N_A = N, N_A
        self.rho, self.eta, self.p_mut = rho, eta, p_mut
        self.T_coop, self.FE_max = T_coop, FE_max
        self.rng = random.Random(seed)
        self.encoder = Encoder(operations, rng=self.rng)
        self.decoder = StageBufferWIPScheduler(operations, buffers)
        self._distance_keys = tuple(self.encoder.ms_index_order)
        n_operations = len(self._distance_keys)
        self._distance_pair_count = (n_operations * (n_operations - 1) // 2
                                     - sum(len(ops) * (len(ops) - 1) // 2
                                           for ops in operations.values()))
        self.P_M: List[Individual] = []
        self.P_S: List[Individual] = []
        self.A: List[Individual] = []
        self.n_evaluations = 0
        self.round = 0
        self.unanchored_shortage_intervals = 0
        self.phi_mass_mismatch_count = 0
        self.machine_priority_move_spans: List[int] = []
        pm_kinds = ("machine_priority", "machine_reassign", "basic_mutation",
                    "fallback_basic_mutation")
        ps_kinds = ("shortage_relink", "basic_mutation",
                    "fallback_basic_mutation_phi_invalid",
                    "fallback_basic_mutation_no_guide",
                    "fallback_basic_mutation_zero_shortage")
        self.diagnostics = {
            "algorithm_variant": self.algorithm_variant,
            "pm_specialized_kind": self.pm_specialized_kind,
            "ps_specialized_kind": self.ps_specialized_kind,
            "pm_archive_guidance_enabled": self.pm_archive_guidance_enabled,
            "ps_archive_guidance_enabled": self.ps_archive_guidance_enabled,
            "pm_moves": {kind: {"generated": 0, "accepted": 0} for kind in pm_kinds},
            "pm_action_sets": [],
            "pm_selected_depths": {kind: [] for kind in pm_kinds[:2]},
            "pm_propagation": {"unblocking_edges_total": 0,
                               "unblocking_with_machine_pred": 0,
                               "unblocking_with_wip_pred": 0,
                               "unblocking_chain_reaches_actionable": 0,
                               "unblocking_chain_dead_end": 0,
                               "dead_end_reasons": {}},
            "pm_action_origins": {kind: {"unblocking": 0, "wip": 0}
                                  for kind in pm_kinds[:2]},
            "pm_selected_origins": {kind: {"unblocking": {"generated": 0, "accepted": 0},
                                          "wip": {"generated": 0, "accepted": 0}}
                                    for kind in pm_kinds[:2]},
            "pm_via_unblocking_records": [],
            "ps_moves": {kind: {"generated": 0, "accepted": 0} for kind in ps_kinds},
            "shortage_diagnosis": {"calls": 0, "complete_phi_calls": 0,
                                   "incomplete_phi_calls": 0},
            "hs": {"sizes": [], "ratios": [], "coverages": [], "coverage_violations": 0},
            "relinking": [],
            "cooperation": {"pm_attempts": 0, "pm_usable": 0, "pm_fallback_self": 0,
                            "ps_attempts": 0, "ps_probes": 0, "ps_valid_guide": 0,
                            "ps_fallback_self": 0, "ps_fallback_basic": 0},
            "source_generated": {},
        }
        self._first_source_by_genotype: Dict[tuple, str] = {}
        self._last_pm_move_kind = ""
        self._last_ps_move_kind = ""
        self._last_ps_guide_source = "self"
        self._pm_action_tags: Dict[int, Dict[str, bool]] = {}
        self._last_pm_selected_tags = {"unblocking": False, "wip": False}
        self._last_pm_selected_action: Optional[Move] = None

    def random_individual(self) -> Individual:
        os_seq = self.encoder.generate_random_os()
        ms_list = self.encoder.generate_random_ms()
        return Individual(os_seq, self.encoder.build_ms_map(ms_list),
                          operation_order=tuple(self.encoder.ms_index_order))

    def evaluate(self, ind: Individual) -> Individual:
        if self.n_evaluations >= self.FE_max:
            raise RuntimeError("FE budget exhausted")
        m, schedule, trace, provenance = self.decoder.decode(
            ind.os_seq, ind.ms_map, return_provenance=True)
        stats = self.decoder.analyze(schedule, trace, m)
        ind.makespan, ind.shortage = m, stats["shortage"]["total_shortage_area"]
        ind.schedule, ind.buffer_trace, ind.stats = schedule, trace, stats
        ind.provenance = provenance
        self.n_evaluations += 1
        return ind

    def genotype_key(self, ind: Individual) -> tuple:
        return (tuple(ind.os_seq),
                tuple(ind.ms_map[key] for key in self.encoder.ms_index_order))

    @staticmethod
    def dominates(a: Individual, b: Individual) -> bool:
        return (a.makespan <= b.makespan and a.shortage <= b.shortage and
                (a.makespan < b.makespan or a.shortage < b.shortage))

    def structural_distance(self, x: Individual, y: Individual) -> float:
        keys = self._distance_keys
        rank_y = {key: position for position, key in enumerate(y.os_seq)}
        tree = [0] * (len(keys) + 1)
        job_ranks: Dict[str, List[int]] = {}
        discordant = 0
        for seen, key in enumerate(x.os_seq):
            rank = rank_y[key]
            cursor = rank + 1
            not_greater = 0
            while cursor:
                not_greater += tree[cursor]
                cursor -= cursor & -cursor
            discordant += seen - not_greater

            same_job = job_ranks.setdefault(key[0], [])
            discordant -= len(same_job) - bisect_right(same_job, rank)
            insort(same_job, rank)

            cursor = rank + 1
            while cursor < len(tree):
                tree[cursor] += 1
                cursor += cursor & -cursor

        d_os = discordant / self._distance_pair_count if self._distance_pair_count else 0.0
        d_ms = sum(x.ms_map[k] != y.ms_map[k] for k in keys) / len(keys) if keys else 0.0
        return 0.5 * (d_os + d_ms)

    @staticmethod
    def _crowding(pop: List[Individual]) -> Dict[int, float]:
        result = {id(ind): 0.0 for ind in pop}
        for attr in ("makespan", "shortage"):
            ordered = sorted(pop, key=lambda ind: getattr(ind, attr))
            if len(ordered) <= 2:
                for ind in ordered:
                    result[id(ind)] = math.inf
                continue
            result[id(ordered[0])] = result[id(ordered[-1])] = math.inf
            span = getattr(ordered[-1], attr) - getattr(ordered[0], attr)
            if span:
                for i in range(1, len(ordered)-1):
                    result[id(ordered[i])] += (
                        getattr(ordered[i+1], attr) - getattr(ordered[i-1], attr)) / span
        return result

    def update_archive(self, candidates: Sequence[Individual]) -> None:
        unique: Dict[tuple, Individual] = {}
        for ind in list(self.A) + list(candidates):
            unique.setdefault(self.genotype_key(ind), ind)
        pool = list(unique.values())
        front = [x for x in pool if not any(self.dominates(y, x) for y in pool if y is not x)]
        while len(front) > self.N_A:
            protected = {id(min(front, key=lambda z: z.makespan)),
                         id(min(front, key=lambda z: z.shortage))}
            crowding = self._crowding(front)
            removable = [z for z in front if id(z) not in protected]
            victim = min(removable, key=lambda z: (crowding[id(z)],
                                                   z.makespan, z.shortage,
                                                   self.genotype_key(z)))
            front.remove(victim)
        self.A = front

    def archive_guide(self) -> Optional[Individual]:
        if len(self.A) < 3:
            return None
        extremes = {id(min(self.A, key=lambda z: z.makespan)),
                    id(min(self.A, key=lambda z: z.shortage))}
        interior = [x for x in self.A if id(x) not in extremes]
        if not interior:
            return None
        return self._sparse_tournament(interior)

    def _sparse_tournament(self, candidates: Sequence[Individual]) -> Individual:
        if len(candidates) == 1:
            return candidates[0]
        a, b = self.rng.sample(list(candidates), 2)
        crowd = self._crowding(self.A)
        if crowd[id(a)] == crowd[id(b)]:
            return self.rng.choice((a, b))
        return a if crowd[id(a)] > crowd[id(b)] else b

    def _diverse_initial_population(self, pool: Sequence[Individual], objective: str) -> List[Individual]:
        ordered = sorted(pool, key=lambda z: getattr(z, objective))
        selected: List[Individual] = []
        index = 0
        while len(selected) < self.N:
            value = getattr(ordered[index], objective)
            group = []
            while index < len(ordered) and getattr(ordered[index], objective) == value:
                group.append(ordered[index])
                index += 1
            slots = min(len(group), self.N - len(selected))
            for _ in range(slots):
                if selected:
                    choice = max(group, key=lambda z: (
                        min(self.structural_distance(z, prior) for prior in selected),
                        self.genotype_key(z)))
                else:
                    choice = min(group, key=self.genotype_key)
                selected.append(choice)
                group = [z for z in group if z is not choice]
        return selected

    def initialize(self) -> None:
        pool = []
        seen = set()
        attempts = 0
        while len(pool) < 2 * self.N:
            ind = self.random_individual()
            key = self.genotype_key(ind)
            attempts += 1
            if key in seen:
                if attempts > 10000:
                    raise ValueError("Instance has insufficient distinct genotypes for 2N initial pool")
                continue
            seen.add(key)
            pool.append(self.evaluate(ind))
            self._first_source_by_genotype.setdefault(key, "initial")
        self.P_M = self._diverse_initial_population(pool, "makespan")
        self.P_S = self._diverse_initial_population(pool, "shortage")
        self.update_archive(pool)

    def _insert(self, seq: List[OpKey], key: OpKey, target: int) -> Optional[List[OpKey]]:
        old = seq.index(key)
        shortened = seq[:old] + seq[old+1:]
        target = max(0, min(len(shortened), target))
        if key[1] > 0:
            target = max(target, shortened.index((key[0], key[1]-1)) + 1)
        if key[1]+1 < len(self.operations[key[0]]):
            target = min(target, shortened.index((key[0], key[1]+1)))
        if target == old:
            return None
        return shortened[:target] + [key] + shortened[target:]

    def _move_before(self, ind: Individual, key: OpKey, other: OpKey) -> Optional[Individual]:
        if key == other or ind.os_seq.index(key) < ind.os_seq.index(other):
            return None
        target = ind.os_seq.index(other)
        seq = self._insert(ind.os_seq, key, target)
        return (Individual(seq, ind.ms_map.copy(), operation_order=ind.operation_order)
                if seq is not None and seq.index(key) < seq.index(other) else None)

    def _move_after(self, ind: Individual, key: OpKey, other: OpKey) -> Optional[Individual]:
        if key == other or ind.os_seq.index(key) > ind.os_seq.index(other):
            return None
        target = ind.os_seq.index(other)
        seq = self._insert(ind.os_seq, key, target)
        return (Individual(seq, ind.ms_map.copy(), operation_order=ind.operation_order)
                if seq is not None and seq.index(key) > seq.index(other) else None)

    def basic_mutation(self, ind: Individual) -> Individual:
        positions = {key: i for i, key in enumerate(ind.os_seq)}
        os_windows = {}
        for key, pos in positions.items():
            lower = positions[(key[0], key[1]-1)] + 1 if key[1] else 0
            upper = (positions[(key[0], key[1]+1)] - 1
                     if key[1]+1 < len(self.operations[key[0]]) else len(ind.os_seq)-1)
            if lower < pos or pos < upper:
                os_windows[key] = (lower, upper)
        ms_moves = [key for key in self.encoder.ms_index_order
                    if len(self.operations[key[0]][key[1]]["machines"]) >= 2]
        options = (["os"] if os_windows else []) + (["ms"] if ms_moves else [])
        if not options:
            return Individual(ind.os_seq[:], ind.ms_map.copy(), operation_order=ind.operation_order)
        if self.rng.choice(options) == "ms":
            key = self.rng.choice(ms_moves)
            machines = [m for m in self.operations[key[0]][key[1]]["machines"]
                        if m != ind.ms_map[key]]
            child = Individual(ind.os_seq[:], ind.ms_map.copy(), operation_order=ind.operation_order)
            child.ms_map[key] = self.rng.choice(machines)
            return child
        key = self.rng.choice(list(os_windows))
        lower, upper = os_windows[key]
        old = positions[key]
        target = self.rng.choice([pos for pos in range(lower, upper+1) if pos != old])
        return Individual(self._insert(ind.os_seq, key, target), ind.ms_map.copy(),
                          operation_order=ind.operation_order)

    def _backtrace(self, provenance: DecodeProvenance, anchors: Set[int]) -> Tuple[Set[int], list, Dict[int, int]]:
        incoming: Dict[int, list] = {}
        for edge in provenance.active_dependencies:
            incoming.setdefault(edge.dst_event_id, []).append(edge)
        visited = set(anchors)
        depths = {node: 0 for node in anchors}
        stack = deque(anchors)
        used = []
        while stack:
            node = stack.popleft()
            for edge in incoming.get(node, []):
                used.append(edge)
                source = edge.src_event_id
                depth = depths[node] + 1
                if source not in depths or depth < depths[source]:
                    depths[source] = depth
                if source not in visited:
                    visited.add(source)
                    stack.append(source)
        return visited, used, depths

    def makespan_actions(self, ind: Individual) -> List[Move]:
        """Trace all active dependencies; extract only machine and MS actions."""
        prov = ind.provenance
        terminals = {event.event_id for event in prov.events.values()
                     if event.kind == "release" and event.time == ind.makespan}
        nodes, edges, depths = self._backtrace(prov, terminals)
        incoming: Dict[int, list] = {}
        for edge in edges:
            incoming.setdefault(edge.dst_event_id, []).append(edge)

        def ancestors(start_ids: Set[int]) -> Set[int]:
            reached = set(start_ids)
            stack = list(start_ids)
            while stack:
                for edge in incoming.get(stack.pop(), []):
                    if edge.src_event_id not in reached:
                        reached.add(edge.src_event_id)
                        stack.append(edge.src_event_id)
            return reached

        unblocking_edges = [edge for edge in edges if edge.kind == "unblocking"]
        via_unblocking = ancestors({edge.src_event_id for edge in unblocking_edges})
        via_wip = ancestors({edge.src_event_id for edge in edges if edge.kind == "wip"})
        actions: List[Move] = []
        action_reach_nodes: Dict[int, Set[int]] = {}
        seen_machine_edges = set()
        for edge in edges:
            src, dst = prov.events[edge.src_event_id], prov.events[edge.dst_event_id]
            if edge.kind == "machine":
                u, o = (src.job, src.op_idx), (dst.job, dst.op_idx)
                relation = (u, o, edge.machine)
                if relation not in seen_machine_edges:
                    seen_machine_edges.add(relation)
                    if self._move_before(ind, o, u) or self._move_after(ind, u, o):
                        move = Move("machine_priority", o, u, depth=depths.get(dst.event_id, 0))
                        actions.append(move)
                        action_reach_nodes[id(move)] = {edge.dst_event_id}
        machine_depths: Dict[OpKey, int] = {}
        op_nodes: Dict[OpKey, Set[int]] = {}
        for node in nodes:
            event = prov.events[node]
            key = (event.job, event.op_idx)
            op_nodes.setdefault(key, set()).add(node)
            machines = self.operations[key[0]][key[1]]["machines"]
            if len(machines) > 1:
                machine_depths[key] = min(machine_depths.get(key, math.inf), depths[node])
        for key, depth in machine_depths.items():
            move = Move("machine_reassign", key, depth=depth)
            actions.append(move)
            action_reach_nodes[id(move)] = op_nodes[key]

        self._pm_action_tags = {
            id(move): {"unblocking": bool(action_reach_nodes[id(move)] & via_unblocking),
                       "wip": bool(action_reach_nodes[id(move)] & via_wip)}
            for move in actions
        }
        propagation = self.diagnostics["pm_propagation"]
        for edge in unblocking_edges:
            propagation["unblocking_edges_total"] += 1
            predecessors = incoming.get(edge.src_event_id, [])
            propagation["unblocking_with_machine_pred"] += int(
                any(pred.kind == "machine" for pred in predecessors))
            propagation["unblocking_with_wip_pred"] += int(
                any(pred.kind == "wip" for pred in predecessors))
            chain = ancestors({edge.src_event_id})
            if any(action_reach_nodes[id(move)] & chain for move in actions):
                propagation["unblocking_chain_reaches_actionable"] += 1
                continue
            propagation["unblocking_chain_dead_end"] += 1
            if any(pred.kind == "machine" and pred.dst_event_id in chain for pred in edges):
                reason = "machine_relation_os_move_illegal"
            elif any(pred.kind == "wip" and pred.dst_event_id in chain for pred in edges):
                reason = "only_wip_chain_no_actionable"
            elif any(prov.events[node].op_idx == 0 for node in chain):
                reason = "first_operation_or_source_boundary"
            else:
                reason = "no_alternative_machine_or_other"
            reasons = propagation["dead_end_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
        return actions

    def vary_makespan(self, seed: Individual) -> Individual:
        self._last_pm_selected_tags = {"unblocking": False, "wip": False}
        self._last_pm_selected_action = None
        if self.rng.random() < self.p_mut:
            self._last_pm_move_kind = "basic_mutation"
            return self.basic_mutation(seed)
        child = self._vary_makespan_specialized(seed)
        return child if child is not None else self.basic_mutation(seed)

    def _vary_makespan_specialized(self, seed: Individual) -> Optional[Individual]:
        self._last_pm_selected_tags = {"unblocking": False, "wip": False}
        self._last_pm_selected_action = None
        actions = self.makespan_actions(seed)
        if not actions:
            self._last_pm_move_kind = "fallback_basic_mutation"
            return None
        counts = {kind: sum(action.kind == kind for action in actions)
                  for kind in self.diagnostics["pm_selected_depths"]}
        self.diagnostics["pm_action_sets"].append({**counts, "total_actions": len(actions)})
        origins = self.diagnostics["pm_action_origins"]
        for candidate in actions:
            for tag, present in self._pm_action_tags.get(id(candidate),
                                                        {"unblocking": False, "wip": False}).items():
                origins[candidate.kind][tag] += int(present)
        action = self.rng.choice(actions)
        self.diagnostics["pm_selected_depths"][action.kind].append(action.depth)
        self._last_pm_selected_tags = self._pm_action_tags.get(
            id(action), {"unblocking": False, "wip": False}).copy()
        self._last_pm_selected_action = action
        if action.kind == "machine_priority":
            child = self._move_before(seed, action.key, action.other)
            moved_key = action.key
            if child is None:
                child = self._move_after(seed, action.other, action.key)
                moved_key = action.other
            if child is not None:
                self._last_pm_move_kind = "machine_priority"
                self.machine_priority_move_spans.append(
                    abs(seed.os_seq.index(moved_key) - child.os_seq.index(moved_key)))
                return child
            self._last_pm_move_kind = "fallback_basic_mutation"
            return None
        self._last_pm_move_kind = "machine_reassign"
        child = Individual(seed.os_seq[:], seed.ms_map.copy(), operation_order=seed.operation_order)
        machines = [m for m in self.operations[action.key[0]][action.key[1]]["machines"]
                    if m != seed.ms_map[action.key]]
        child.ms_map[action.key] = self.rng.choice(machines)
        return child

    def _generate_pm_child(self, cooperation: bool) -> Tuple[Individual, Individual, str]:
        self._last_pm_selected_tags = {"unblocking": False, "wip": False}
        self._last_pm_selected_action = None
        if self.pm_specialized_kind == "BasicMutation":
            seed = self.tournament(self.P_M, "makespan")
            return seed, self.vary_makespan(seed), "P_M_self"
        if self.rng.random() < self.p_mut:
            seed = self.tournament(self.P_M, "makespan")
            self._last_pm_move_kind = "basic_mutation"
            return seed, self.basic_mutation(seed), "P_M_self"
        archive_pick = None
        if cooperation and self.pm_archive_guidance_enabled:
            counts = self.diagnostics["cooperation"]
            counts["pm_attempts"] += 1
            archive_pick = self.archive_guide()
            counts["pm_usable" if archive_pick is not None else "pm_fallback_self"] += 1
        seed = archive_pick if archive_pick is not None else self.tournament(self.P_M, "makespan")
        child = self._vary_makespan_specialized(seed)
        if child is None:
            # Even an unsuccessful archive-assisted MGS falls back to a specialist parent.
            if archive_pick is not None:
                seed = self.tournament(self.P_M, "makespan")
            return seed, self.basic_mutation(seed), "P_M_self"
        return seed, child, "P_M_archive" if archive_pick is not None else "P_M_self"

    def diagnose_shortage(self, ind: Individual) -> ShortageDiagnosis:
        self.diagnostics["shortage_diagnosis"]["calls"] += 1
        prov, stats = ind.provenance, ind.stats["shortage"]
        intervals = []
        phi: Dict[OpKey, float] = {}
        all_nodes: Set[int] = set()
        all_edges = []
        unanchored = 0
        for bid, events in prov.buffer_events.items():
            start = stats["per_buffer_active_start"][bid]
            end = stats["per_buffer_active_end"][bid]
            if end <= start:
                continue
            low = stats["per_buffer_low_wip"][bid]
            ordered = sorted(events, key=lambda ev: (ev.time, ev.event_id))
            times = sorted({start, end} | {ev.time for ev in ordered if start <= ev.time <= end})
            # Match decoder._reset_buffers; future events must not seed the state.
            initial_content = self.buffers[bid].get("init_content") or []
            level = len(set(initial_content))
            event_index = 0
            while event_index < len(ordered) and ordered[event_index].time < start:
                level = ordered[event_index].level_after
                event_index += 1
            pre_start_boundary = ((ordered[event_index-1].time, ordered[event_index-1].event_id)
                                  if event_index else (start, -math.inf))
            interval_start = None
            opening_boundary = None
            area = 0.0
            for i, time in enumerate(times):
                gap_before_events = max(0, low - level)
                latest_opening_event = None
                while event_index < len(ordered) and ordered[event_index].time == time:
                    event = ordered[event_index]
                    before = max(0, low - level)
                    level = event.level_after
                    if before == 0 and max(0, low - level) > 0:
                        latest_opening_event = (event.time, event.event_id)
                    event_index += 1
                if i == len(times)-1:
                    gap = 0
                else:
                    gap = max(0, low - level)
                if gap and interval_start is None:
                    interval_start = time
                    opening_boundary = (pre_start_boundary if gap_before_events > 0
                                        else latest_opening_event or (time, -math.inf))
                if gap:
                    area += gap * (times[i+1] - time)
                elif interval_start is not None:
                    left, right = interval_start, time
                    anchors = set()
                    for ev in ordered:
                        if ev.cause_event_id is None:
                            continue
                        before = max(0, low - ev.level_before)
                        after = max(0, low - ev.level_after)
                        if after > before and left <= ev.time < right:
                            anchors.add(ev.cause_event_id)
                        elif after < before and left < ev.time <= right:
                            anchors.add(ev.cause_event_id)
                    nodes, edges, _ = self._backtrace(prov, anchors)
                    ops = {(prov.events[node].job, prov.events[node].op_idx)
                           for node in nodes if node in prov.events}
                    intervals.append((bid, left, right, area, ops))
                    if ops:
                        share = area / len(ops)
                        for op in ops:
                            phi[op] = phi.get(op, 0.0) + share
                    else:
                        unanchored += 1
                    all_nodes.update(nodes)
                    all_edges.extend(edges)
                    interval_start, opening_boundary, area = None, None, 0.0
        if unanchored:
            self.unanchored_shortage_intervals += unanchored
            warnings.warn(f"{unanchored} shortage interval(s) have no factual operation anchor")
        diagnosis_counts = self.diagnostics["shortage_diagnosis"]
        if not unanchored and math.isclose(sum(phi.values()), ind.shortage,
                                          rel_tol=0.0, abs_tol=1e-9):
            diagnosis_counts["complete_phi_calls"] += 1
        else:
            diagnosis_counts["incomplete_phi_calls"] += 1
        return ShortageDiagnosis(intervals, phi, all_nodes, all_edges, unanchored)

    def high_influence_set(self, ind: Individual, diagnosis: ShortageDiagnosis) -> Set[OpKey]:
        if not ind.shortage:
            return set()
        ordered = sorted(diagnosis.phi, key=lambda k: (-diagnosis.phi[k], k))
        selected, mass = set(), 0.0
        for key in ordered:
            selected.add(key)
            mass += diagnosis.phi[key]
            if mass / ind.shortage >= self.rho:
                break
        return selected

    def _machine_shortage_evidence(self, seed: Individual,
                                   diagnosis: ShortageDiagnosis) -> frozenset:
        machine_ops = set()
        for edge in diagnosis.influence_dependencies:
            if edge.kind == "machine":
                for node in (edge.src_event_id, edge.dst_event_id):
                    ev = seed.provenance.events[node]
                    machine_ops.add((ev.job, ev.op_idx))
        return frozenset(machine_ops)

    def _os_discrepancy(self, current: Individual, guide: Individual,
                        key: OpKey, positions=None, guide_positions=None) -> int:
        if positions is None:
            positions = {op: i for i, op in enumerate(current.os_seq)}
        if guide_positions is None:
            guide_positions = {op: i for i, op in enumerate(guide.os_seq)}
        return sum((positions[key] < positions[other]) !=
                   (guide_positions[key] < guide_positions[other])
                   for other in current.os_seq if other[0] != key[0])

    def _legal_os_insertion_bounds(self, seq: List[OpKey], key: OpKey,
                                    positions) -> Tuple[int, int]:
        old = positions[key]
        lower, upper = 0, len(seq) - 1
        if key[1] > 0:
            predecessor = positions[(key[0], key[1] - 1)]
            lower = predecessor - (predecessor > old) + 1
        if key[1] + 1 < len(self.operations[key[0]]):
            successor = positions[(key[0], key[1] + 1)]
            upper = successor - (successor > old)
        return lower, upper

    def _best_os_relink_action(self, current: Individual, guide: Individual,
                               key: OpKey, positions=None,
                               guide_positions=None) -> Optional[OSRelinkAction]:
        if positions is None:
            positions = {op: i for i, op in enumerate(current.os_seq)}
        if guide_positions is None:
            guide_positions = {op: i for i, op in enumerate(guide.os_seq)}
        before = self._os_discrepancy(current, guide, key, positions, guide_positions)
        if before == 0:
            return None
        old = positions[key]
        shortened = current.os_seq[:old] + current.os_seq[old + 1:]
        lower, upper = self._legal_os_insertion_bounds(current.os_seq, key, positions)
        # Sweep every global gap. Crossing one incomparable operation changes D by one.
        distance = sum(guide_positions[key] > guide_positions[other]
                       for other in shortened if other[0] != key[0])
        best = None
        for gap in range(len(shortened) + 1):
            if lower <= gap <= upper:
                score = (distance, abs(gap - old), gap)
                if best is None or score < best:
                    best = score
            if gap < len(shortened):
                other = shortened[gap]
                if other[0] != key[0]:
                    distance += (1 if guide_positions[key] < guide_positions[other] else -1)
        after, displacement, target = best
        if after >= before:
            return None
        return OSRelinkAction(key, target, before, after, displacement)

    def _build_relink_units(self, seed: Individual, guide: Individual,
                            high: Set[OpKey], diagnosis: ShortageDiagnosis,
                            evidence=None) -> Tuple[RelinkUnit, ...]:
        if evidence is None:
            evidence = self._machine_shortage_evidence(seed, diagnosis)
        positions = {op: i for i, op in enumerate(seed.os_seq)}
        guide_positions = {op: i for i, op in enumerate(guide.os_seq)}
        units = []
        for key in sorted(high):
            if self._best_os_relink_action(seed, guide, key, positions, guide_positions) is not None:
                units.append(RelinkUnit(key, "OS"))
            if key in evidence and seed.ms_map[key] != guide.ms_map[key]:
                units.append(RelinkUnit(key, "MS"))
        # Fixed unit priority: phi descending, operation key, then OS before MS.
        return tuple(sorted(units, key=lambda u: (-diagnosis.phi.get(u.key, 0.0),
                                                 u.key, 0 if u.kind == "OS" else 1)))

    def choose_guide(self, x: Individual, pool: Sequence[Individual],
                     high: Set[OpKey], diagnosis: ShortageDiagnosis,
                     evidence=None) -> Optional[Individual]:
        eligible = []
        for y in pool:
            if y is x or y.shortage > x.shortage:
                continue
            if self._build_relink_units(x, y, high, diagnosis, evidence):
                eligible.append(y)
        better = [y for y in eligible if y.shortage < x.shortage]
        candidates = better or eligible
        # Preserve population order within the preferred shortage eligibility group.
        return candidates[0] if candidates else None

    def choose_archive_guide_for_shortage(self, x: Individual,
                                          diagnosis: ShortageDiagnosis,
                                          high: Set[OpKey], evidence=None) -> Optional[Individual]:
        if len(self.A) < 3:
            return None
        extremes = {id(min(self.A, key=lambda z: z.makespan)),
                    id(min(self.A, key=lambda z: z.shortage))}
        valid = [y for y in self.A
                 if id(y) not in extremes and y is not x
                 and self._build_relink_units(x, y, high, diagnosis, evidence)]
        return self._sparse_tournament(valid) if valid else None

    def vary_shortage(self, x: Individual, guide_pool: Sequence[Individual],
                      archive_assisted: bool = False) -> Individual:
        self._last_ps_guide_source = "self"
        if self.rng.random() < self.p_mut:
            self._last_ps_move_kind = "basic_mutation"
            return self.basic_mutation(x)
        if x.shortage == 0:
            self._last_ps_move_kind = "fallback_basic_mutation_zero_shortage"
            return self.basic_mutation(x)
        diagnosis = self.diagnose_shortage(x)
        phi_mass = sum(diagnosis.phi.values())
        if diagnosis.unanchored_intervals or not math.isclose(
                phi_mass, x.shortage, rel_tol=0.0, abs_tol=1e-9):
            if not diagnosis.unanchored_intervals:
                self.phi_mass_mismatch_count += 1
                warnings.warn(f"Shortage influence mass mismatch: phi={phi_mass}, shortage={x.shortage}")
            self._last_ps_move_kind = "fallback_basic_mutation_phi_invalid"
            return self.basic_mutation(x)
        high = self.high_influence_set(x, diagnosis)
        evidence = self._machine_shortage_evidence(x, diagnosis)
        hs_stats = self.diagnostics["hs"]
        size = len(high)
        coverage = sum(diagnosis.phi[key] for key in high) / x.shortage
        hs_stats["sizes"].append(size)
        hs_stats["ratios"].append(size / len(self.encoder.ms_index_order))
        hs_stats["coverages"].append(coverage)
        if coverage + 1e-12 < self.rho:
            hs_stats["coverage_violations"] += 1
        if archive_assisted:
            self.diagnostics["cooperation"]["ps_attempts"] += 1
            self.diagnostics["cooperation"]["ps_probes"] += 1
        guide = (self.choose_archive_guide_for_shortage(x, diagnosis, high, evidence)
                 if archive_assisted else None)
        if archive_assisted:
            if guide is None:
                self.diagnostics["cooperation"]["ps_fallback_self"] += 1
            else:
                self.diagnostics["cooperation"]["ps_valid_guide"] += 1
                self._last_ps_guide_source = "archive"
        if guide is None:
            guide = self.choose_guide(x, guide_pool, high, diagnosis, evidence)
        if guide is None:
            if archive_assisted:
                self.diagnostics["cooperation"]["ps_fallback_basic"] += 1
            self._last_ps_move_kind = "fallback_basic_mutation_no_guide"
            return self.basic_mutation(x)
        initial_units = self._build_relink_units(x, guide, high, diagnosis, evidence)
        if not initial_units:
            self._last_ps_move_kind = "fallback_basic_mutation_no_guide"
            return self.basic_mutation(x)
        self._last_ps_move_kind = "shortage_relink"
        steps = max(1, math.ceil(self.eta * len(initial_units)))
        fe_before = self.n_evaluations
        child = Individual(x.os_seq[:], x.ms_map.copy(), operation_order=x.operation_order)
        used_units = set()
        applied = []
        skipped = []
        guide_positions = {op: i for i, op in enumerate(guide.os_seq)}
        while len(used_units) < steps:
            positions = {op: i for i, op in enumerate(child.os_seq)}
            for unit in initial_units:
                if unit in used_units:
                    continue
                if unit.kind == "OS":
                    action = self._best_os_relink_action(child, guide, unit.key, positions, guide_positions)
                    if action is None:
                        skipped.append((unit.key, unit.kind))
                        continue
                    child.os_seq = self._insert(child.os_seq, unit.key, action.target_index)
                    applied.append({"key": unit.key, "kind": unit.kind,
                                    "target_index": action.target_index,
                                    "d_before": action.d_before, "d_after": action.d_after,
                                    "displacement": action.displacement})
                else:
                    machine = guide.ms_map[unit.key]
                    if child.ms_map[unit.key] == machine:
                        skipped.append((unit.key, unit.kind))
                        continue
                    if machine not in self.operations[unit.key[0]][unit.key[1]]["machines"]:
                        raise ValueError(f"Illegal guide machine for {unit.key}: {machine}")
                    child.ms_map[unit.key] = machine
                    applied.append({"key": unit.key, "kind": unit.kind, "machine": machine})
                used_units.add(unit)
                break
            else:
                break
        os_applied = [item for item in applied if item["kind"] == "OS"]
        self.diagnostics["relinking"].append({
            "seed_os": x.os_seq[:], "guide_os": guide.os_seq[:], "final_os": child.os_seq[:],
            "guide_ms": [(key, guide.ms_map[key]) for key in sorted(high)],
            "final_ms": [(key, child.ms_map[key]) for key in sorted(high)],
            "high_operations": sorted(high), "high_count": len(high),
            "fixed_phi": [(key, diagnosis.phi[key]) for key in sorted(diagnosis.phi)],
            "machine_evidence": sorted(evidence),
            "initial_units": [(unit.key, unit.kind) for unit in initial_units],
            "initial_unit_count": len(initial_units),
            "initial_os_unit_count": sum(unit.kind == "OS" for unit in initial_units),
            "initial_ms_unit_count": sum(unit.kind == "MS" for unit in initial_units),
            "K": steps, "applied_unit_count": len(applied),
            "applied_os_units": len(os_applied),
            "applied_ms_units": len(applied) - len(os_applied),
            "skipped_unactionable_units": len(skipped), "skipped_units": skipped,
            "applied_units": applied,
            "total_os_discrepancy_reduction": sum(item["d_before"] - item["d_after"]
                                                  for item in os_applied),
            "fe_before": fe_before, "fe_after": self.n_evaluations,
        })
        return child

    def tournament(self, population: Sequence[Individual], objective: str) -> Individual:
        if len(population) == 1:
            return population[0]
        a, b = self.rng.sample(list(population), 2)
        if getattr(a, objective) != getattr(b, objective):
            return a if getattr(a, objective) < getattr(b, objective) else b
        novelty_a = min(self.structural_distance(a, z) for z in population if z is not a)
        novelty_b = min(self.structural_distance(b, z) for z in population if z is not b)
        if novelty_a != novelty_b:
            return a if novelty_a > novelty_b else b
        return self.rng.choice((a, b))

    def similarity_replace(self, population: List[Individual], child: Individual,
                           objective: str) -> bool:
        nearest = min(population, key=lambda z: self.structural_distance(child, z))
        if getattr(child, objective) < getattr(nearest, objective):
            population[population.index(nearest)] = child
            return True
        return False

    def run(self, diagnostic_callback: Optional[Callable[["WIPGraphDualPopulation"], None]] = None) -> List[Individual]:
        if not self.P_M:
            self.initialize()
        if diagnostic_callback is not None:
            diagnostic_callback(self)
        while self.n_evaluations < self.FE_max:
            self.round += 1
            available = min(2 * self.N, self.FE_max - self.n_evaluations)
            m_count = available // 2 + (available % 2 if self.round % 2 else 0)
            s_count = available - m_count
            cooperation = self.round % self.T_coop == 0
            offspring = []
            for side, count in (("M", m_count), ("S", s_count)):
                for _ in range(count):
                    if side == "M":
                        seed, child, source = self._generate_pm_child(cooperation)
                    else:
                        seed = self.tournament(self.P_S, "shortage")
                        child = self.vary_shortage(seed, self.P_S,
                                                   archive_assisted=cooperation and self.ps_archive_guidance_enabled)
                        source = ("P_S_archive" if self._last_ps_guide_source == "archive"
                                  else "P_S_self")
                    self.evaluate(child)
                    genotype = self.genotype_key(child)
                    self._first_source_by_genotype.setdefault(genotype, source)
                    generated = self.diagnostics["source_generated"]
                    generated[source] = generated.get(source, 0) + 1
                    offspring.append(child)
                    accepted = self.similarity_replace(self.P_M if side == "M" else self.P_S,
                                                       child, "makespan" if side == "M" else "shortage")
                    group = self.diagnostics["pm_moves" if side == "M" else "ps_moves"]
                    kind = self._last_pm_move_kind if side == "M" else self._last_ps_move_kind
                    group[kind]["generated"] += 1
                    group[kind]["accepted"] += int(accepted)
                    if side == "M" and kind in self.diagnostics["pm_selected_origins"]:
                        for tag, present in self._last_pm_selected_tags.items():
                            if present:
                                origin = self.diagnostics["pm_selected_origins"][kind][tag]
                                origin["generated"] += 1
                                origin["accepted"] += int(accepted)
                        if self._last_pm_selected_tags["unblocking"]:
                            action = self._last_pm_selected_action
                            before = {(rec["job"], rec["op"]): rec for rec in seed.schedule}
                            after = {(rec["job"], rec["op"]): rec for rec in child.schedule}
                            signature = lambda rec: (rec["machine"], rec["start"], rec["end"],
                                                     rec.get("release", rec["end"]))
                            changed = any(signature(before[key]) != signature(after[key])
                                          for key in before)
                            target_before, target_after = before[action.key], after[action.key]
                            start_id = child.provenance.op_events[action.key]["start"]
                            limiting = sorted({edge.kind for edge in child.provenance.active_dependencies
                                               if edge.dst_event_id == start_id
                                               and edge.kind in ("machine", "wip")})
                            self.diagnostics["pm_via_unblocking_records"].append({
                                "fe_index": self.n_evaluations, "kind": kind,
                                "key": action.key, "other": action.other,
                                "machine_before": seed.ms_map[action.key],
                                "machine_after": child.ms_map[action.key],
                                "start_before": target_before["start"],
                                "start_after": target_after["start"],
                                "release_before": target_before.get("release", target_before["end"]),
                                "release_after": target_after.get("release", target_after["end"]),
                                "makespan_before": seed.makespan,
                                "makespan_after": child.makespan,
                                "schedule_changed": changed, "accepted": bool(accepted),
                                "limiting_causes_after": limiting,
                            })
            self.update_archive(offspring)
            if diagnostic_callback is not None:
                diagnostic_callback(self)
        return self.A

    def get_pareto_front(self) -> List[Individual]:
        return self.A[:]
