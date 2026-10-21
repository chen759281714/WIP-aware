"""H-LNS dual population with in-module OS/MS joint repair.

The original steady-state MGS, WIP evaluation and archive flow are shared;
P_S uses a single H-guided destroy/repair candidate, never SIS relinking.
"""

from __future__ import annotations

import math
import random
import warnings
from bisect import bisect_left, bisect_right, insort
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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


@dataclass
class ShortageDiagnosis:
    intervals: List[Tuple[str, int, int, float, Set[OpKey]]]
    phi: Dict[OpKey, float]
    influence_events: Set[int]
    influence_dependencies: list
    unanchored_intervals: int = 0


class HLNSRepairError(ValueError):
    """Invalid genotype or broken repair invariant; never a retry signal."""

def validate_priority_os(os_seq: Sequence[OpKey]) -> None:
    seen = set()
    expected: Dict[str, int] = {}
    for op in os_seq:
        if (not isinstance(op, tuple) or len(op) != 2 or
                not isinstance(op[1], int) or isinstance(op[1], bool) or op in seen):
            raise HLNSRepairError("OS requires unique explicit operation tuples")
        job, index = op
        if index != expected.get(job, 0):
            raise HLNSRepairError("OS violates fixed job precedence")
        expected[job] = index + 1
        seen.add(op)

def parent_deviation_probability(size: int) -> float:
    """q controls repair choices, not p_mut or a required changed fraction."""
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("Relaxed set must be nonempty")
    return 1.0 / (1.0 + math.sqrt(size))

def _validate_q(q: float) -> None:
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be finite and in [0,1]")

def alternative_probabilities(ranks: Sequence[int]) -> List[float]:
    """Ready ranks sorted in parent order; alternatives exclude ranks[0]."""
    if len(ranks) < 2 or list(ranks) != sorted(set(ranks)):
        raise ValueError("Expected at least two distinct sorted ready ranks")
    weights = [1.0 / (1 + rank - ranks[0]) for rank in ranks[1:]]
    total = sum(weights)
    return [weight / total for weight in weights]

def _weighted_choice(alternatives: Sequence[int], probabilities: Sequence[float], rng) -> int:
    draw = rng.random()
    cumulative = 0.0
    for rank, probability in zip(alternatives, probabilities):
        cumulative += probability
        if draw < cumulative:
            return rank
    return alternatives[-1]

def biased_topological_os_repair(
    parent_os: Sequence[OpKey], relaxed_set: Iterable[OpKey], q: float, rng,
    *, parent_bias: bool = True,
) -> List[OpKey]:
    """Rebuild all O under job edges and the non-relaxed parent projection.

    Only adjacent F genes form skeleton edges. No stage blocks, original
    H/F priority edges, or intermediate decoding are added.
    """
    _validate_q(q)
    validate_priority_os(parent_os)
    high = set(relaxed_set)
    if not high <= set(parent_os):
        raise HLNSRepairError("Unknown relaxed operation")
    n = len(parent_os)
    successors: List[List[int]] = [[] for _ in range(n)]
    indegree = [0] * n
    edges: Set[Tuple[int, int]] = set()

    def edge(src: int, dst: int) -> None:
        if (src, dst) not in edges:
            edges.add((src, dst))
            successors[src].append(dst)
            indegree[dst] += 1

    previous_job: Dict[str, int] = {}
    previous_fixed = None
    for rank, op in enumerate(parent_os):
        if op[0] in previous_job:
            edge(previous_job[op[0]], rank)
        previous_job[op[0]] = rank
        if op not in high:
            if previous_fixed is not None:
                edge(previous_fixed, rank)
            previous_fixed = rank
    ready = [rank for rank in range(n) if indegree[rank] == 0]
    result = []
    while ready:
        if len(ready) == 1:
            selected = ready[0]
        elif not parent_bias:
            selected = rng.choice(ready)
        elif rng.random() < q:
            selected = _weighted_choice(ready[1:], alternative_probabilities(ready), rng)
        else:
            selected = ready[0]
        ready.remove(selected)
        result.append(parent_os[selected])
        for nxt in successors[selected]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                insort(ready, nxt)
    if len(result) != n:
        raise HLNSRepairError("Topological repair did not complete")
    validate_priority_os(result)
    if set(result) != set(parent_os) or [op for op in result if op not in high] != [op for op in parent_os if op not in high]:
        raise HLNSRepairError("Repair changed coverage or fixed OS skeleton")
    return result

def biased_machine_repair(
    parent_ms: Mapping[OpKey, str], relaxed_set: Iterable[OpKey],
    legal_machines: Mapping[OpKey, Sequence[str]], q: float, rng,
    *, parent_bias: bool = True,
) -> Dict[OpKey, str]:
    """Use stable encoder-operation and instance-machine order, never speeds."""
    _validate_q(q)
    high = set(relaxed_set)
    if set(parent_ms) != set(legal_machines) or not high <= set(parent_ms):
        raise HLNSRepairError("MS coverage/relaxed operations mismatch")
    child = dict(parent_ms)
    for op, machines in legal_machines.items():
        if not machines or len(set(machines)) != len(machines) or parent_ms[op] not in machines:
            raise HLNSRepairError("Illegal parent machine or legal-machine list")
        if op not in high or len(machines) == 1:
            continue
        if not parent_bias:
            child[op] = rng.choice(list(machines))
        elif rng.random() < q:
            child[op] = rng.choice([machine for machine in machines if machine != parent_ms[op]])
    return child

def restricted_atomic_fallback(engine, parent: Individual, relaxed_set: Iterable[OpKey],
                               rng, *, component: Optional[str] = None) -> Optional[Individual]:
    """Filter the original BasicMutation choice stream, without copying it.

    With both components this is genotype/RNG-identical to the previously
    verified RestrictedBasicMutation. A component mask only serves single-
    component ablations. None means exhausted, not a request to resample.
    """
    if component not in (None, "OS", "MS"):
        raise ValueError("Unknown fallback component")
    high = set(relaxed_set)
    if not high <= set(parent.os_seq):
        raise HLNSRepairError("Unknown relaxed operation")
    positions = {op: i for i, op in enumerate(parent.os_seq)}
    os_available = False
    ms_available = False
    for op in high:
        pos = positions[op]
        lower = positions[(op[0], op[1]-1)] + 1 if op[1] else 0
        upper = (positions[(op[0], op[1]+1)] - 1
                 if op[1]+1 < len(engine.operations[op[0]]) else len(parent.os_seq)-1)
        os_available |= lower < pos or pos < upper
        ms_available |= len(engine.operations[op[0]][op[1]]["machines"]) >= 2
    allowed = ((["os"] if os_available and component != "MS" else []) +
               (["ms"] if ms_available and component != "OS" else []))
    if not allowed:
        return None

    class RestrictedChoices:
        calls = 0

        def choice(self, options):
            self.calls += 1
            if self.calls == 1:
                return rng.choice([kind for kind in options if kind in allowed])
            if self.calls == 2:
                return rng.choice([op for op in options if op in high])
            return rng.choice(options)

    original = engine.rng
    stream = RestrictedChoices()
    try:
        engine.rng = stream
        child = engine.basic_mutation(parent)
    finally:
        engine.rng = original
    if stream.calls != 3:
        raise HLNSRepairError("Original BasicMutation choice contract changed")
    return child

class HLNSDiagnosisError(ValueError):
    """A broken verified diagnosis invariant, not an ordinary global fallback."""


class WIPGraphDualPopulation:
    """H-LNS P_S and MGS P_M with the original shared Pareto archive."""

    def __init__(self, operations, buffers, *, variant=None, collect_hlns_stats=True,
                 N: int = 100, N_A: int = 200, rho: float = 0.8,
                 p_mut: float = 0.1, T_coop: int = 10,
                 FE_max: int = 10000, seed: Optional[int] = None):
        if variant is None:
            raise ValueError("The previous Full implementation is no longer provided by "
                             "this module. Specify an explicit H-LNS variant.")
        if type(collect_hlns_stats) is not bool:
            raise TypeError("collect_hlns_stats must be boolean")
        if not isinstance(variant, str) or variant != self.algorithm_variant:
            raise ValueError(f"{type(self).__name__} requires variant={self.algorithm_variant!r}; "
                             "instantiate the corresponding independent subclass for an ablation")
        if N < 1 or N_A < 2 or FE_max < 2 * N or T_coop < 1:
            raise ValueError("Require N>=1, N_A>=2, FE_max>=2N, T_coop>=1")
        if not (0 < rho <= 1 and 0 <= p_mut <= 1):
            raise ValueError("Invalid rho or p_mut")
        self.operations, self.buffers = operations, buffers
        self.N, self.N_A = N, N_A
        self.rho, self.p_mut = rho, p_mut
        self.T_coop, self.FE_max = T_coop, FE_max
        self.rng = random.Random(seed)
        self.encoder = Encoder(operations, rng=self.rng)
        self.decoder = StageBufferWIPScheduler(operations, buffers)
        self._distance_keys = tuple(self.encoder.ms_index_order)
        self._distance_rank_cache: Dict[tuple, Dict[OpKey, int]] = {}
        self._os_pair_cache: Dict[tuple, int] = {}
        self._dependency_index_cache: Dict[int, tuple] = {}
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
        self.machine_priority_move_spans: List[int] = []
        pm_kinds = ("machine_priority", "machine_reassign", "basic_mutation",
                    "fallback_basic_mutation")
        ps_kinds = ("basic_mutation", "hlns_joint")
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
            "cooperation": {"pm_attempts": 0, "pm_usable": 0, "pm_fallback_self": 0},
            "source_generated": {},
        }
        self._first_source_by_genotype: Dict[tuple, str] = {}
        self._last_pm_move_kind = ""
        self._last_ps_move_kind = ""
        self._pm_action_tags: Dict[int, Dict[str, bool]] = {}
        self._last_pm_selected_tags = {"unblocking": False, "wip": False}
        self._last_pm_selected_action: Optional[Move] = None
        self.collect_hlns_stats = collect_hlns_stats
        self._hlns_legal_machines = {op: tuple(operations[op[0]][op[1]]["machines"])
                                    for op in self.encoder.ms_index_order}
        self._hlns_pending = None
        self._hlns_evaluated_child = None
        self.diagnostics["hlns"] = {key: 0 for key in (
            "hlns_calls", "hlns_normal_children", "hlns_same_genotype_fallbacks",
            "hlns_global_fallbacks", "hlns_unchanged_children", "hlns_diagnosis_errors",
            "hlns_os_changed_children", "hlns_ms_changed_children", "hlns_both_changed_children",
            "hlns_parent_shortage_improvement_count", "hlns_population_replacement_count",
            "H_count", "H_size_total", "H_ratio_total",
            "normal_repair_both_changed_children")}
        self.diagnostics["hlns"]["fallback_reasons"] = Counter()
        self.diagnostics["hlns"]["diagnosis_error_reasons"] = Counter()
        self.diagnostics["hlns"]["normal_repair_ms_changes"] = []
        self.diagnostics["hlns"]["normal_repair_os_pair_changes"] = []

    @property
    def algorithm_variant(self):
        return "HLNS_FULL"

    @property
    def pm_specialized_kind(self):
        return "MGS"

    @property
    def ps_specialized_kind(self):
        return "HLNS"

    @property
    def pm_archive_guidance_enabled(self):
        return True

    @property
    def ps_archive_guidance_enabled(self):
        return False

    def random_individual(self) -> Individual:
        os_seq = self.encoder.generate_random_os()
        ms_list = self.encoder.generate_random_ms()
        return Individual(os_seq, self.encoder.build_ms_map(ms_list),
                          operation_order=tuple(self.encoder.ms_index_order))

    def _evaluate_wip_exact(self, ind: Individual) -> Individual:
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

    def _os_discordant_pairs(self, x: Individual, y: Individual) -> int:
        """Count changed cross-job order pairs, including cross-stage pairs."""
        keys = self._distance_keys
        x_key, y_key = tuple(x.os_seq), tuple(y.os_seq)
        pair_key = (x_key, y_key)
        cached = self._os_pair_cache.get(pair_key)
        if cached is None:
            cached = self._os_pair_cache.get((y_key, x_key))
        if cached is not None:
            return cached
        # Immutable keys prevent stale ranks even when a caller edits an OS in place.
        os_key = y_key
        rank_y = self._distance_rank_cache.get(os_key)
        if rank_y is None:
            rank_y = {key: position for position, key in enumerate(y.os_seq)}
            if len(self._distance_rank_cache) >= 128:
                self._distance_rank_cache.pop(next(iter(self._distance_rank_cache)))
            self._distance_rank_cache[os_key] = rank_y
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
            if not same_job or rank > same_job[-1]:
                same_job.append(rank)
            else:
                discordant -= len(same_job) - bisect_right(same_job, rank)
                insort(same_job, rank)

            cursor = rank + 1
            while cursor < len(tree):
                tree[cursor] += 1
                cursor += cursor & -cursor

        if len(self._os_pair_cache) >= 256:
            self._os_pair_cache.pop(next(iter(self._os_pair_cache)))
        self._os_pair_cache[pair_key] = discordant
        return discordant

    def structural_distance(self, x: Individual, y: Individual) -> float:
        keys = self._distance_keys
        discordant = self._os_discordant_pairs(x, y)
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

    def _sample_archive_guide(self) -> Optional[Individual]:
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
            nearest = ({id(z): min(self.structural_distance(z, prior) for prior in selected)
                        for z in group} if selected else {})
            for slot in range(slots):
                if selected:
                    choice = max(group, key=lambda z: (
                        nearest[id(z)],
                        self.genotype_key(z)))
                else:
                    choice = min(group, key=self.genotype_key)
                selected.append(choice)
                group = [z for z in group if z is not choice]
                if slot + 1 < slots:
                    for z in group:
                        distance = self.structural_distance(z, choice)
                        nearest[id(z)] = min(nearest.get(id(z), math.inf), distance)
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

    def _relative_move_is_legal(self, key: OpKey, other: OpKey,
                                positions: Dict[OpKey, int], before: bool) -> bool:
        """Test the existing insertion without allocating an OS or a child."""
        if key == other:
            return False
        old, other_pos = positions[key], positions[other]
        if (before and old < other_pos) or (not before and old > other_pos):
            return False
        target = other_pos
        if key[1] > 0:
            pred = positions[(key[0], key[1]-1)]
            target = max(target, pred - (pred > old) + 1)
        if key[1]+1 < len(self.operations[key[0]]):
            succ = positions[(key[0], key[1]+1)]
            target = min(target, succ - (succ > old))
        if target == old:
            return False
        shifted_other = other_pos - (other_pos > old)
        shifted_other += target <= shifted_other
        return target < shifted_other if before else target > shifted_other

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

    def _incoming_dependencies(self, provenance: DecodeProvenance) -> Dict[int, list]:
        """Reuse only the index; snapshot edges to detect external list edits."""
        key = id(provenance)
        snapshot = tuple(provenance.active_dependencies)
        cached = self._dependency_index_cache.get(key)
        if cached is not None and cached[0] is provenance and cached[1] == snapshot:
            return cached[2]
        incoming: Dict[int, list] = {}
        for edge in snapshot:
            incoming.setdefault(edge.dst_event_id, []).append(edge)
        if len(self._dependency_index_cache) >= 4:
            self._dependency_index_cache.pop(next(iter(self._dependency_index_cache)))
        self._dependency_index_cache[key] = (provenance, snapshot, incoming)
        return incoming

    def _backtrace(self, provenance: DecodeProvenance, anchors: Set[int],
                   incoming: Optional[Dict[int, list]] = None) -> Tuple[Set[int], list, Dict[int, int]]:
        if incoming is None:
            incoming = self._incoming_dependencies(provenance)
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
        positions = {key: pos for pos, key in enumerate(ind.os_seq)}
        action_reach_nodes: Dict[int, Set[int]] = {}
        seen_machine_edges = set()
        for edge in edges:
            src, dst = prov.events[edge.src_event_id], prov.events[edge.dst_event_id]
            if edge.kind == "machine":
                u, o = (src.job, src.op_idx), (dst.job, dst.op_idx)
                relation = (u, o, edge.machine)
                if relation not in seen_machine_edges:
                    seen_machine_edges.add(relation)
                    if (self._relative_move_is_legal(o, u, positions, before=True) or
                            self._relative_move_is_legal(u, o, positions, before=False)):
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

    def _mgs_child(self, seed: Individual) -> Optional[Individual]:
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
        incoming = self._incoming_dependencies(prov)
        for bid, events in prov.buffer_events.items():
            start = stats["per_buffer_active_start"][bid]
            end = stats["per_buffer_active_end"][bid]
            if end <= start:
                continue
            low = stats["per_buffer_low_wip"][bid]
            ordered = sorted(events, key=lambda ev: (ev.time, ev.event_id))
            event_times = [ev.time for ev in ordered]
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
                    for ev in ordered[bisect_left(event_times, left):bisect_right(event_times, right)]:
                        if ev.cause_event_id is None:
                            continue
                        before = max(0, low - ev.level_before)
                        after = max(0, low - ev.level_after)
                        if after > before and left <= ev.time < right:
                            anchors.add(ev.cause_event_id)
                        elif after < before and left < ev.time <= right:
                            anchors.add(ev.cause_event_id)
                    nodes, edges, _ = self._backtrace(prov, anchors, incoming)
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

    def vary_shortage(self, x: Individual) -> Individual:
        self._last_ps_move_kind = "basic_mutation"
        mutation = self.rng.random() < self.p_mut
        if mutation:
            return self.basic_mutation(x)
        self._last_ps_move_kind = "hlns_joint"
        stats = self.diagnostics["hlns"]
        stats["hlns_calls"] += 1
        relaxed = self.select_destroy_set(x)
        if not relaxed:
            return self._finish_hlns_child(x, self._global_fallback(x))
        child = self.joint_repair(x, relaxed, self.rng)
        if self.genotype_key(child) == self.genotype_key(x):
            stats["hlns_same_genotype_fallbacks"] += 1
            fallback = restricted_atomic_fallback(self, x, relaxed, self.rng)
            if fallback is not None:
                self._validate_candidate(x, fallback, relaxed)
                child = fallback
            else:
                stats["fallback_reasons"]["restricted_exhausted"] += 1
                child = self._global_fallback(x)
        else:
            stats["hlns_normal_children"] += 1
            if self.collect_hlns_stats:
                ms_changes = sum(x.ms_map[op] != child.ms_map[op] for op in self._distance_keys)
                os_changes = self._os_discordant_pairs(x, child)
                stats["normal_repair_ms_changes"].append(ms_changes)
                stats["normal_repair_os_pair_changes"].append(os_changes)
                stats["normal_repair_both_changed_children"] += int(ms_changes > 0 and os_changes > 0)
        return self._finish_hlns_child(x, child)

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

    def _similarity_replace_nearest(self, population: List[Individual], child: Individual,
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
                        child = self.vary_shortage(seed)
                        source = "P_S_self"
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

    @property
    def hlns_statistics(self):
        result = self.diagnostics["hlns"].copy()
        n = result["H_count"]
        result["mean_H_size"] = result["H_size_total"]/n if n else 0.0
        result["mean_H_ratio"] = result["H_ratio_total"]/n if n else 0.0
        calls = result["hlns_calls"]
        result["same_genotype_fallback_ratio"] = result["hlns_same_genotype_fallbacks"]/calls if calls else 0.0
        result["fallback_reasons"] = dict(result["fallback_reasons"])
        result["diagnosis_error_reasons"] = dict(result["diagnosis_error_reasons"])
        for key in ("normal_repair_ms_changes", "normal_repair_os_pair_changes"):
            result[key] = result[key][:]
        return result

    def _vary_makespan_specialized(self, seed):
        return self._mgs_child(seed)

    def archive_guide(self):
        if not self.pm_archive_guidance_enabled:
            return None
        return self._sample_archive_guide()

    def _diagnosis_error(self, reason):
        stats = self.diagnostics["hlns"]
        stats["hlns_diagnosis_errors"] += 1
        stats["diagnosis_error_reasons"][reason] += 1
        raise HLNSDiagnosisError(reason)

    def _diagnostic_high(self, parent):
        if parent.shortage == 0:
            return set(), "zero_shortage"
        if parent.shortage is None or parent.stats is None or parent.provenance is None:
            return set(), "unavailable_diagnosis"
        if not math.isfinite(parent.shortage) or parent.shortage < 0:
            self._diagnosis_error("invalid_parent_shortage")
        try:
            diagnosis = self.diagnose_shortage(parent)
        except Exception as exc:
            # Record and re-raise unknown programming/interface failures verbatim.
            stats = self.diagnostics["hlns"]
            stats["hlns_diagnosis_errors"] += 1
            stats["diagnosis_error_reasons"][type(exc).__name__] += 1
            raise
        legal = set(self.encoder.ms_index_order)
        if (diagnosis.unanchored_intervals or not set(diagnosis.phi) <= legal or
                not diagnosis.influence_events <= set(parent.provenance.events)):
            self._diagnosis_error("invalid_anchors_or_operations")
        if (any(not math.isfinite(v) or v < 0 for v in diagnosis.phi.values()) or
                not math.isclose(sum(diagnosis.phi.values()), parent.shortage, rel_tol=0, abs_tol=1e-9) or
                not math.isclose(sum(row[3] for row in diagnosis.intervals), parent.shortage, rel_tol=0, abs_tol=1e-9)):
            self._diagnosis_error("attribution_mass_mismatch")
        high = self.high_influence_set(parent, diagnosis)
        if not high:
            return set(), "empty_H"
        ordered = sorted(diagnosis.phi, key=lambda op: (-diagnosis.phi[op], op))
        n = len(high)
        # Match high_influence_set's accumulation, including threshold rounding.
        prefix_mass, previous_mass = 0.0, 0.0
        for op in ordered[:n]:
            previous_mass = prefix_mass
            prefix_mass += diagnosis.phi[op]
        if (set(ordered[:n]) != set(high) or
                prefix_mass/parent.shortage < self.rho or
                previous_mass/parent.shortage >= self.rho):
            self._diagnosis_error("H_not_minimal_rho_prefix")
        return set(high), None

    def select_destroy_set(self, parent: Individual) -> Set[OpKey]:
        """Destroy exactly the operations in the formal diagnosis H."""
        high, reason = self._diagnostic_high(parent)
        if reason:
            self.diagnostics["hlns"]["fallback_reasons"][reason] += 1
            return set()
        stats = self.diagnostics["hlns"]
        stats["H_count"] += 1
        stats["H_size_total"] += len(high)
        stats["H_ratio_total"] += len(high)/len(self.encoder.ms_index_order)
        return high

    def _validate_candidate(self, parent, child, relaxed):
        self.encoder.validate_os(parent.os_seq)
        self.encoder.validate_os(child.os_seq)
        self.encoder.build_ms_map([child.ms_map[op] for op in self.encoder.ms_index_order])
        if set(child.ms_map) != set(self.encoder.ms_index_order):
            raise HLNSRepairError("MS operation coverage mismatch")
        if [op for op in parent.os_seq if op not in relaxed] != [op for op in child.os_seq if op not in relaxed]:
            raise HLNSRepairError("Non-relaxed OS skeleton changed")
        if any(child.ms_map[op] != parent.ms_map[op] for op in self.encoder.ms_index_order if op not in relaxed):
            raise HLNSRepairError("Non-relaxed machine changed")

    def joint_repair(self, parent: Individual, relaxed_set: Iterable[OpKey], rng=None) -> Individual:
        """One full joint candidate, not an exact solution of a joint subproblem."""
        rng = self.rng if rng is None else rng
        high = set(relaxed_set)
        if not high <= set(self.encoder.ms_index_order):
            raise HLNSRepairError("Unknown relaxed operation")
        q = parent_deviation_probability(len(high))
        self.encoder.validate_os(parent.os_seq)
        if set(parent.ms_map) != set(self.encoder.ms_index_order):
            raise HLNSRepairError("Invalid parent MS coverage")
        self.encoder.build_ms_map([parent.ms_map[op] for op in self.encoder.ms_index_order])
        os_seq = biased_topological_os_repair(parent.os_seq, high, q, rng, parent_bias=True)
        ms_map = biased_machine_repair(parent.ms_map, high, self._hlns_legal_machines, q, rng, parent_bias=True)
        child = Individual(os_seq, ms_map, operation_order=tuple(self.encoder.ms_index_order))
        self._validate_candidate(parent, child, high)
        return child

    def _global_fallback(self, parent):
        self.diagnostics["hlns"]["hlns_global_fallbacks"] += 1
        return self.basic_mutation(parent)

    def _finish_hlns_child(self, parent, child):
        stats = self.diagnostics["hlns"]
        same = child.os_seq == parent.os_seq and child.ms_map == parent.ms_map
        stats["hlns_unchanged_children"] += int(same)
        if self.collect_hlns_stats:
            os_changed = child.os_seq != parent.os_seq
            ms_changed = child.ms_map != parent.ms_map
            stats["hlns_os_changed_children"] += int(os_changed)
            stats["hlns_ms_changed_children"] += int(ms_changed)
            stats["hlns_both_changed_children"] += int(os_changed and ms_changed)
            self._hlns_pending = (child, parent.shortage)
        return child

    def evaluate(self, ind):
        result = self._evaluate_wip_exact(ind)
        pending = self._hlns_pending
        if self.collect_hlns_stats and pending is not None and pending[0] is ind:
            if pending[1] is not None and ind.shortage < pending[1]:
                self.diagnostics["hlns"]["hlns_parent_shortage_improvement_count"] += 1
            self._hlns_pending = None
            self._hlns_evaluated_child = ind
        return result

    def similarity_replace(self, population, child, objective):
        accepted = self._similarity_replace_nearest(population, child, objective)
        if self.collect_hlns_stats and objective == "shortage" and child is self._hlns_evaluated_child:
            self.diagnostics["hlns"]["hlns_population_replacement_count"] += int(accepted)
            self._hlns_evaluated_child = None
        return accepted


# Compatibility name only; both names refer to the same H-LNS class.
HLNSDualPopulation = WIPGraphDualPopulation
