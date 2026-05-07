from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import random
import math
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.solution.encoder import Encoder
from src.solution.decoder import StageBufferWIPScheduler


@dataclass
class Individual:
    OS: List[str]
    MS: List[str]

    fitness: Optional[float] = None
    makespan: Optional[int] = None
    shortage: Optional[float] = None

    rank: Optional[int] = None
    crowding_distance: float = 0.0

    schedule: Optional[List[Dict[str, Any]]] = None
    buffer_trace: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None

    origin_task: Optional[str] = None

    # SPEA2 内部字段
    strength: float = 0.0
    raw_fitness: float = 0.0
    density: float = 0.0

    def copy(self) -> "Individual":
        return Individual(
            OS=self.OS[:],
            MS=self.MS[:],
            fitness=self.fitness,
            makespan=self.makespan,
            shortage=self.shortage,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            schedule=[rec.copy() for rec in self.schedule] if self.schedule is not None else None,
            buffer_trace={k: v[:] for k, v in self.buffer_trace.items()} if self.buffer_trace is not None else None,
            stats=self.stats.copy() if self.stats is not None else None,
            origin_task=self.origin_task,
            strength=self.strength,
            raw_fitness=self.raw_fitness,
            density=self.density,
        )


class BaselineSPEA2:
    """
    SPEA2 baseline for WIP-aware multi-objective scheduling.

    Objectives:
    - min makespan
    - min shortage

    Implementation follows SPEA2:
    - strength S(i)
    - raw fitness R(i)
    - k-th nearest-neighbor density D(i)
    - F(i) = R(i) + D(i)
    - fixed-size external archive
    - archive truncation by nearest-neighbor distance
    """

    def __init__(
        self,
        operations: Dict[str, List[Dict[str, Any]]],
        buffers: Dict[str, Dict[str, Any]],
        pop_size: int = 100,
        archive_size: Optional[int] = None,
        n_generations: int = 100,
        max_evaluations: Optional[int] = None,
        snapshot_interval: Optional[int] = None,
        crossover_rate: float = 0.8,
        os_mutation_rate: float = 0.2,
        ms_mutation_rate: float = 0.2,
        tournament_size: int = 2,
        seed: Optional[int] = None,
    ):
        if pop_size <= 1:
            raise ValueError("pop_size 必须 > 1")
        if archive_size is not None and archive_size <= 1:
            raise ValueError("archive_size 必须 > 1")
        if n_generations <= 0:
            raise ValueError("n_generations 必须 > 0")
        if max_evaluations is not None and max_evaluations <= 0:
            raise ValueError("max_evaluations 必须 > 0")
        if snapshot_interval is not None and snapshot_interval <= 0:
            raise ValueError("snapshot_interval 必须 > 0")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate 必须在 [0,1] 内")
        if not (0.0 <= os_mutation_rate <= 1.0):
            raise ValueError("os_mutation_rate 必须在 [0,1] 内")
        if not (0.0 <= ms_mutation_rate <= 1.0):
            raise ValueError("ms_mutation_rate 必须在 [0,1] 内")
        if tournament_size <= 0:
            raise ValueError("tournament_size 必须 > 0")

        self.operations = operations
        self.buffers = buffers

        self.pop_size = pop_size
        self.archive_size = archive_size if archive_size is not None else pop_size

        self.n_generations = n_generations
        self.max_evaluations = max_evaluations
        self.snapshot_interval = snapshot_interval
        self.n_evaluations = 0

        self.crossover_rate = crossover_rate
        self.os_mutation_rate = os_mutation_rate
        self.ms_mutation_rate = ms_mutation_rate
        self.tournament_size = tournament_size
        self.seed = seed
        self.rng = random.Random(seed)

        self.encoder = Encoder(self.operations, rng=self.rng)
        self.scheduler = StageBufferWIPScheduler(self.operations, self.buffers)

        self.population: List[Individual] = []
        self.archive: List[Individual] = []

        self.best_individual: Optional[Individual] = None

        self.history_best_fitness: List[float] = []
        self.history_best_shortage: List[float] = []
        self.history_eval_counts: List[int] = []
        self.history_fronts: List[Dict[str, Any]] = []
        self._last_snapshot_eval: int = -1

    # =========================================================
    # 基础工具
    # =========================================================

    def has_budget(self) -> bool:
        return self.max_evaluations is None or self.n_evaluations < self.max_evaluations

    def remaining_budget(self) -> Optional[int]:
        if self.max_evaluations is None:
            return None
        return max(0, self.max_evaluations - self.n_evaluations)

    def initialize_individual(self, origin_task: Optional[str] = None) -> Individual:
        return Individual(
            OS=self.encoder.generate_random_os(),
            MS=self.encoder.generate_random_ms(),
            origin_task=origin_task,
        )

    def initialize_population(self) -> None:
        self.population = [
            self.initialize_individual(origin_task="spea2_population")
            for _ in range(self.pop_size)
        ]
        self.archive = []

    # =========================================================
    # 评价
    # =========================================================

    def evaluate_individual(self, ind: Individual, store_stats: bool = True) -> float:
        if not self.has_budget():
            raise RuntimeError("Evaluation budget exhausted")

        ms_map = self.encoder.build_ms_map(ind.MS)

        makespan, schedule, buffer_trace = self.scheduler.decode(
            os_seq=ind.OS,
            ms_map=ms_map
        )

        ind.makespan = makespan
        ind.schedule = schedule
        ind.buffer_trace = buffer_trace

        ind.stats = self.scheduler.analyze(
            schedule=schedule,
            buffer_trace=buffer_trace,
            makespan=makespan
        )

        ind.shortage = float(ind.stats["shortage"]["total_shortage_area"])

        # 会在 SPEA2 fitness assignment 中被覆盖；
        # 这里先给兼容字段一个可用值。
        ind.fitness = float(makespan)

        self.n_evaluations += 1
        return ind.fitness

    def evaluate_population(
        self,
        population: Optional[List[Individual]] = None,
        store_stats: bool = True
    ) -> None:
        if population is None:
            population = self.population

        if not population:
            raise ValueError("population 为空，无法评价")

        for ind in population:
            if not self.has_budget():
                break
            if ind.makespan is None or ind.shortage is None:
                self.evaluate_individual(ind, store_stats=store_stats)

        evaluated = [
            ind for ind in population
            if ind.makespan is not None and ind.shortage is not None
        ]
        if evaluated:
            self._update_best(evaluated)

    # =========================================================
    # 多目标基础函数
    # =========================================================

    def get_objectives(self, ind: Individual) -> Tuple[float, float]:
        if ind.makespan is None or ind.shortage is None:
            raise ValueError("个体尚未完成双目标评价")
        return float(ind.makespan), float(ind.shortage)

    def dominates(self, a: Individual, b: Individual) -> bool:
        a1, a2 = self.get_objectives(a)
        b1, b2 = self.get_objectives(b)
        return (
            a1 <= b1 and
            a2 <= b2 and
            (a1 < b1 or a2 < b2)
        )

    def get_pareto_front(self, population: Optional[List[Individual]] = None) -> List[Individual]:
        if population is None:
            population = self.archive if self.archive else self.population

        valid = [
            ind for ind in population
            if ind.makespan is not None and ind.shortage is not None
        ]

        pareto = []
        for p in valid:
            dominated = False
            for q in valid:
                if p is q:
                    continue
                if self.dominates(q, p):
                    dominated = True
                    break
            if not dominated:
                c = p.copy()
                c.rank = 0
                pareto.append(c)

        pareto.sort(key=lambda x: (x.makespan, x.shortage))
        return pareto

    def get_best_individual(self, population: Optional[List[Individual]] = None) -> Individual:
        front = self.get_pareto_front(population)
        if not front:
            raise ValueError("population 中没有可用个体")
        front.sort(key=lambda ind: (ind.makespan, ind.shortage))
        return front[0]

    def _update_best(self, population: Optional[List[Individual]] = None) -> None:
        if population is None:
            population = self.archive if self.archive else self.population

        current_best = self.get_best_individual(population)

        if self.best_individual is None:
            self.best_individual = current_best.copy()
            return

        cur = (current_best.makespan, current_best.shortage)
        old = (self.best_individual.makespan, self.best_individual.shortage)
        if cur < old:
            self.best_individual = current_best.copy()

    # =========================================================
    # SPEA2 fitness assignment
    # =========================================================

    def objective_distance(self, a: Individual, b: Individual) -> float:
        a1, a2 = self.get_objectives(a)
        b1, b2 = self.get_objectives(b)
        return math.sqrt((a1 - b1) ** 2 + (a2 - b2) ** 2)

    def assign_spea2_fitness(self, union: List[Individual]) -> None:
        """
        SPEA2:
        S(i) = number of individuals dominated by i
        R(i) = sum of S(j) for all j dominating i
        D(i) = 1 / (sigma_k(i) + 2)
        F(i) = R(i) + D(i)
        """
        valid = [
            ind for ind in union
            if ind.makespan is not None and ind.shortage is not None
        ]
        n = len(valid)
        if n == 0:
            return

        dominates_matrix = [[False] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(valid[i], valid[j]):
                    dominates_matrix[i][j] = True

        strengths = [0.0] * n
        for i in range(n):
            strengths[i] = float(sum(1 for j in range(n) if dominates_matrix[i][j]))

        raw = [0.0] * n
        for i in range(n):
            raw[i] = sum(strengths[j] for j in range(n) if dominates_matrix[j][i])

        k = int(math.sqrt(n))
        k = max(1, min(k, n - 1)) if n > 1 else 1

        densities = [0.0] * n
        for i in range(n):
            if n <= 1:
                sigma_k = 0.0
            else:
                distances = []
                for j in range(n):
                    if i == j:
                        continue
                    distances.append(self.objective_distance(valid[i], valid[j]))
                distances.sort()
                sigma_k = distances[k - 1] if distances else 0.0

            densities[i] = 1.0 / (sigma_k + 2.0)

        for i, ind in enumerate(valid):
            ind.strength = strengths[i]
            ind.raw_fitness = raw[i]
            ind.density = densities[i]
            ind.fitness = raw[i] + densities[i]

            # 兼容外部代码：非支配个体 rank=0，其余 rank=1
            ind.rank = 0 if raw[i] == 0 else 1
            ind.crowding_distance = 1.0 / densities[i] if densities[i] > 0 else float("inf")

    # =========================================================
    # SPEA2 environmental selection
    # =========================================================

    def environmental_selection(self, union: List[Individual]) -> List[Individual]:
        valid = [
            ind for ind in union
            if ind.makespan is not None and ind.shortage is not None and ind.fitness is not None
        ]

        if not valid:
            return []

        # SPEA2: copy all individuals with F < 1 into next archive.
        next_archive = [ind.copy() for ind in valid if ind.fitness < 1.0]

        if len(next_archive) == self.archive_size:
            return next_archive

        if len(next_archive) < self.archive_size:
            dominated_sorted = sorted(
                [ind for ind in valid if ind.fitness >= 1.0],
                key=lambda ind: (ind.fitness, ind.makespan, ind.shortage)
            )

            needed = self.archive_size - len(next_archive)
            next_archive.extend(ind.copy() for ind in dominated_sorted[:needed])

            # 极端情况下 union 不足 archive_size，允许 archive 暂时不足。
            return next_archive

        # len(next_archive) > archive_size
        return self.truncate_archive(next_archive, self.archive_size)

    def truncate_archive(
        self,
        archive: List[Individual],
        target_size: int
    ) -> List[Individual]:
        """
        SPEA2 truncation:
        iteratively remove the individual whose nearest-neighbor distance vector
        is lexicographically smallest.
        """
        truncated = [ind.copy() for ind in archive]

        while len(truncated) > target_size:
            n = len(truncated)

            distance_vectors = []
            for i in range(n):
                distances = []
                for j in range(n):
                    if i == j:
                        continue
                    distances.append(self.objective_distance(truncated[i], truncated[j]))
                distances.sort()
                distance_vectors.append(distances)

            remove_idx = min(
                range(n),
                key=lambda idx: (
                    distance_vectors[idx],
                    truncated[idx].makespan,
                    truncated[idx].shortage,
                )
            )

            truncated.pop(remove_idx)

        return truncated

    # =========================================================
    # 选择 / 交叉 / 变异
    # =========================================================

    def tournament_select(self, population: List[Individual]) -> Individual:
        if len(population) < self.tournament_size:
            raise ValueError("population 数量小于 tournament_size")

        candidates = self.rng.sample(population, self.tournament_size)
        winner = min(
            candidates,
            key=lambda ind: (
                ind.fitness if ind.fitness is not None else float("inf"),
                ind.makespan if ind.makespan is not None else float("inf"),
                ind.shortage if ind.shortage is not None else float("inf"),
            )
        )
        return winner.copy()

    def crossover_os(self, os1: List[str], os2: List[str]) -> Tuple[List[str], List[str]]:
        jobs = list(self.operations.keys())
        n_jobs = len(jobs)

        if n_jobs <= 1:
            return os1[:], os2[:]

        k = self.rng.randint(1, n_jobs - 1)
        keep_jobs = set(self.rng.sample(jobs, k))

        child1 = [None] * len(os1)
        child2 = [None] * len(os2)

        for i, gene in enumerate(os1):
            if gene in keep_jobs:
                child1[i] = gene

        for i, gene in enumerate(os2):
            if gene in keep_jobs:
                child2[i] = gene

        fill1 = [g for g in os2 if g not in keep_jobs]
        fill2 = [g for g in os1 if g not in keep_jobs]

        ptr1 = 0
        ptr2 = 0

        for i in range(len(child1)):
            if child1[i] is None:
                child1[i] = fill1[ptr1]
                ptr1 += 1

        for i in range(len(child2)):
            if child2[i] is None:
                child2[i] = fill2[ptr2]
                ptr2 += 1

        return child1, child2

    def crossover_ms(self, ms1: List[str], ms2: List[str]) -> Tuple[List[str], List[str]]:
        child1 = []
        child2 = []

        for g1, g2 in zip(ms1, ms2):
            if self.rng.random() < 0.5:
                child1.append(g1)
                child2.append(g2)
            else:
                child1.append(g2)
                child2.append(g1)

        return child1, child2

    def crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if self.rng.random() > self.crossover_rate:
            c1 = p1.copy()
            c2 = p2.copy()
            self.reset_individual_evaluation(c1)
            self.reset_individual_evaluation(c2)
            return c1, c2

        child1_os, child2_os = self.crossover_os(p1.OS, p2.OS)
        child1_ms, child2_ms = self.crossover_ms(p1.MS, p2.MS)

        return Individual(OS=child1_os, MS=child1_ms), Individual(OS=child2_os, MS=child2_ms)

    def mutate_os(self, os_seq: List[str]) -> List[str]:
        new_os = os_seq[:]

        if len(new_os) < 2:
            return new_os

        if self.rng.random() >= self.os_mutation_rate:
            return new_os

        op_type = self.rng.choice(["swap", "insert"])

        if op_type == "swap":
            i, j = self.rng.sample(range(len(new_os)), 2)
            new_os[i], new_os[j] = new_os[j], new_os[i]
        else:
            i, j = self.rng.sample(range(len(new_os)), 2)
            gene = new_os.pop(i)
            new_os.insert(j, gene)

        return new_os

    def mutate_ms(self, ms_list: List[str]) -> List[str]:
        new_ms = ms_list[:]

        for idx, (job, op) in enumerate(self.encoder.ms_index_order):
            if self.rng.random() < self.ms_mutation_rate:
                machine_dict = self.operations[job][op]["machines"]
                legal_machines = list(machine_dict.keys())

                if len(legal_machines) == 1:
                    new_ms[idx] = legal_machines[0]
                    continue

                if self.rng.random() < 0.7:
                    best_machine = min(legal_machines, key=lambda m: machine_dict[m])
                    new_ms[idx] = best_machine
                else:
                    new_ms[idx] = self.rng.choice(legal_machines)

        return new_ms

    def reset_individual_evaluation(self, ind: Individual) -> None:
        ind.fitness = None
        ind.makespan = None
        ind.shortage = None
        ind.rank = None
        ind.crowding_distance = 0.0
        ind.schedule = None
        ind.buffer_trace = None
        ind.stats = None
        ind.strength = 0.0
        ind.raw_fitness = 0.0
        ind.density = 0.0

    def mutate(self, ind: Individual) -> Individual:
        new_ind = ind.copy()
        new_ind.OS = self.mutate_os(new_ind.OS)
        new_ind.MS = self.mutate_ms(new_ind.MS)
        self.reset_individual_evaluation(new_ind)
        return new_ind

    def generate_offspring(self) -> List[Individual]:
        if not self.archive:
            return []

        remaining = self.remaining_budget()
        if remaining is not None and remaining <= 0:
            return []

        target_size = self.pop_size if remaining is None else min(self.pop_size, remaining)
        offspring: List[Individual] = []

        while len(offspring) < target_size:
            parent1 = self.tournament_select(self.archive)
            parent2 = self.tournament_select(self.archive)

            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            child1.origin_task = "spea2_offspring"
            child2.origin_task = "spea2_offspring"

            offspring.append(child1)
            if len(offspring) < target_size:
                offspring.append(child2)

        return offspring

    # =========================================================
    # 快照
    # =========================================================

    def export_pareto_front_snapshot(
        self,
        population: Optional[List[Individual]] = None
    ) -> List[Dict[str, Any]]:
        front = self.get_pareto_front(population)
        snapshot = []

        for ind in front:
            crowd = ind.crowding_distance
            if crowd == float("inf"):
                crowd = "inf"

            snapshot.append({
                "makespan": ind.makespan,
                "shortage": ind.shortage,
                "rank": ind.rank,
                "crowding_distance": crowd,
                "fitness": ind.fitness,
                "OS": ind.OS[:],
                "MS": ind.MS[:],
            })

        return snapshot

    def record_front_snapshot(self, force: bool = False) -> None:
        if self.snapshot_interval is None:
            return

        if not self.archive and not self.population:
            return

        current_eval = int(self.n_evaluations)

        if not force:
            if current_eval <= 0:
                return
            if current_eval - self._last_snapshot_eval < self.snapshot_interval:
                return

        if self.history_fronts and self.history_fronts[-1]["eval_count"] == current_eval:
            return

        snapshot = {
            "eval_count": current_eval,
            "pareto_front": self.export_pareto_front_snapshot(self.archive if self.archive else self.population)
        }

        self.history_fronts.append(snapshot)
        self._last_snapshot_eval = current_eval

    # =========================================================
    # 单代更新 / 主循环
    # =========================================================

    def run_one_generation(self, store_stats: bool = False) -> None:
        if not self.has_budget():
            return

        offspring = self.generate_offspring()

        if offspring:
            self.evaluate_population(offspring, store_stats=store_stats)
            offspring = [
                ind for ind in offspring
                if ind.makespan is not None and ind.shortage is not None
            ]

        self.population = offspring

        union = [ind.copy() for ind in self.population] + [ind.copy() for ind in self.archive]
        union = [
            ind for ind in union
            if ind.makespan is not None and ind.shortage is not None
        ]

        if not union:
            return

        self.assign_spea2_fitness(union)
        self.archive = self.environmental_selection(union)

        if self.archive:
            self.assign_spea2_fitness(self.archive)
            self._update_best(self.archive)

    def run(
        self,
        store_stats_init: bool = True,
        store_stats_generations: bool = False,
        verbose: bool = True
    ) -> Individual:

        self.n_evaluations = 0
        self.population = []
        self.archive = []
        self.best_individual = None

        self.history_best_fitness = []
        self.history_best_shortage = []
        self.history_eval_counts = []
        self.history_fronts = []
        self._last_snapshot_eval = -1

        self.initialize_population()

        self.evaluate_population(self.population, store_stats=store_stats_init)
        self.population = [
            ind for ind in self.population
            if ind.makespan is not None and ind.shortage is not None
        ]

        if not self.population:
            raise RuntimeError("初始化阶段预算不足，未能完成任何个体评价")

        union = [ind.copy() for ind in self.population]
        self.assign_spea2_fitness(union)
        self.archive = self.environmental_selection(union)

        if self.archive:
            self.assign_spea2_fitness(self.archive)
            self._update_best(self.archive)
        else:
            self._update_best(self.population)

        init_best = self.get_best_individual(self.archive if self.archive else self.population)
        self.history_best_fitness = [float(init_best.makespan)]
        self.history_best_shortage = [float(init_best.shortage)]
        self.history_eval_counts = [int(self.n_evaluations)]

        self.record_front_snapshot(force=True)

        if verbose:
            print(
                f"[Init] rep_makespan = {init_best.makespan}, "
                f"rep_shortage = {init_best.shortage}, "
                f"archive_size = {len(self.archive)}, "
                f"pareto_size = {len(self.get_pareto_front())}, "
                f"evals = {self.n_evaluations}"
            )

        gen = 0
        while True:
            if self.max_evaluations is not None:
                if not self.has_budget():
                    break
            else:
                if gen >= self.n_generations:
                    break

            gen += 1
            prev_evals = self.n_evaluations

            self.run_one_generation(store_stats=store_stats_generations)

            if self.n_evaluations == prev_evals:
                break

            best = self.get_best_individual(self.archive if self.archive else self.population)
            self.history_best_fitness.append(float(best.makespan))
            self.history_best_shortage.append(float(best.shortage))
            self.history_eval_counts.append(int(self.n_evaluations))

            self.record_front_snapshot(force=False)

            if verbose:
                print(
                    f"[Gen {gen}] rep_makespan = {best.makespan}, "
                    f"rep_shortage = {best.shortage}, "
                    f"archive_size = {len(self.archive)}, "
                    f"pareto_size = {len(self.get_pareto_front())}, "
                    f"evals = {self.n_evaluations}"
                )

        self.record_front_snapshot(force=True)

        if self.best_individual is None:
            self._update_best(self.archive if self.archive else self.population)

        return self.best_individual.copy()

    # =========================================================
    # 展示
    # =========================================================

    def print_population_summary(self, population: Optional[List[Individual]] = None, top_k: int = 5) -> None:
        if population is None:
            population = self.archive if self.archive else self.population

        if not population:
            print("population is empty")
            return

        valid = [
            ind for ind in population
            if ind.makespan is not None and ind.shortage is not None
        ]
        valid.sort(
            key=lambda ind: (
                ind.fitness if ind.fitness is not None else float("inf"),
                ind.makespan,
                ind.shortage
            )
        )

        print(f"Population size: {len(valid)}")
        print(f"Pareto size: {len(self.get_pareto_front(population))}")

        k = min(top_k, len(valid))
        print(f"Top {k} individuals:")
        for i in range(k):
            ind = valid[i]
            print(
                f"[{i}] fitness={ind.fitness}, raw={ind.raw_fitness}, density={ind.density}, "
                f"makespan={ind.makespan}, shortage={ind.shortage}, "
                f"OS_len={len(ind.OS)}, MS_len={len(ind.MS)}"
            )

    def print_best_summary(self) -> None:
        if self.best_individual is None:
            print("best individual is None")
            return

        ind = self.best_individual
        print("===== Best SPEA2 Representative Individual =====")
        print(f"fitness  : {ind.fitness}")
        print(f"raw      : {ind.raw_fitness}")
        print(f"density  : {ind.density}")
        print(f"makespan : {ind.makespan}")
        print(f"shortage : {ind.shortage}")
        print(f"origin   : {ind.origin_task}")
        print(f"OS       : {ind.OS}")
        print(f"MS       : {ind.MS}")

        if ind.stats is not None:
            print(f"blocking : {ind.stats['blocking']['total_blocking_time']}")