from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import random
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.solution.encoder import Encoder
from src.solution.decoder import StageBufferWIPScheduler


@dataclass
class Individual:
    """
    个体表示：
    - OS: job-based operation sequence
    - MS: machine selection list（与 encoder.ms_index_order 对齐）
    """
    OS: List[str]
    MS: List[str]

    fitness: Optional[float] = None
    makespan: Optional[int] = None
    schedule: Optional[List[Dict[str, Any]]] = None
    buffer_trace: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None

    def copy(self) -> "Individual":
        return Individual(
            OS=self.OS[:],
            MS=self.MS[:],
            fitness=self.fitness,
            makespan=self.makespan,
            schedule=[rec.copy() for rec in self.schedule] if self.schedule is not None else None,
            buffer_trace={k: v[:] for k, v in self.buffer_trace.items()} if self.buffer_trace is not None else None,
            stats=self.stats.copy() if self.stats is not None else None,
        )


class BaselineGA:
    """
    单种群基础进化搜索（Baseline）

    包含：
    1. 初始化
    2. 评价
    3. 锦标赛选择
    4. OS/MS 交叉
    5. OS/MS 变异
    6. 环境选择
    7. run 主循环
    """

    def __init__(
        self,
        operations: Dict[str, List[Dict[str, Any]]],
        buffers: Dict[str, Dict[str, Any]],
        pop_size: int = 20,
        n_generations: int = 100,
        crossover_rate: float = 0.8,
        os_mutation_rate: float = 0.2,
        ms_mutation_rate: float = 0.2,
        tournament_size: int = 2,
        seed: Optional[int] = None,
    ):
        if pop_size <= 0:
            raise ValueError("pop_size 必须 > 0")
        if n_generations <= 0:
            raise ValueError("n_generations 必须 > 0")
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
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.os_mutation_rate = os_mutation_rate
        self.ms_mutation_rate = ms_mutation_rate
        self.tournament_size = tournament_size
        self.seed = seed

        self.rng = random.Random(seed)

        # 注意：Encoder 已支持 rng 传入
        self.encoder = Encoder(self.operations, rng=self.rng)
        self.scheduler = StageBufferWIPScheduler(self.operations, self.buffers)

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history_best_fitness: List[float] = []

    # =========================
    # 初始化
    # =========================

    def initialize_individual(self) -> Individual:
        os_seq = self.encoder.generate_random_os()
        ms_list = self.encoder.generate_random_ms()
        return Individual(OS=os_seq, MS=ms_list)

    def initialize_population(self) -> List[Individual]:
        population = [self.initialize_individual() for _ in range(self.pop_size)]
        self.population = population
        return population

    # =========================
    # 评价
    # =========================

    def evaluate_individual(self, ind: Individual, store_stats: bool = True) -> float:
        ms_map = self.encoder.build_ms_map(ind.MS)

        makespan, schedule, buffer_trace = self.scheduler.decode(
            os_seq=ind.OS,
            ms_map=ms_map
        )

        ind.makespan = makespan
        ind.fitness = float(makespan)
        ind.schedule = schedule
        ind.buffer_trace = buffer_trace

        if store_stats:
            ind.stats = self.scheduler.analyze(
                schedule=schedule,
                buffer_trace=buffer_trace,
                makespan=makespan
            )
        else:
            ind.stats = None

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
            self.evaluate_individual(ind, store_stats=store_stats)

        self._update_best(population)

    # =========================
    # 排序 / 最优解维护
    # =========================

    def sort_population(self, population: Optional[List[Individual]] = None) -> List[Individual]:
        if population is None:
            population = self.population

        if any(ind.fitness is None for ind in population):
            raise ValueError("存在未评价个体，不能排序")

        return sorted(population, key=lambda ind: ind.fitness)

    def get_best_individual(self, population: Optional[List[Individual]] = None) -> Individual:
        sorted_pop = self.sort_population(population)
        return sorted_pop[0]

    def _update_best(self, population: Optional[List[Individual]] = None) -> None:
        current_best = self.get_best_individual(population)

        if self.best_individual is None:
            self.best_individual = current_best.copy()
            return

        if current_best.fitness < self.best_individual.fitness:
            self.best_individual = current_best.copy()

    # =========================
    # 选择
    # =========================

    def tournament_select(self, population: Optional[List[Individual]] = None) -> Individual:
        if population is None:
            population = self.population

        if len(population) < self.tournament_size:
            raise ValueError("population 数量小于 tournament_size")

        candidates = self.rng.sample(population, self.tournament_size)
        winner = min(candidates, key=lambda ind: ind.fitness)
        return winner.copy()

    # =========================
    # 交叉
    # =========================

    def crossover_os(self, os1: List[str], os2: List[str]) -> Tuple[List[str], List[str]]:
        """
        job-based POX 交叉
        """
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
        """
        MS 均匀交叉
        """
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
            return p1.copy(), p2.copy()

        child1_os, child2_os = self.crossover_os(p1.OS, p2.OS)
        child1_ms, child2_ms = self.crossover_ms(p1.MS, p2.MS)

        child1 = Individual(OS=child1_os, MS=child1_ms)
        child2 = Individual(OS=child2_os, MS=child2_ms)

        return child1, child2

    # =========================
    # 变异
    # =========================

    def mutate_os(self, os_seq: List[str]) -> List[str]:
        """
        OS swap mutation
        """
        new_os = os_seq[:]

        if len(new_os) < 2:
            return new_os

        if self.rng.random() < self.os_mutation_rate:
            i, j = self.rng.sample(range(len(new_os)), 2)
            new_os[i], new_os[j] = new_os[j], new_os[i]

        return new_os

    def mutate_ms(self, ms_list: List[str]) -> List[str]:
        """
        MS 随机重选合法机器
        """
        new_ms = ms_list[:]

        for idx, (job, op) in enumerate(self.encoder.ms_index_order):
            if self.rng.random() < self.ms_mutation_rate:
                legal_machines = list(self.operations[job][op]["machines"].keys())
                new_ms[idx] = self.rng.choice(legal_machines)

        return new_ms

    def reset_individual_evaluation(self, ind: Individual) -> None:
        ind.fitness = None
        ind.makespan = None
        ind.schedule = None
        ind.buffer_trace = None
        ind.stats = None

    def mutate(self, ind: Individual) -> Individual:
        new_ind = ind.copy()
        new_ind.OS = self.mutate_os(new_ind.OS)
        new_ind.MS = self.mutate_ms(new_ind.MS)

        self.reset_individual_evaluation(new_ind)
        return new_ind

    # =========================
    # 子代生成 / 单代更新
    # =========================

    def generate_offspring(self) -> List[Individual]:
        offspring = []

        while len(offspring) < self.pop_size:
            parent1 = self.tournament_select(self.population)
            parent2 = self.tournament_select(self.population)

            child1, child2 = self.crossover(parent1, parent2)

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            offspring.append(child1)
            if len(offspring) < self.pop_size:
                offspring.append(child2)

        return offspring

    def run_one_generation(self, store_stats: bool = False) -> None:
        """
        baseline 单代更新：
        1. 生成子代
        2. 评价子代
        3. 父代 + 子代 合并
        4. 截断保留前 pop_size
        5. 更新 best
        """
        offspring = self.generate_offspring()
        self.evaluate_population(offspring, store_stats=store_stats)

        merged = self.population + offspring
        merged_sorted = self.sort_population(merged)

        self.population = [ind.copy() for ind in merged_sorted[:self.pop_size]]
        self._update_best(self.population)

    # =========================
    # 主循环
    # =========================

    def run(
        self,
        store_stats_init: bool = True,
        store_stats_generations: bool = False,
        verbose: bool = True
    ) -> Individual:
        self.initialize_population()
        self.evaluate_population(self.population, store_stats=store_stats_init)

        init_best = self.get_best_individual(self.population)
        self.history_best_fitness = [init_best.fitness]

        if verbose:
            print(f"[Init] best fitness = {init_best.fitness}, makespan = {init_best.makespan}")

        for gen in range(1, self.n_generations + 1):
            self.run_one_generation(store_stats=store_stats_generations)

            best = self.get_best_individual(self.population)
            self.history_best_fitness.append(best.fitness)

            if verbose:
                print(f"[Gen {gen}] best fitness = {best.fitness}, makespan = {best.makespan}")

        return self.best_individual.copy()

    # =========================
    # 调试 / 展示
    # =========================

    def print_population_summary(self, population: Optional[List[Individual]] = None, top_k: int = 5) -> None:
        if population is None:
            population = self.population

        if not population:
            print("population is empty")
            return

        sorted_pop = self.sort_population(population)
        k = min(top_k, len(sorted_pop))

        print(f"Population size: {len(sorted_pop)}")
        print(f"Top {k} individuals:")
        for i in range(k):
            ind = sorted_pop[i]
            print(
                f"[{i}] fitness={ind.fitness}, makespan={ind.makespan}, "
                f"OS_len={len(ind.OS)}, MS_len={len(ind.MS)}"
            )

    def print_best_summary(self) -> None:
        if self.best_individual is None:
            print("best individual is None")
            return

        ind = self.best_individual
        print("===== Best Individual =====")
        print(f"fitness  : {ind.fitness}")
        print(f"makespan : {ind.makespan}")
        print(f"OS       : {ind.OS}")
        print(f"MS       : {ind.MS}")

        if ind.stats is not None:
            total_blocking = ind.stats["blocking"]["total_blocking_time"]
            print(f"blocking : {total_blocking}")