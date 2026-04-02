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

    fitness: Optional[float] = None   # 仅保留兼容性，不作为主排序依据
    makespan: Optional[int] = None
    shortage: Optional[float] = None

    rank: Optional[int] = None
    crowding_distance: float = 0.0

    schedule: Optional[List[Dict[str, Any]]] = None
    buffer_trace: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None

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
        )


class BaselineNSGA2:
    """
    基础版 NSGA-II：
    - 双目标：min makespan, min shortage
    - 无局部搜索
    - 保留与 EliteLSGA 尽量一致的编码/交叉/变异风格
    """

    def __init__(
        self,
        operations: Dict[str, List[Dict[str, Any]]],
        buffers: Dict[str, Dict[str, Any]],
        pop_size: int = 100,
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

        self.encoder = Encoder(self.operations, rng=self.rng)
        self.scheduler = StageBufferWIPScheduler(self.operations, self.buffers)

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None

        # 这里记录“代表解”的历史，便于兼容现有实验脚本
        self.history_best_fitness: List[float] = []
        self.history_best_shortage: List[float] = []

    # =========================
    # 初始化相关
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
    # 评价相关
    # =========================

    def evaluate_individual(self, ind: Individual, store_stats: bool = True) -> float:
        ms_map = self.encoder.build_ms_map(ind.MS)

        makespan, schedule, buffer_trace = self.scheduler.decode(
            os_seq=ind.OS,
            ms_map=ms_map
        )

        ind.makespan = makespan
        ind.schedule = schedule
        ind.buffer_trace = buffer_trace

        # 双目标下 shortage 依赖 stats，因此这里始终计算
        ind.stats = self.scheduler.analyze(
            schedule=schedule,
            buffer_trace=buffer_trace,
            makespan=makespan
        )

        ind.shortage = float(ind.stats["shortage"]["total_shortage_area"])

        # fitness 仅保留兼容性
        ind.fitness = float(makespan)
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
    # 多目标相关工具函数
    # =========================

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

    def better(self, a: Individual, b: Individual) -> bool:
        if self.dominates(a, b):
            return True
        if self.dominates(b, a):
            return False

        if a.rank is not None and b.rank is not None:
            if a.rank != b.rank:
                return a.rank < b.rank
            return a.crowding_distance > b.crowding_distance

        return (a.makespan, a.shortage) < (b.makespan, b.shortage)

    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        S = {}
        n = {}
        fronts: List[List[Individual]] = [[]]

        for p in population:
            pid = id(p)
            S[pid] = []
            n[pid] = 0

            for q in population:
                if p is q:
                    continue
                if self.dominates(p, q):
                    S[pid].append(q)
                elif self.dominates(q, p):
                    n[pid] += 1

            if n[pid] == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                pid = id(p)
                for q in S[pid]:
                    qid = id(q)
                    n[qid] -= 1
                    if n[qid] == 0:
                        q.rank = i + 1
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            i += 1

        return fronts

    def assign_crowding_distance(self, front: List[Individual]) -> None:
        if not front:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return

        # 目标1：makespan
        front.sort(key=lambda x: x.makespan)
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        min_val = front[0].makespan
        max_val = front[-1].makespan
        if max_val > min_val:
            for i in range(1, len(front) - 1):
                if front[i].crowding_distance != float("inf"):
                    front[i].crowding_distance += (
                        (front[i + 1].makespan - front[i - 1].makespan) / (max_val - min_val)
                    )

        # 目标2：shortage
        front.sort(key=lambda x: x.shortage)
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        min_val = front[0].shortage
        max_val = front[-1].shortage
        if max_val > min_val:
            for i in range(1, len(front) - 1):
                if front[i].crowding_distance != float("inf"):
                    front[i].crowding_distance += (
                        (front[i + 1].shortage - front[i - 1].shortage) / (max_val - min_val)
                    )

    def assign_rank_and_crowding(self, population: List[Individual]) -> List[List[Individual]]:
        fronts = self.fast_non_dominated_sort(population)
        for front in fronts:
            self.assign_crowding_distance(front)
        return fronts

    def get_pareto_front(self, population: Optional[List[Individual]] = None) -> List[Individual]:
        if population is None:
            population = self.population
        fronts = self.assign_rank_and_crowding(population)
        return [ind.copy() for ind in fronts[0]] if fronts else []

    # =========================
    # 排序 / 最优解维护
    # =========================

    def sort_population(self, population: Optional[List[Individual]] = None) -> List[Individual]:
        if population is None:
            population = self.population

        if any(ind.makespan is None or ind.shortage is None for ind in population):
            raise ValueError("存在未评价个体，不能排序")

        self.assign_rank_and_crowding(population)

        return sorted(
            population,
            key=lambda ind: (ind.rank, -ind.crowding_distance, ind.makespan, ind.shortage)
        )

    def get_best_individual(self, population: Optional[List[Individual]] = None) -> Individual:
        sorted_pop = self.sort_population(population)
        return sorted_pop[0]

    def _update_best(self, population: Optional[List[Individual]] = None) -> None:
        current_best = self.get_best_individual(population)

        if self.best_individual is None:
            self.best_individual = current_best.copy()
            return

        if self.better(current_best, self.best_individual):
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

        best = candidates[0]
        for cand in candidates[1:]:
            if self.better(cand, best):
                best = cand

        return best.copy()

    # =========================
    # 交叉
    # =========================

    def crossover_os(self, os1: List[str], os2: List[str]) -> Tuple[List[str], List[str]]:
        """
        标准 POX（Precedence Operation Crossover）
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
        混合 OS 变异：
        - swap
        - insert
        """
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
        """
        半贪婪 MS 变异：
        - 70% 概率选择更快机器
        - 30% 概率随机选择合法机器
        """
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
        单代进化（双目标 NSGA-II 风格）：
        1. 生成子代
        2. 评价子代
        3. 父代 + 子代 合并
        4. 非支配排序 + 拥挤距离，保留前 pop_size
        """
        offspring = self.generate_offspring()
        self.evaluate_population(offspring, store_stats=store_stats)

        merged = self.population + offspring
        fronts = self.assign_rank_and_crowding(merged)

        new_population: List[Individual] = []

        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                front_sorted = sorted(
                    front,
                    key=lambda ind: (ind.rank, -ind.crowding_distance, ind.makespan, ind.shortage)
                )
                new_population.extend(ind.copy() for ind in front_sorted)
            else:
                remaining = self.pop_size - len(new_population)
                front_sorted = sorted(front, key=lambda ind: ind.crowding_distance, reverse=True)
                new_population.extend(ind.copy() for ind in front_sorted[:remaining])
                break

        self.population = new_population
        self.assign_rank_and_crowding(self.population)
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
        self.assign_rank_and_crowding(self.population)

        init_best = self.get_best_individual(self.population)
        self.history_best_fitness = [float(init_best.makespan)]
        self.history_best_shortage = [float(init_best.shortage)]

        if verbose:
            print(
                f"[Init] rep_makespan = {init_best.makespan}, "
                f"rep_shortage = {init_best.shortage}, "
                f"rank = {init_best.rank}"
            )

        for gen in range(1, self.n_generations + 1):
            self.run_one_generation(store_stats=store_stats_generations)

            best = self.get_best_individual(self.population)
            self.history_best_fitness.append(float(best.makespan))
            self.history_best_shortage.append(float(best.shortage))

            if verbose:
                pareto_size = len(self.get_pareto_front(self.population))
                print(
                    f"[Gen {gen}] rep_makespan = {best.makespan}, "
                    f"rep_shortage = {best.shortage}, "
                    f"rank = {best.rank}, "
                    f"pareto_size = {pareto_size}"
                )

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
                f"[{i}] rank={ind.rank}, crowd={ind.crowding_distance:.3f}, "
                f"makespan={ind.makespan}, shortage={ind.shortage}, "
                f"OS_len={len(ind.OS)}, MS_len={len(ind.MS)}"
            )

    def print_best_summary(self) -> None:
        if self.best_individual is None:
            print("best individual is None")
            return

        ind = self.best_individual
        print("===== Representative Individual =====")
        print(f"rank     : {ind.rank}")
        print(f"crowd    : {ind.crowding_distance}")
        print(f"makespan : {ind.makespan}")
        print(f"shortage : {ind.shortage}")
        print(f"OS       : {ind.OS}")
        print(f"MS       : {ind.MS}")

        if ind.stats is not None:
            total_blocking = ind.stats["blocking"]["total_blocking_time"]
            print(f"blocking : {total_blocking}")