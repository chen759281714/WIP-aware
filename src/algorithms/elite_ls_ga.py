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


class EliteLSGA:
    """
    单种群 GA + blocking-triggered elite local search

    这是从原三模式统一代码中抽取出的“elite”完整独立版本：
    - 不包含 offspring local search
    - 不包含 ls_apply_mode 切换
    - 只保留当前已验证效果更好的 elite LS 逻辑
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
        elite_ls_count: int = 2,
        ls_max_tries: int = 4,
        ls_blocking_threshold: int = 3,
        ls_require_positive_blocking: bool = True,
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
        if elite_ls_count < 0:
            raise ValueError("elite_ls_count 必须 >= 0")
        if ls_max_tries <= 0:
            raise ValueError("ls_max_tries 必须 > 0")
        if ls_blocking_threshold < 0:
            raise ValueError("ls_blocking_threshold 必须 >= 0")

        self.operations = operations
        self.buffers = buffers

        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.os_mutation_rate = os_mutation_rate
        self.ms_mutation_rate = ms_mutation_rate
        self.tournament_size = tournament_size
        self.seed = seed

        # elite local search 参数
        self.elite_ls_count = elite_ls_count
        self.ls_max_tries = ls_max_tries
        self.ls_blocking_threshold = ls_blocking_threshold
        self.ls_require_positive_blocking = ls_require_positive_blocking

        self.rng = random.Random(seed)

        self.encoder = Encoder(self.operations, rng=self.rng)
        self.scheduler = StageBufferWIPScheduler(self.operations, self.buffers)

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history_best_fitness: List[float] = []

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
        标准 POX（Precedence Operation Crossover）

        做法：
        - 随机选取一个 job 子集 keep_jobs
        - child1 保留 os1 中 keep_jobs 的相对位置
        - 其余空位按 os2 中非 keep_jobs 的出现顺序填充
        - child2 对称处理
        """
        jobs = list(self.operations.keys())
        n_jobs = len(jobs)

        if n_jobs <= 1:
            return os1[:], os2[:]

        k = self.rng.randint(1, n_jobs - 1)
        keep_jobs = set(self.rng.sample(jobs, k))

        child1 = [None] * len(os1)
        child2 = [None] * len(os2)

        # child1: 保留父代1中 keep_jobs 的位置
        for i, gene in enumerate(os1):
            if gene in keep_jobs:
                child1[i] = gene

        # child2: 保留父代2中 keep_jobs 的位置
        for i, gene in enumerate(os2):
            if gene in keep_jobs:
                child2[i] = gene

        # 用另一个父代中“不属于 keep_jobs”的基因按顺序填空
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

        目的：
        - swap 负责基础扰动
        - insert 更适合长序列，通常对调度顺序影响更有效
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

        else:  # insert
            i, j = self.rng.sample(range(len(new_os)), 2)
            gene = new_os.pop(i)
            new_os.insert(j, gene)

        return new_os

    def mutate_ms(self, ms_list: List[str]) -> List[str]:
        """
        半贪婪 MS 变异：
        - 70% 概率选择更快机器
        - 30% 概率随机选择合法机器

        说明：
        - 仍保留随机性，避免过早收敛
        - 但比纯随机更有方向性
        """
        new_ms = ms_list[:]

        for idx, (job, op) in enumerate(self.encoder.ms_index_order):
            if self.rng.random() < self.ms_mutation_rate:
                machine_dict = self.operations[job][op]["machines"]
                legal_machines = list(machine_dict.keys())

                if len(legal_machines) == 1:
                    new_ms[idx] = legal_machines[0]
                    continue

                # 70% 选最快机器，30% 随机
                if self.rng.random() < 0.7:
                    best_machine = min(legal_machines, key=lambda m: machine_dict[m])
                    new_ms[idx] = best_machine
                else:
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
    # WIP-aware 局部搜索工具函数
    # =========================

    def get_total_blocking_time(self, ind: Individual) -> int:
        if ind.stats is None:
            raise ValueError("ind.stats is None，无法提取 total_blocking_time")
        return int(ind.stats["blocking"]["total_blocking_time"])

    def should_trigger_local_search(self, ind: Individual) -> bool:
        if ind.stats is None:
            raise ValueError("ind.stats is None，无法判断是否触发局部搜索")

        total_blocking = self.get_total_blocking_time(ind)

        if not self.ls_require_positive_blocking:
            return True

        return total_blocking >= self.ls_blocking_threshold

    def get_critical_blocking_ops(self, ind: Individual, top_k: int = 3) -> List[Dict[str, Any]]:
        if ind.stats is None:
            raise ValueError("ind.stats is None，无法提取 blocking 工序")

        op_blocking = ind.stats["blocking"]["per_op_blocking_time"]
        op_blocking = sorted(op_blocking, key=lambda x: x["blocking"], reverse=True)
        op_blocking = [x for x in op_blocking if x["blocking"] > 0]

        return op_blocking[:top_k]

    def find_os_positions_of_job(self, os_seq: List[str], job: str) -> List[int]:
        return [i for i, g in enumerate(os_seq) if g == job]

    # =========================
    # 邻域生成
    # =========================

    def neighbor_ms_reassign(self, ind: Individual, target_job: str, target_op: int) -> List[Individual]:
        neighbors = []

        target_idx = None
        for idx, (job, op) in enumerate(self.encoder.ms_index_order):
            if job == target_job and op == target_op:
                target_idx = idx
                break

        if target_idx is None:
            return neighbors

        current_machine = ind.MS[target_idx]
        legal_machines = list(self.operations[target_job][target_op]["machines"].keys())

        for m in legal_machines:
            if m == current_machine:
                continue

            nei = ind.copy()
            nei.MS[target_idx] = m
            self.reset_individual_evaluation(nei)
            neighbors.append(nei)

        return neighbors

    def neighbor_os_swap(self, ind: Individual, target_job: str, max_neighbors: int = 3) -> List[Individual]:
        neighbors = []

        positions = self.find_os_positions_of_job(ind.OS, target_job)
        if not positions:
            return neighbors

        tried_pairs = set()
        max_attempt_loops = max_neighbors * 5
        attempt = 0

        while len(neighbors) < max_neighbors and attempt < max_attempt_loops:
            attempt += 1

            i = self.rng.choice(positions)
            j = self.rng.randrange(len(ind.OS))

            if i == j:
                continue

            pair = tuple(sorted((i, j)))
            if pair in tried_pairs:
                continue
            tried_pairs.add(pair)

            nei = ind.copy()
            nei.OS[i], nei.OS[j] = nei.OS[j], nei.OS[i]
            self.reset_individual_evaluation(nei)
            neighbors.append(nei)

        return neighbors
    
    def neighbor_os_insert(self, ind: Individual, target_job: str, max_neighbors: int = 3) -> List[Individual]:
        """
        围绕 target_job 生成若干 OS insert 邻居

        做法：
        - 先找到 target_job 在 OS 中的出现位置
        - 随机选一个出现位置 i
        - 再随机选一个插入位置 j
        - 将 OS[i] 取出后插入到 j
        """
        neighbors = []

        positions = self.find_os_positions_of_job(ind.OS, target_job)
        if not positions:
            return neighbors

        tried_moves = set()
        max_attempt_loops = max_neighbors * 8
        attempt = 0

        while len(neighbors) < max_neighbors and attempt < max_attempt_loops:
            attempt += 1

            i = self.rng.choice(positions)
            j = self.rng.randrange(len(ind.OS))

            if i == j:
                continue

            move = (i, j)
            if move in tried_moves:
                continue
            tried_moves.add(move)

            nei = ind.copy()

            gene = nei.OS.pop(i)
            nei.OS.insert(j, gene)

            self.reset_individual_evaluation(nei)
            neighbors.append(nei)

        return neighbors

    # =========================
    # 局部搜索
    # =========================

    def local_search(self, ind: Individual) -> Individual:
        current = ind.copy()

        if current.fitness is None or current.stats is None:
            self.evaluate_individual(current, store_stats=True)

        if not self.should_trigger_local_search(current):
            return current

        critical_ops = self.get_critical_blocking_ops(current, top_k=3)
        if not critical_ops:
            return current

        tries = 0

        for rec in critical_ops:
            if tries >= self.ls_max_tries:
                break

            job = rec["job"]
            op = rec["op"]

            ms_neighbors = self.neighbor_ms_reassign(current, job, op)
            for nei in ms_neighbors:
                self.evaluate_individual(nei, store_stats=True)
                tries += 1

                if nei.fitness < current.fitness:
                    return nei

                if tries >= self.ls_max_tries:
                    return current

            os_neighbors = self.neighbor_os_swap(current, job, max_neighbors=2)
            for nei in os_neighbors:
                self.evaluate_individual(nei, store_stats=True)
                tries += 1

                if nei.fitness < current.fitness:
                    return nei

                if tries >= self.ls_max_tries:
                    return current
                
            insert_neighbors = self.neighbor_os_insert(current, job, max_neighbors=2)
            for nei in insert_neighbors:
                self.evaluate_individual(nei, store_stats=True)
                tries += 1

                if nei.fitness < current.fitness:
                    return nei

                if tries >= self.ls_max_tries:
                    return current

        return current

    # =========================
    # Elite local search 应用
    # =========================

    def apply_local_search_to_elites(self) -> None:
        if not self.population:
            return

        self.population = self.sort_population(self.population)

        n = min(self.elite_ls_count, len(self.population))
        for i in range(n):
            improved = self.local_search(self.population[i])
            self.population[i] = improved

        self.population = self.sort_population(self.population)
        self._update_best(self.population)

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
        单代进化：
        1. 生成子代
        2. 评价子代
        3. 父代 + 子代 合并
        4. 截断保留前 pop_size
        5. 对精英做局部搜索
        6. 更新 best
        """
        offspring = self.generate_offspring()
        self.evaluate_population(offspring, store_stats=store_stats)

        merged = self.population + offspring
        merged_sorted = self.sort_population(merged)

        self.population = [ind.copy() for ind in merged_sorted[:self.pop_size]]

        # elite-based local search
        self.apply_local_search_to_elites()

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