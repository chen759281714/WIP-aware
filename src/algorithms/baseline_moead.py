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
    """
    个体表示：
    - OS: job-based operation sequence
    - MS: machine selection list（与 encoder.ms_index_order 对齐）
    """
    OS: List[str]
    MS: List[str]

    fitness: Optional[float] = None   # 兼容字段
    makespan: Optional[int] = None
    shortage: Optional[float] = None

    schedule: Optional[List[Dict[str, Any]]] = None
    buffer_trace: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None

    # MOEA/D 里常用，记录该个体对应的权重向量编号（可选）
    subproblem_id: Optional[int] = None

    def copy(self) -> "Individual":
        return Individual(
            OS=self.OS[:],
            MS=self.MS[:],
            fitness=self.fitness,
            makespan=self.makespan,
            shortage=self.shortage,
            schedule=[rec.copy() for rec in self.schedule] if self.schedule is not None else None,
            buffer_trace={k: v[:] for k, v in self.buffer_trace.items()} if self.buffer_trace is not None else None,
            stats=self.stats.copy() if self.stats is not None else None,
            subproblem_id=self.subproblem_id,
        )


class BaselineMOEAD:
    """
    基础版 MOEA/D（面向双目标）：
    - 双目标：min makespan, min shortage
    - decomposition-based baseline
    - 使用 Tchebycheff 标量化
    - 与现有 NSGA-II 保持相近编码/交叉/变异风格
    - 使用统一 max_evaluations 作为停止条件
    """

    def __init__(
        self,
        operations: Dict[str, List[Dict[str, Any]]],
        buffers: Dict[str, Dict[str, Any]],
        pop_size: int = 100,
        n_generations: int = 100,
        max_evaluations: Optional[int] = None,
        snapshot_interval: Optional[int] = None,
        crossover_rate: float = 0.9,
        os_mutation_rate: float = 0.15,
        ms_mutation_rate: float = 0.15,
        neighborhood_size: int = 10,
        neighbor_mating_prob: float = 0.9,
        max_replace: int = 2,
        seed: Optional[int] = None,
    ):
        if pop_size <= 1:
            raise ValueError("pop_size 必须 > 1")
        if n_generations <= 0:
            raise ValueError("n_generations 必须 > 0")
        if max_evaluations is not None and max_evaluations <= 0:
            raise ValueError("max_evaluations 必须 > 0")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate 必须在 [0,1] 内")
        if not (0.0 <= os_mutation_rate <= 1.0):
            raise ValueError("os_mutation_rate 必须在 [0,1] 内")
        if not (0.0 <= ms_mutation_rate <= 1.0):
            raise ValueError("ms_mutation_rate 必须在 [0,1] 内")
        if neighborhood_size <= 0:
            raise ValueError("neighborhood_size 必须 > 0")
        if not (0.0 <= neighbor_mating_prob <= 1.0):
            raise ValueError("neighbor_mating_prob 必须在 [0,1] 内")
        if max_replace <= 0:
            raise ValueError("max_replace 必须 > 0")
        if snapshot_interval is not None and snapshot_interval <= 0:
            raise ValueError("snapshot_interval 必须 > 0")

        self.operations = operations
        self.buffers = buffers

        self.pop_size = pop_size
        self.n_generations = n_generations
        self.max_evaluations = max_evaluations
        self.snapshot_interval = snapshot_interval
        self.n_evaluations = 0

        self.crossover_rate = crossover_rate
        self.os_mutation_rate = os_mutation_rate
        self.ms_mutation_rate = ms_mutation_rate

        self.neighborhood_size = min(neighborhood_size, pop_size)
        self.neighbor_mating_prob = neighbor_mating_prob
        self.max_replace = max_replace

        self.seed = seed
        self.rng = random.Random(seed)

        self.encoder = Encoder(self.operations, rng=self.rng)
        self.scheduler = StageBufferWIPScheduler(self.operations, self.buffers)

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None

        self.history_best_fitness: List[float] = []
        self.history_best_shortage: List[float] = []

        # 按评价次数记录历史，供后续离线画 GD/IGD 收敛曲线
        self.history_eval_counts: List[int] = []
        self.history_fronts: List[Dict[str, Any]] = []
        self._last_snapshot_eval: int = -1

        # MOEA/D 核心数据
        self.weight_vectors: List[Tuple[float, float]] = []
        self.neighborhoods: List[List[int]] = []
        self.ideal_point: List[float] = [float("inf"), float("inf")]

    # =========================================================
    # 基础工具
    # =========================================================

    def has_budget(self) -> bool:
        return self.max_evaluations is None or self.n_evaluations < self.max_evaluations

    def remaining_budget(self) -> Optional[int]:
        if self.max_evaluations is None:
            return None
        return max(0, self.max_evaluations - self.n_evaluations)

    # =========================================================
    # 初始化
    # =========================================================

    def initialize_individual(self, subproblem_id: Optional[int] = None) -> Individual:
        os_seq = self.encoder.generate_random_os()
        ms_list = self.encoder.generate_random_ms()
        return Individual(OS=os_seq, MS=ms_list, subproblem_id=subproblem_id)

    def build_weight_vectors(self) -> None:
        """
        双目标情形下，均匀生成权重向量：
        (0,1), (1/(N-1), 1-...), ..., (1,0)
        """
        self.weight_vectors = []

        if self.pop_size == 1:
            self.weight_vectors.append((0.5, 0.5))
            return

        for i in range(self.pop_size):
            w1 = i / (self.pop_size - 1)
            w2 = 1.0 - w1
            self.weight_vectors.append((w1, w2))

    def build_neighborhoods(self) -> None:
        """
        根据权重向量间欧氏距离构造邻域
        """
        self.neighborhoods = []

        for i, wi in enumerate(self.weight_vectors):
            dists = []
            for j, wj in enumerate(self.weight_vectors):
                dist = math.sqrt((wi[0] - wj[0]) ** 2 + (wi[1] - wj[1]) ** 2)
                dists.append((dist, j))
            dists.sort(key=lambda x: x[0])
            neigh = [j for _, j in dists[:self.neighborhood_size]]
            self.neighborhoods.append(neigh)

    def initialize_population(self) -> List[Individual]:
        self.build_weight_vectors()
        self.build_neighborhoods()

        self.population = [
            self.initialize_individual(subproblem_id=i)
            for i in range(self.pop_size)
        ]
        return self.population

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
        ind.fitness = float(makespan)

        self.n_evaluations += 1

        self.update_ideal_point(ind)

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
            self.evaluate_individual(ind, store_stats=store_stats)

        evaluated = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]
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
            population = self.population

        if not population:
            return []

        valid_pop = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]

        pareto = []
        for p in valid_pop:
            dominated = False
            for q in valid_pop:
                if p is q:
                    continue
                if self.dominates(q, p):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p.copy())

        return pareto
    
    def export_pareto_front_snapshot(
        self,
        population: Optional[List[Individual]] = None
    ) -> List[Dict[str, Any]]:
        """
        导出当前 Pareto front 的精简快照，用于后续离线计算 GD/IGD 收敛曲线。
        不保存完整 schedule / stats，避免 JSON 过大。
        """
        front = self.get_pareto_front(population)
        snapshot = []

        for ind in front:
            snapshot.append({
                "makespan": ind.makespan,
                "shortage": ind.shortage,
                "subproblem_id": ind.subproblem_id,
                "OS": ind.OS[:],
                "MS": ind.MS[:],
            })

        return snapshot

    def record_front_snapshot(self, force: bool = False) -> None:
        """
        按评价次数记录当前种群的 Pareto front 快照。
        - 若 snapshot_interval is None，则不记录
        - force=True 时强制记录（常用于初始化结束 / 算法结束）
        """
        if self.snapshot_interval is None:
            return

        if not self.population:
            return

        current_eval = int(self.n_evaluations)

        if not force:
            if current_eval <= 0:
                return
            if current_eval - self._last_snapshot_eval < self.snapshot_interval:
                return

        # 避免同一 eval_count 重复记录
        if self.history_fronts and self.history_fronts[-1]["eval_count"] == current_eval:
            return

        snapshot = {
            "eval_count": current_eval,
            "pareto_front": self.export_pareto_front_snapshot(self.population)
        }

        self.history_fronts.append(snapshot)
        self._last_snapshot_eval = current_eval

    def get_best_individual(self, population: Optional[List[Individual]] = None) -> Individual:
        """
        为兼容现有实验脚本，返回一个“代表解”：
        - 先取 Pareto front
        - 再按 makespan, shortage 排序，取第一个
        """
        if population is None:
            population = self.population

        pareto = self.get_pareto_front(population)
        if not pareto:
            raise ValueError("population 中没有可用个体")

        pareto.sort(key=lambda ind: (ind.makespan, ind.shortage))
        return pareto[0]

    def _update_best(self, population: Optional[List[Individual]] = None) -> None:
        if population is None:
            population = self.population

        current_best = self.get_best_individual(population)

        if self.best_individual is None:
            self.best_individual = current_best.copy()
            return

        cur_obj = (current_best.makespan, current_best.shortage)
        best_obj = (self.best_individual.makespan, self.best_individual.shortage)

        if cur_obj < best_obj:
            self.best_individual = current_best.copy()

    # =========================================================
    # MOEA/D 核心
    # =========================================================

    def update_ideal_point(self, ind: Individual) -> None:
        f1, f2 = self.get_objectives(ind)
        self.ideal_point[0] = min(self.ideal_point[0], f1)
        self.ideal_point[1] = min(self.ideal_point[1], f2)

    def scalarizing_function(self, ind: Individual, weight: Tuple[float, float]) -> float:
        """
        Tchebycheff 标量化
        """
        f1, f2 = self.get_objectives(ind)
        z1, z2 = self.ideal_point

        # 避免权重为0时完全失效，按 MOEA/D 常见处理给极小值
        w1 = weight[0] if weight[0] > 1e-12 else 1e-6
        w2 = weight[1] if weight[1] > 1e-12 else 1e-6

        return max(
            w1 * abs(f1 - z1),
            w2 * abs(f2 - z2)
        )

    def choose_mating_indices(self, subproblem_idx: int) -> List[int]:
        if self.rng.random() < self.neighbor_mating_prob:
            return self.neighborhoods[subproblem_idx][:]
        return list(range(len(self.population)))

    # =========================================================
    # 交叉
    # =========================================================

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

    # =========================================================
    # 变异
    # =========================================================

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
        ind.schedule = None
        ind.buffer_trace = None
        ind.stats = None

    def mutate(self, ind: Individual) -> Individual:
        new_ind = ind.copy()
        new_ind.OS = self.mutate_os(new_ind.OS)
        new_ind.MS = self.mutate_ms(new_ind.MS)

        self.reset_individual_evaluation(new_ind)
        return new_ind

    # =========================================================
    # 单步进化
    # =========================================================

    def evolve_subproblem(self, i: int, store_stats: bool = False) -> bool:
        """
        对第 i 个子问题执行一次 MOEA/D 更新。
        返回：
        - True: 发生了至少一次新评价
        - False: 因预算不足或其他原因未进行新评价
        """
        if not self.has_budget():
            return False

        mating_pool = self.choose_mating_indices(i)
        if len(mating_pool) < 2:
            mating_pool = list(range(len(self.population)))

        p1_idx, p2_idx = self.rng.sample(mating_pool, 2)
        parent1 = self.population[p1_idx]
        parent2 = self.population[p2_idx]

        child1, child2 = self.crossover(parent1, parent2)

        child1 = self.mutate(child1)
        child2 = self.mutate(child2)

        # 随机选一个子代进行更新（基础版）
        child = child1 if self.rng.random() < 0.5 else child2
        child.subproblem_id = i

        self.evaluate_individual(child, store_stats=store_stats)

        # 在邻域内更新
        update_indices = self.neighborhoods[i] if self.rng.random() < self.neighbor_mating_prob else list(range(len(self.population)))
        self.rng.shuffle(update_indices)

        replace_count = 0
        for idx in update_indices:
            weight = self.weight_vectors[idx]
            old_value = self.scalarizing_function(self.population[idx], weight)
            new_value = self.scalarizing_function(child, weight)

            if new_value <= old_value:
                replacement = child.copy()
                replacement.subproblem_id = idx
                self.population[idx] = replacement
                replace_count += 1

            if replace_count >= self.max_replace:
                break

        self._update_best([child])
        return True

    def run_one_generation(self, store_stats: bool = False) -> None:
        if not self.has_budget():
            return

        updated = False
        order = list(range(self.pop_size))
        self.rng.shuffle(order)

        for i in order:
            if not self.has_budget():
                break
            did_eval = self.evolve_subproblem(i, store_stats=store_stats)
            if did_eval:
                updated = True

        if not updated:
            return

    # =========================================================
    # 主循环
    # =========================================================

    def run(
        self,
        store_stats_init: bool = True,
        store_stats_generations: bool = False,
        verbose: bool = True
    ) -> Individual:
        self.n_evaluations = 0
        self.population = []
        self.best_individual = None

        self.history_best_fitness = []
        self.history_best_shortage = []
        self.history_eval_counts = []
        self.history_fronts = []
        self._last_snapshot_eval = -1

        self.ideal_point = [float("inf"), float("inf")]

        self.initialize_population()
        self.evaluate_population(self.population, store_stats=store_stats_init)

        # 只保留真正完成评价的初始化个体
        self.population = [
            ind for ind in self.population
            if ind.makespan is not None and ind.shortage is not None
        ]

        if len(self.population) < self.pop_size:
            raise RuntimeError(
                f"初始化阶段预算不足，MOEA/D 仅完成了 {len(self.population)}/{self.pop_size} 个个体评价"
            )

        init_best = self.get_best_individual(self.population)
        self.history_best_fitness = [float(init_best.makespan)]
        self.history_best_shortage = [float(init_best.shortage)]
        self.history_eval_counts = [int(self.n_evaluations)]

        self.record_front_snapshot(force=True)

        if verbose:
            print(
                f"[Init] rep_makespan = {init_best.makespan}, "
                f"rep_shortage = {init_best.shortage}, "
                f"pareto_size = {len(self.get_pareto_front(self.population))}, "
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

            best = self.get_best_individual(self.population)
            self.history_best_fitness.append(float(best.makespan))
            self.history_best_shortage.append(float(best.shortage))
            self.history_eval_counts.append(int(self.n_evaluations))

            self.record_front_snapshot(force=False)

            if verbose:
                pareto_size = len(self.get_pareto_front(self.population))
                print(
                    f"[Gen {gen}] rep_makespan = {best.makespan}, "
                    f"rep_shortage = {best.shortage}, "
                    f"pareto_size = {pareto_size}, "
                    f"evals = {self.n_evaluations}"
                )

        self.record_front_snapshot(force=True)
        return self.best_individual.copy()

    # =========================================================
    # 调试 / 展示
    # =========================================================

    def print_population_summary(self, population: Optional[List[Individual]] = None, top_k: int = 5) -> None:
        if population is None:
            population = self.population

        if not population:
            print("population is empty")
            return

        pareto = self.get_pareto_front(population)
        print(f"Population size: {len(population)}")
        print(f"Pareto size: {len(pareto)}")

        valid_pop = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]
        valid_pop.sort(key=lambda ind: (ind.makespan, ind.shortage))

        k = min(top_k, len(valid_pop))
        print(f"Top {k} individuals:")
        for i in range(k):
            ind = valid_pop[i]
            print(
                f"[{i}] makespan={ind.makespan}, shortage={ind.shortage}, "
                f"subproblem_id={ind.subproblem_id}, "
                f"OS_len={len(ind.OS)}, MS_len={len(ind.MS)}"
            )

    def print_best_summary(self) -> None:
        if self.best_individual is None:
            print("best individual is None")
            return

        ind = self.best_individual
        print("===== Representative Individual =====")
        print(f"makespan     : {ind.makespan}")
        print(f"shortage     : {ind.shortage}")
        print(f"subproblem_id: {ind.subproblem_id}")
        print(f"OS           : {ind.OS}")
        print(f"MS           : {ind.MS}")

        if ind.stats is not None:
            total_blocking = ind.stats["blocking"]["total_blocking_time"]
            print(f"blocking     : {total_blocking}")