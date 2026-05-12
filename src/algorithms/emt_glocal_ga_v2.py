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

    # ---------- 主任务（真实 WIP）上的评价 ----------
    fitness: Optional[float] = None
    makespan: Optional[int] = None
    shortage: Optional[float] = None

    rank: Optional[int] = None
    crowding_distance: float = 0.0

    schedule: Optional[List[Dict[str, Any]]] = None
    buffer_trace: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None

    # ---------- 兼容旧字段 ----------
    # 新版本不再使用 NoWIP/GAT，但保留字段，避免旧实验脚本读取时报错
    global_fitness: Optional[float] = None
    global_makespan: Optional[int] = None

    # ---------- 调试字段 ----------
    # "main" / "critical" / "local" / None
    origin_task: Optional[str] = None

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
            global_fitness=self.global_fitness,
            global_makespan=self.global_makespan,
            origin_task=self.origin_task,
        )


class EMTGLocalGAV2:
    """
    CPAT-LAT 三种群协同进化算法：

    1) Main Task Population (MT)
       - 真实 WIP / buffer 约束
       - 双目标：min makespan, min shortage
       - 使用非支配排序 + 拥挤距离维护 Pareto 解集

    2) Critical-Path Auxiliary Task Population (CPAT)
       - 使用真实 WIP / buffer 约束评价
       - 围绕 makespan 关键结构构造邻域
       - 环境选择按 (makespan, shortage) 升序
       - 替代原 GAT，不再使用 NoWIP 解码器

    3) Local Auxiliary Task Population (LAT)
       - 沿用 shortage-aware 邻域结构
       - 环境选择按 (shortage, makespan) 升序

    接口保持与原 EMTGLocalGAV2 尽量一致：
    - global_pop_size 在本版本中表示 critical_pop_size
    - global_active / global_best_history 等旧字段保留为兼容字段，但不参与算法
    """

    def __init__(
        self,
        operations: Dict[str, List[Dict[str, Any]]],
        buffers: Dict[str, Dict[str, Any]],
        pop_size: int = 100,
        n_generations: int = 100,
        max_evaluations: Optional[int] = None,
        snapshot_interval: Optional[int] = None,
        crossover_rate: float = 0.8,
        os_mutation_rate: float = 0.2,
        ms_mutation_rate: float = 0.2,
        tournament_size: int = 2,
        seed: Optional[int] = None,

        # ----- EMT / multi-pop 参数 -----
        # 为兼容旧实验脚本，这里 global_pop_size 被解释为 critical_pop_size
        global_pop_size: Optional[int] = None,
        local_pop_size: Optional[int] = None,

        # ----- 旧 GAT 参数：保留接口但不再使用 -----
        gat_improve_window: int = 5,
        gat_improve_threshold: float = 0.005,

        # ----- 局部辅助任务参数 -----
        local_elite_count: int = 6,
        local_neighbors_per_elite: int = 6,
        local_os_mutation_rate: float = 0.35,
        local_ms_mutation_rate: float = 0.35,
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
        if gat_improve_window <= 0:
            raise ValueError("gat_improve_window 必须 > 0")
        if gat_improve_threshold < 0:
            raise ValueError("gat_improve_threshold 必须 >= 0")
        if local_elite_count <= 0:
            raise ValueError("local_elite_count 必须 > 0")
        if local_neighbors_per_elite <= 0:
            raise ValueError("local_neighbors_per_elite 必须 > 0")
        if not (0.0 <= local_os_mutation_rate <= 1.0):
            raise ValueError("local_os_mutation_rate 必须在 [0,1] 内")
        if not (0.0 <= local_ms_mutation_rate <= 1.0):
            raise ValueError("local_ms_mutation_rate 必须在 [0,1] 内")
        if max_evaluations is not None and max_evaluations <= 0:
            raise ValueError("max_evaluations 必须 > 0")
        if snapshot_interval is not None and snapshot_interval <= 0:
            raise ValueError("snapshot_interval 必须 > 0")
        if global_pop_size is not None and global_pop_size <= 0:
            raise ValueError("global_pop_size/critical_pop_size 必须 > 0")
        if local_pop_size is not None and local_pop_size <= 0:
            raise ValueError("local_pop_size 必须 > 0")

        self.operations = operations
        self.buffers = buffers

        self.pop_size = pop_size
        self.critical_pop_size = global_pop_size if global_pop_size is not None else pop_size
        self.global_pop_size = self.critical_pop_size  # 兼容旧字段名
        self.local_pop_size = local_pop_size if local_pop_size is not None else pop_size

        self.n_generations = n_generations
        self.max_evaluations = max_evaluations
        self.snapshot_interval = snapshot_interval
        self.n_evaluations = 0

        self.crossover_rate = crossover_rate
        self.os_mutation_rate = os_mutation_rate
        self.ms_mutation_rate = ms_mutation_rate
        self.tournament_size = tournament_size
        self.seed = seed

        # 旧 GAT 参数保留，但不参与控制
        self.gat_improve_window = gat_improve_window
        self.gat_improve_threshold = gat_improve_threshold

        self.local_elite_count = local_elite_count
        self.local_neighbors_per_elite = local_neighbors_per_elite
        self.local_os_mutation_rate = local_os_mutation_rate
        self.local_ms_mutation_rate = local_ms_mutation_rate

        self.rng = random.Random(seed)

        self.encoder = Encoder(self.operations, rng=self.rng)
        self.main_scheduler = StageBufferWIPScheduler(self.operations, self.buffers)

        # 三个种群
        self.main_population: List[Individual] = []
        self.critical_population: List[Individual] = []
        self.local_population: List[Individual] = []

        # 兼容旧代码：global_population 指向 critical_population
        self.global_population: List[Individual] = self.critical_population
        self.population: List[Individual] = self.main_population

        self.best_individual: Optional[Individual] = None

        self.history_best_fitness: List[float] = []
        self.history_best_shortage: List[float] = []
        self.history_eval_counts: List[int] = []
        self.history_fronts: List[Dict[str, Any]] = []
        self._last_snapshot_eval: int = -1

        # 旧 GAT 状态字段，保留兼容
        self.global_active: bool = False
        self.global_best_history: List[float] = []

    # =========================================================
    # 基础工具
    # =========================================================

    def initialize_individual(self, origin_task: Optional[str] = None) -> Individual:
        os_seq = self.encoder.generate_random_os()
        ms_list = self.encoder.generate_random_ms()
        return Individual(OS=os_seq, MS=ms_list, origin_task=origin_task)

    def generate_fastest_ms(self) -> List[str]:
        fastest_ms = []

        for job, op_idx in self.encoder.ms_index_order:
            machine_dict = self.operations[job][op_idx]["machines"]
            fastest_machine = min(machine_dict.keys(), key=lambda m: machine_dict[m])
            fastest_ms.append(fastest_machine)

        return fastest_ms

    def move_operation_in_os(
        self,
        os_seq: List[str],
        job: str,
        op_idx: int,
        shift: int
    ) -> List[str]:
        pos = self.find_os_position_of_operation(os_seq, job, op_idx)
        if pos is None:
            return os_seq

        new_pos = max(0, min(len(os_seq) - 1, pos + shift))
        if new_pos == pos:
            return os_seq

        new_os = os_seq[:]
        gene = new_os.pop(pos)
        new_os.insert(new_pos, gene)
        return new_os

    def sample_buffer_ids(self, init_n_buffers: int) -> List[str]:
        buffer_ids = list(self.buffers.keys())
        if not buffer_ids or init_n_buffers <= 0:
            return []
        k = min(init_n_buffers, len(buffer_ids))
        return self.rng.sample(buffer_ids, k)

    def make_supply_forward_os(
        self,
        os_seq: List[str],
        init_n_buffers: int = 2,
        init_max_ops_per_buffer: int = 2,
        init_max_shift: int = 6
    ) -> List[str]:
        new_os = os_seq[:]

        for buffer_id in self.sample_buffer_ids(init_n_buffers):
            supply_ops = self.get_upstream_supply_ops_of_buffer(buffer_id)
            if not supply_ops:
                continue

            n_ops = min(init_max_ops_per_buffer, len(supply_ops))
            for job, op_idx in self.rng.sample(supply_ops, n_ops):
                shift = -self.rng.randint(1, max(1, init_max_shift))
                new_os = self.move_operation_in_os(new_os, job, op_idx, shift)

        return new_os

    def make_consume_backward_os(
        self,
        os_seq: List[str],
        init_n_buffers: int = 2,
        init_max_ops_per_buffer: int = 2,
        init_max_shift: int = 6
    ) -> List[str]:
        new_os = os_seq[:]

        for buffer_id in self.sample_buffer_ids(init_n_buffers):
            consume_ops = self.get_downstream_consume_ops_of_buffer(buffer_id)
            if not consume_ops:
                continue

            n_ops = min(init_max_ops_per_buffer, len(consume_ops))
            for job, op_idx in self.rng.sample(consume_ops, n_ops):
                shift = self.rng.randint(1, max(1, init_max_shift))
                new_os = self.move_operation_in_os(new_os, job, op_idx, shift)

        return new_os

    def initialize_critical_individual(self, index: int) -> Individual:
        split = self.critical_pop_size // 2

        if index < split:
            return self.initialize_individual(origin_task="critical")

        ind = Individual(
            OS=self.encoder.generate_random_os(),
            MS=self.generate_fastest_ms(),
            origin_task="critical"
        )
        self.reset_individual_evaluation(ind)
        return ind

    def initialize_local_individual(self, index: int) -> Individual:
        random_count = self.local_pop_size // 2
        supply_count = self.local_pop_size // 4
        supply_end = random_count + supply_count

        if index < random_count:
            return self.initialize_individual(origin_task="local")

        ind = self.initialize_individual(origin_task="local")

        if index < supply_end:
            ind.OS = self.make_supply_forward_os(ind.OS)
        else:
            ind.OS = self.make_consume_backward_os(ind.OS)

        ind.origin_task = "local"
        self.reset_individual_evaluation(ind)
        return ind

    def initialize_populations(self) -> None:
        self.main_population = [
            self.initialize_individual(origin_task="main")
            for _ in range(self.pop_size)
        ]
        self.critical_population = [
            self.initialize_critical_individual(i)
            for i in range(self.critical_pop_size)
        ]
        self.local_population = [
            self.initialize_local_individual(i)
            for i in range(self.local_pop_size)
        ]

        # 兼容字段同步
        self.global_population = self.critical_population
        self.population = self.main_population

    def has_budget(self) -> bool:
        return self.max_evaluations is None or self.n_evaluations < self.max_evaluations

    def remaining_budget(self) -> Optional[int]:
        if self.max_evaluations is None:
            return None
        return max(0, self.max_evaluations - self.n_evaluations)

    # =========================================================
    # 评价
    # =========================================================

    def evaluate_individual(
        self,
        ind: Individual,
        task: str = "main"
    ) -> float:
        """
        本版本所有任务均使用真实 WIP / buffer 约束评价。
        task 参数仅用于兼容旧接口。
        """
        if not self.has_budget():
            raise RuntimeError("Evaluation budget exhausted")

        ms_map = self.encoder.build_ms_map(ind.MS)

        makespan, schedule, buffer_trace = self.main_scheduler.decode(
            os_seq=ind.OS,
            ms_map=ms_map
        )

        ind.makespan = makespan
        ind.schedule = schedule
        ind.buffer_trace = buffer_trace
        ind.stats = self.main_scheduler.analyze(
            schedule=schedule,
            buffer_trace=buffer_trace,
            makespan=makespan
        )
        ind.shortage = float(ind.stats["shortage"]["total_shortage_area"])
        ind.fitness = float(makespan)

        # 兼容旧字段：global_makespan 也填 makespan，避免旧分析脚本访问时报错
        if task in {"global", "critical"}:
            ind.global_makespan = makespan
            ind.global_fitness = float(makespan)

        self.n_evaluations += 1
        return ind.fitness

    def evaluate_population_main(
        self,
        population: List[Individual],
        store_stats: bool = True
    ) -> None:
        if not population:
            raise ValueError("population 为空，无法评价主任务")

        for ind in population:
            if not self.has_budget():
                break
            self.evaluate_individual(ind, task="main")

        evaluated = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]
        if evaluated:
            self.assign_rank_and_crowding(evaluated)
            self._update_best(evaluated)

    def evaluate_population_critical(
        self,
        population: List[Individual],
        store_stats: bool = True
    ) -> None:
        if not population:
            raise ValueError("population 为空，无法评价 CPAT")

        for ind in population:
            if not self.has_budget():
                break
            self.evaluate_individual(ind, task="critical")

        evaluated = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]
        if evaluated:
            self._update_best(evaluated)

    # 兼容旧接口
    def evaluate_population_global(self, population: List[Individual]) -> None:
        self.evaluate_population_critical(population, store_stats=False)

    def ensure_evaluated_on_main(
        self,
        population: List[Individual],
        store_stats: bool = True
    ) -> List[Individual]:
        result = []

        for ind in population:
            if ind.makespan is None or ind.shortage is None:
                if not self.has_budget():
                    break
                self.evaluate_individual(ind, task="main")

            if ind.makespan is not None and ind.shortage is not None:
                result.append(ind)

        return result

    def ensure_evaluated_on_global(self, population: List[Individual]) -> List[Individual]:
        return self.ensure_evaluated_on_main(population, store_stats=False)

    # =========================================================
    # 多目标工具
    # =========================================================

    def get_objectives(self, ind: Individual) -> Tuple[float, float]:
        if ind.makespan is None or ind.shortage is None:
            raise ValueError("个体尚未完成主任务双目标评价")
        return float(ind.makespan), float(ind.shortage)

    def dominates(self, a: Individual, b: Individual) -> bool:
        a1, a2 = self.get_objectives(a)
        b1, b2 = self.get_objectives(b)
        return a1 <= b1 and a2 <= b2 and (a1 < b1 or a2 < b2)

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
        valid_population = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]
        if not valid_population:
            return []

        S = {}
        n = {}
        fronts: List[List[Individual]] = [[]]

        for p in valid_population:
            pid = id(p)
            S[pid] = []
            n[pid] = 0

            for q in valid_population:
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

    def sort_population(self, population: Optional[List[Individual]] = None) -> List[Individual]:
        if population is None:
            population = self.main_population

        valid_population = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]
        if not valid_population:
            raise ValueError("population 中没有已评价个体，不能排序")

        self.assign_rank_and_crowding(valid_population)
        return sorted(
            valid_population,
            key=lambda ind: (ind.rank, -ind.crowding_distance, ind.makespan, ind.shortage)
        )

    def get_best_individual(self, population: Optional[List[Individual]] = None) -> Individual:
        if population is None:
            population = self.main_population
        sorted_pop = self.sort_population(population)
        return sorted_pop[0]

    def get_pareto_front(self, population: Optional[List[Individual]] = None) -> List[Individual]:
        if population is None:
            population = self.main_population

        valid_population = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]
        if not valid_population:
            return []

        fronts = self.assign_rank_and_crowding(valid_population)
        return [ind.copy() for ind in fronts[0]] if fronts else []

    def _update_best(self, population: Optional[List[Individual]] = None) -> None:
        if population is None:
            population = self.main_population

        valid_population = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]
        if not valid_population:
            return

        current_best = self.get_best_individual(valid_population)

        if self.best_individual is None:
            self.best_individual = current_best.copy()
            return

        if self.better(current_best, self.best_individual):
            self.best_individual = current_best.copy()

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
                "OS": ind.OS[:],
                "MS": ind.MS[:],
            })

        return snapshot

    def record_front_snapshot(self, force: bool = False) -> None:
        if self.snapshot_interval is None:
            return
        if not self.main_population:
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
            "pareto_front": self.export_pareto_front_snapshot(self.main_population)
        }

        self.history_fronts.append(snapshot)
        self._last_snapshot_eval = current_eval

    # =========================================================
    # 选择 / 交叉 / 变异
    # =========================================================

    def tournament_select_main(self, population: Optional[List[Individual]] = None) -> Individual:
        if population is None:
            population = self.main_population

        valid_population = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]
        if len(valid_population) < self.tournament_size:
            raise ValueError("population 数量小于 tournament_size")

        candidates = self.rng.sample(valid_population, self.tournament_size)
        best = candidates[0]
        for cand in candidates[1:]:
            if self.better(cand, best):
                best = cand
        return best.copy()

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

    def mutate_os(self, os_seq: List[str], mutation_rate: float) -> List[str]:
        new_os = os_seq[:]

        if len(new_os) < 2:
            return new_os

        if self.rng.random() >= mutation_rate:
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

    def mutate_ms(self, ms_list: List[str], mutation_rate: float) -> List[str]:
        new_ms = ms_list[:]

        for idx, (job, op) in enumerate(self.encoder.ms_index_order):
            if self.rng.random() < mutation_rate:
                machine_dict = self.operations[job][op]["machines"]
                legal_machines = list(machine_dict.keys())

                if len(legal_machines) == 1:
                    new_ms[idx] = legal_machines[0]
                    continue

                # 半贪婪：70% 选最快机器，30% 随机
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
        ind.global_fitness = None
        ind.global_makespan = None

    def mutate(
        self,
        ind: Individual,
        os_mut_rate: Optional[float] = None,
        ms_mut_rate: Optional[float] = None
    ) -> Individual:
        if os_mut_rate is None:
            os_mut_rate = self.os_mutation_rate
        if ms_mut_rate is None:
            ms_mut_rate = self.ms_mutation_rate

        new_ind = ind.copy()
        new_ind.OS = self.mutate_os(new_ind.OS, mutation_rate=os_mut_rate)
        new_ind.MS = self.mutate_ms(new_ind.MS, mutation_rate=ms_mut_rate)
        self.reset_individual_evaluation(new_ind)
        return new_ind

    # =========================================================
    # 通用 OS/MS 工具
    # =========================================================

    def find_os_position_of_operation(self, os_seq: List[str], job: str, op_idx: int) -> Optional[int]:
        """
        在 job-based OS 中，第 op_idx 道工序对应的是该 job 第 op_idx+1 次出现的位置
        """
        count = 0
        for pos, g in enumerate(os_seq):
            if g == job:
                if count == op_idx:
                    return pos
                count += 1
        return None

    def get_ms_index(self, target_job: str, target_op: int) -> Optional[int]:
        for idx, (job, op) in enumerate(self.encoder.ms_index_order):
            if job == target_job and op == target_op:
                return idx
        return None

    def make_os_insert_neighbor(
        self,
        ind: Individual,
        job: str,
        op_idx: int,
        new_pos: int,
        origin_task: str
    ) -> Optional[Individual]:
        pos = self.find_os_position_of_operation(ind.OS, job, op_idx)
        if pos is None:
            return None
        if pos == new_pos:
            return None

        new_pos = max(0, min(len(ind.OS) - 1, new_pos))

        nei = ind.copy()
        gene = nei.OS.pop(pos)
        nei.OS.insert(new_pos, gene)
        nei.origin_task = origin_task
        self.reset_individual_evaluation(nei)
        return nei

    def take_neighbors_by_budget(
        self,
        batches: List[List[Individual]],
        budget: int
    ) -> List[Individual]:
        """
        从多个邻域算子结果中轮转取样，避免一个算子独占每个 elite 的邻域预算。
        """
        if budget <= 0:
            return []

        selected: List[Individual] = []
        positions = [0 for _ in batches]

        while len(selected) < budget:
            progressed = False
            for batch_idx, batch in enumerate(batches):
                pos = positions[batch_idx]
                if pos >= len(batch):
                    continue
                selected.append(batch[pos])
                positions[batch_idx] += 1
                progressed = True
                if len(selected) >= budget:
                    break

            if not progressed:
                break

        return selected

    # =========================================================
    # CPAT：关键路径 / makespan-aware 邻域
    # =========================================================

    def get_last_release_record(self, ind: Individual) -> Optional[Dict[str, Any]]:
        if not ind.schedule:
            return None
        return max(ind.schedule, key=lambda r: (int(r.get("release", r.get("end", 0))), int(r.get("end", 0))))

    def get_job_records(self, ind: Individual, job: str) -> List[Dict[str, Any]]:
        if ind.schedule is None:
            return []
        records = [r for r in ind.schedule if r["job"] == job]
        records.sort(key=lambda r: int(r["op"]))
        return records

    def get_machine_records(self, ind: Individual, machine: str) -> List[Dict[str, Any]]:
        if ind.schedule is None:
            return []
        records = [r for r in ind.schedule if r["machine"] == machine]
        records.sort(key=lambda r: (int(r["start"]), int(r["end"]), str(r["job"]), int(r["op"])))
        return records

    def get_machine_load_scores(self, ind: Individual) -> List[Tuple[float, str]]:
        if ind.stats is None:
            return []

        busy = ind.stats.get("machines", {}).get("per_machine_busy_time", {})
        blocking = ind.stats.get("blocking", {}).get("per_machine_blocking_time", {})

        scores = []
        for m in self.main_scheduler.machines:
            b = float(busy.get(m, 0.0))
            blk = float(blocking.get(m, 0.0))
            scores.append((b + blk, m))

        scores.sort(reverse=True, key=lambda x: x[0])
        return scores

    def get_top_blocking_records(self, ind: Individual, top_k: int = 3) -> List[Dict[str, Any]]:
        if not ind.schedule:
            return []

        records = []
        for r in ind.schedule:
            blk = int(r.get("release", r.get("end", 0))) - int(r.get("end", 0))
            if blk > 0 and r.get("buffer_out", None) is not None:
                rr = r.copy()
                rr["blocking"] = blk
                records.append(rr)

        records.sort(key=lambda x: x["blocking"], reverse=True)
        return records[:top_k]

    def neighbor_os_forward_last_job(
        self,
        ind: Individual,
        max_neighbors: int = 4,
        max_shift: int = 8
    ) -> List[Individual]:
        """
        邻域 1：最后释放工件链前移。
        """
        last = self.get_last_release_record(ind)
        if last is None:
            return []

        target_job = last["job"]
        job_records = self.get_job_records(ind, target_job)
        if not job_records:
            return []

        # 从后往前处理该 job 的工序，优先压缩靠近 makespan 的尾部工序
        job_records.sort(key=lambda r: int(r["op"]), reverse=True)

        neighbors = []
        tried = set()

        for rec in job_records:
            job = rec["job"]
            op_idx = int(rec["op"])
            pos = self.find_os_position_of_operation(ind.OS, job, op_idx)
            if pos is None:
                continue

            for shift in range(1, max_shift + 1):
                new_pos = max(0, pos - shift)
                key = (job, op_idx, pos, new_pos)
                if key in tried:
                    continue
                tried.add(key)

                nei = self.make_os_insert_neighbor(ind, job, op_idx, new_pos, origin_task="critical")
                if nei is not None:
                    neighbors.append(nei)

                if len(neighbors) >= max_neighbors:
                    return neighbors

        return neighbors

    def neighbor_ms_reassign_critical_ops(
        self,
        ind: Individual,
        max_neighbors: int = 4,
        max_machines_per_op: int = 2
    ) -> List[Individual]:
        """
        邻域 2：关键工序换更快机器。
        关键工序来自最后完工 job 链 + 高负载机器上的尾部工序。
        """
        critical_ops: List[Tuple[str, int]] = []

        last = self.get_last_release_record(ind)
        if last is not None:
            for rec in self.get_job_records(ind, last["job"]):
                critical_ops.append((rec["job"], int(rec["op"])))

        # 加入前两台瓶颈机器上的后半段工序
        for _, m in self.get_machine_load_scores(ind)[:2]:
            records = self.get_machine_records(ind, m)
            if records:
                tail = records[len(records) // 2:]
                for rec in tail[-3:]:
                    critical_ops.append((rec["job"], int(rec["op"])))

        # 去重但保序
        seen = set()
        unique_ops = []
        for op_key in critical_ops:
            if op_key not in seen:
                seen.add(op_key)
                unique_ops.append(op_key)

        neighbors = []

        for job, op_idx in unique_ops:
            target_idx = self.get_ms_index(job, op_idx)
            if target_idx is None:
                continue

            current_machine = ind.MS[target_idx]
            machine_dict = self.operations[job][op_idx]["machines"]
            legal_machines = sorted(machine_dict.keys(), key=lambda m: machine_dict[m])

            count = 0
            for m in legal_machines:
                if m == current_machine:
                    continue
                # 优先考虑更快机器；若没有更快机器，允许少量其他机器增加多样性
                if machine_dict[m] >= machine_dict[current_machine] and count > 0:
                    continue

                nei = ind.copy()
                nei.MS[target_idx] = m
                nei.origin_task = "critical"
                self.reset_individual_evaluation(nei)
                neighbors.append(nei)

                count += 1
                if count >= max_machines_per_op or len(neighbors) >= max_neighbors:
                    break

            if len(neighbors) >= max_neighbors:
                return neighbors

        return neighbors

    def neighbor_blocking_release_forward_consumer(
        self,
        ind: Individual,
        max_neighbors: int = 4,
        max_shift: int = 8
    ) -> List[Individual]:
        """
        邻域 3：blocking-aware 释放邻域。
        对 blocking 最大的工序，其 buffer_out 满导致释放延迟，
        因此尝试把该 buffer 的下游消费工序前移，让 buffer 更早腾空。
        """
        if ind.schedule is None:
            return []

        neighbors = []
        tried = set()

        for blk_rec in self.get_top_blocking_records(ind, top_k=3):
            buffer_out = blk_rec.get("buffer_out", None)
            if buffer_out is None:
                continue

            consume_records = []
            for rec in ind.schedule:
                job = rec["job"]
                op_idx = int(rec["op"])
                op_def = self.operations[job][op_idx]
                if op_def.get("buffer_in", None) == buffer_out:
                    consume_records.append(rec)

            # 优先前移在阻塞结束附近或之后才发生的消费工序
            release_t = int(blk_rec.get("release", blk_rec.get("end", 0)))
            consume_records.sort(key=lambda r: abs(int(r["start"]) - release_t))

            for rec in consume_records[:4]:
                job = rec["job"]
                op_idx = int(rec["op"])
                pos = self.find_os_position_of_operation(ind.OS, job, op_idx)
                if pos is None:
                    continue

                for shift in range(1, max_shift + 1):
                    new_pos = max(0, pos - shift)
                    key = (job, op_idx, pos, new_pos, buffer_out)
                    if key in tried:
                        continue
                    tried.add(key)

                    nei = self.make_os_insert_neighbor(ind, job, op_idx, new_pos, origin_task="critical")
                    if nei is not None:
                        neighbors.append(nei)

                    if len(neighbors) >= max_neighbors:
                        return neighbors

        return neighbors

    def neighbor_bottleneck_machine_forward(
        self,
        ind: Individual,
        max_neighbors: int = 4,
        max_shift: int = 6
    ) -> List[Individual]:
        """
        邻域 4：高负载机器尾部关键工序前移。
        """
        neighbors = []
        tried = set()

        for _, machine in self.get_machine_load_scores(ind)[:2]:
            records = self.get_machine_records(ind, machine)
            if len(records) < 2:
                continue

            # 优先处理靠后的工序，因为它们更接近 makespan
            for rec in reversed(records[-5:]):
                job = rec["job"]
                op_idx = int(rec["op"])
                pos = self.find_os_position_of_operation(ind.OS, job, op_idx)
                if pos is None:
                    continue

                for shift in range(1, max_shift + 1):
                    new_pos = max(0, pos - shift)
                    key = (machine, job, op_idx, pos, new_pos)
                    if key in tried:
                        continue
                    tried.add(key)

                    nei = self.make_os_insert_neighbor(ind, job, op_idx, new_pos, origin_task="critical")
                    if nei is not None:
                        neighbors.append(nei)

                    if len(neighbors) >= max_neighbors:
                        return neighbors

        return neighbors

    def generate_critical_offspring(self) -> List[Individual]:
        """
        CPAT 生成逻辑：
        - 从主种群 + critical_population 的 makespan 优秀个体中选种子
        - 围绕关键完工链、关键机器、blocking 释放构造邻域
        - 不足则使用轻微扰动补齐
        """
        seed_pool = []
        if self.main_population:
            seed_pool.extend(self.sort_population(self.main_population)[:min(self.local_elite_count, len(self.main_population))])
        if self.critical_population:
            valid_critical = [ind for ind in self.critical_population if ind.makespan is not None and ind.shortage is not None]
            valid_critical.sort(key=lambda ind: (ind.makespan, ind.shortage))
            seed_pool.extend(valid_critical[:min(self.local_elite_count, len(valid_critical))])

        if not seed_pool:
            return [self.initialize_individual(origin_task="critical") for _ in range(self.critical_pop_size)]

        # 去掉重复目标点，避免单一解产生过多邻居
        unique_seed = {}
        for ind in seed_pool:
            if ind.makespan is None or ind.shortage is None:
                continue
            key = (float(ind.makespan), float(ind.shortage))
            if key not in unique_seed:
                unique_seed[key] = ind
        elites = list(unique_seed.values())
        if not elites:
            elites = seed_pool

        candidates: List[Individual] = []

        for ind in elites:
            clone = ind.copy()
            clone.origin_task = "critical"
            candidates.append(clone)

            neighbor_budget = self.local_neighbors_per_elite
            candidates.extend(self.take_neighbors_by_budget(
                batches=[
                    self.neighbor_os_forward_last_job(ind, max_neighbors=neighbor_budget, max_shift=8),
                    self.neighbor_ms_reassign_critical_ops(ind, max_neighbors=neighbor_budget, max_machines_per_op=2),
                    self.neighbor_blocking_release_forward_consumer(ind, max_neighbors=neighbor_budget, max_shift=8),
                    self.neighbor_bottleneck_machine_forward(ind, max_neighbors=neighbor_budget, max_shift=6),
                ],
                budget=neighbor_budget
            ))

        while len(candidates) < self.critical_pop_size:
            base = self.rng.choice(elites).copy()
            nei = self.mutate(
                base,
                os_mut_rate=self.local_os_mutation_rate,
                ms_mut_rate=self.local_ms_mutation_rate
            )
            nei.origin_task = "critical"
            candidates.append(nei)

        self.rng.shuffle(candidates)
        return candidates[:self.critical_pop_size]

    # =========================================================
    # LAT：shortage-aware 邻域，保留原设计
    # =========================================================

    def get_critical_shortage_buffers_from_main(
        self,
        ind: Individual,
        top_k: int = 2
    ) -> List[str]:
        if ind.stats is None:
            return []

        shortage_dict = ind.stats["shortage"].get("per_buffer_shortage_area", {})
        if not shortage_dict:
            return []

        items = sorted(shortage_dict.items(), key=lambda x: x[1], reverse=True)
        items = [x for x in items if x[1] > 0]
        return [bid for bid, _ in items[:top_k]]

    def get_upstream_supply_ops_of_buffer(self, buffer_id: str) -> List[Tuple[str, int]]:
        result = []
        for job, ops in self.operations.items():
            for op_idx, op in enumerate(ops):
                if op.get("buffer_out", None) == buffer_id:
                    result.append((job, op_idx))
        return result

    def get_downstream_consume_ops_of_buffer(self, buffer_id: str) -> List[Tuple[str, int]]:
        result = []
        for job, ops in self.operations.items():
            for op_idx, op in enumerate(ops):
                if op.get("buffer_in", None) == buffer_id:
                    result.append((job, op_idx))
        return result

    def extract_shortage_intervals(
        self,
        ind: Individual,
        buffer_id: str
    ) -> List[Tuple[int, int, int]]:
        if ind.buffer_trace is None or ind.makespan is None:
            return []

        events = ind.buffer_trace.get(buffer_id, [])
        if not events:
            return []

        low_wip = int(self.buffers[buffer_id].get("low_wip", 1))
        T = int(ind.makespan)
        events_sorted = sorted(events, key=lambda x: x[0])
        intervals = []

        for i in range(len(events_sorted) - 1):
            t_i, level_i, _, _ = events_sorted[i]
            t_j, _, _, _ = events_sorted[i + 1]

            t_i = int(t_i)
            t_j = int(t_j)
            level_i = int(level_i)

            if t_j <= t_i:
                continue

            gap = low_wip - level_i
            if gap > 0:
                intervals.append((t_i, t_j, gap))

        last_t, last_level, _, _ = events_sorted[-1]
        last_t = int(last_t)
        last_level = int(last_level)
        if last_t < T:
            gap = low_wip - last_level
            if gap > 0:
                intervals.append((last_t, T, gap))

        return intervals

    def get_most_critical_shortage_interval(
        self,
        ind: Individual,
        buffer_id: str
    ) -> Optional[Tuple[int, int, int]]:
        intervals = self.extract_shortage_intervals(ind, buffer_id)
        if not intervals:
            return None
        return max(intervals, key=lambda x: (x[1] - x[0]) * x[2])

    def neighbor_ms_reassign_supply_to_buffer(
        self,
        ind: Individual,
        buffer_id: str,
        max_neighbors_per_op: int = 2
    ) -> List[Individual]:
        neighbors = []
        supply_ops = self.get_upstream_supply_ops_of_buffer(buffer_id)

        for job, op_idx in supply_ops:
            target_idx = self.get_ms_index(job, op_idx)
            if target_idx is None:
                continue

            current_machine = ind.MS[target_idx]
            machine_dict = self.operations[job][op_idx]["machines"]
            legal_machines = sorted(machine_dict.keys(), key=lambda m: machine_dict[m])

            count = 0
            for m in legal_machines:
                if m == current_machine:
                    continue

                nei = ind.copy()
                nei.MS[target_idx] = m
                nei.origin_task = "local"
                self.reset_individual_evaluation(nei)
                neighbors.append(nei)

                count += 1
                if count >= max_neighbors_per_op:
                    break

        return neighbors

    def neighbor_os_forward_insert_supply_to_buffer(
        self,
        ind: Individual,
        buffer_id: str,
        max_neighbors: int = 3,
        max_shift: int = 6
    ) -> List[Individual]:
        neighbors = []
        supply_ops = self.get_upstream_supply_ops_of_buffer(buffer_id)
        tried = set()

        for job, op_idx in supply_ops:
            pos = self.find_os_position_of_operation(ind.OS, job, op_idx)
            if pos is None:
                continue

            for shift in range(1, max_shift + 1):
                new_pos = max(0, pos - shift)
                move = (job, op_idx, pos, new_pos)
                if move in tried:
                    continue
                tried.add(move)

                nei = self.make_os_insert_neighbor(ind, job, op_idx, new_pos, origin_task="local")
                if nei is not None:
                    neighbors.append(nei)

                if len(neighbors) >= max_neighbors:
                    return neighbors

        return neighbors

    def neighbor_os_backward_insert_consume_from_buffer(
        self,
        ind: Individual,
        buffer_id: str,
        max_neighbors: int = 3,
        max_shift: int = 6
    ) -> List[Individual]:
        neighbors = []
        consume_ops = self.get_downstream_consume_ops_of_buffer(buffer_id)
        tried = set()
        n = len(ind.OS)

        for job, op_idx in consume_ops:
            pos = self.find_os_position_of_operation(ind.OS, job, op_idx)
            if pos is None:
                continue

            for shift in range(1, max_shift + 1):
                new_pos = min(n - 1, pos + shift)
                move = (job, op_idx, pos, new_pos)
                if move in tried:
                    continue
                tried.add(move)

                nei = self.make_os_insert_neighbor(ind, job, op_idx, new_pos, origin_task="local")
                if nei is not None:
                    neighbors.append(nei)

                if len(neighbors) >= max_neighbors:
                    return neighbors

        return neighbors

    def neighbor_shortage_window_alignment(
        self,
        ind: Individual,
        buffer_id: str,
        max_neighbors: int = 4
    ) -> List[Individual]:
        if ind.schedule is None:
            return []

        critical_interval = self.get_most_critical_shortage_interval(ind, buffer_id)
        if critical_interval is None:
            return []

        t_start, t_end, _ = critical_interval
        neighbors = []

        late_supply_ops = []
        for rec in ind.schedule:
            job = rec["job"]
            op_idx = int(rec["op"])
            op_def = self.operations[job][op_idx]
            if op_def.get("buffer_out", None) == buffer_id and rec["end"] > t_start:
                late_supply_ops.append((job, op_idx, rec["end"]))

        late_supply_ops.sort(key=lambda x: x[2])

        for job, op_idx, _ in late_supply_ops[:2]:
            pos = self.find_os_position_of_operation(ind.OS, job, op_idx)
            if pos is None:
                continue

            new_pos = max(0, pos - 4)
            nei = self.make_os_insert_neighbor(ind, job, op_idx, new_pos, origin_task="local")
            if nei is not None:
                neighbors.append(nei)

            if len(neighbors) >= max_neighbors:
                return neighbors

        early_consume_ops = []
        for rec in ind.schedule:
            job = rec["job"]
            op_idx = int(rec["op"])
            op_def = self.operations[job][op_idx]
            if op_def.get("buffer_in", None) == buffer_id and rec["start"] < t_end:
                early_consume_ops.append((job, op_idx, rec["start"]))

        early_consume_ops.sort(key=lambda x: x[2])

        for job, op_idx, _ in early_consume_ops[:2]:
            pos = self.find_os_position_of_operation(ind.OS, job, op_idx)
            if pos is None:
                continue

            new_pos = min(len(ind.OS) - 1, pos + 4)
            nei = self.make_os_insert_neighbor(ind, job, op_idx, new_pos, origin_task="local")
            if nei is not None:
                neighbors.append(nei)

            if len(neighbors) >= max_neighbors:
                return neighbors

        return neighbors

    def find_os_positions_of_job(self, os_seq: List[str], job: str) -> List[int]:
        return [i for i, g in enumerate(os_seq) if g == job]

    def neighbor_ms_reassign(self, ind: Individual, target_job: str, target_op: int) -> List[Individual]:
        neighbors = []
        target_idx = self.get_ms_index(target_job, target_op)
        if target_idx is None:
            return neighbors

        current_machine = ind.MS[target_idx]
        legal_machines = list(self.operations[target_job][target_op]["machines"].keys())

        for m in legal_machines:
            if m == current_machine:
                continue

            nei = ind.copy()
            nei.MS[target_idx] = m
            nei.origin_task = "local"
            self.reset_individual_evaluation(nei)
            neighbors.append(nei)

        return neighbors

    def generate_local_offspring(self) -> List[Individual]:
        if not self.main_population:
            return [self.initialize_individual(origin_task="local") for _ in range(self.local_pop_size)]

        seed_pool = []
        seed_pool.extend(self.sort_population(self.main_population)[:min(self.local_elite_count, len(self.main_population))])

        if self.local_population:
            valid_local = [ind for ind in self.local_population if ind.makespan is not None and ind.shortage is not None]
            valid_local.sort(key=lambda ind: (ind.shortage, ind.makespan))
            seed_pool.extend(valid_local[:min(self.local_elite_count, len(valid_local))])

        if not seed_pool:
            return [self.initialize_individual(origin_task="local") for _ in range(self.local_pop_size)]

        # 去重
        unique_seed = {}
        for ind in seed_pool:
            if ind.makespan is None or ind.shortage is None:
                continue
            key = (float(ind.makespan), float(ind.shortage))
            if key not in unique_seed:
                unique_seed[key] = ind
        elites = list(unique_seed.values()) or seed_pool

        local_candidates: List[Individual] = []

        for ind in elites:
            clone = ind.copy()
            clone.origin_task = "local"
            local_candidates.append(clone)

            critical_buffers = self.get_critical_shortage_buffers_from_main(ind, top_k=2)

            if critical_buffers:
                batches: List[List[Individual]] = []
                for bid in critical_buffers:
                    batches.append(
                        self.neighbor_ms_reassign_supply_to_buffer(ind, bid, max_neighbors_per_op=1)
                    )
                    batches.append(
                        self.neighbor_os_forward_insert_supply_to_buffer(
                            ind,
                            bid,
                            max_neighbors=self.local_neighbors_per_elite,
                            max_shift=6
                        )
                    )
                    batches.append(
                        self.neighbor_os_backward_insert_consume_from_buffer(
                            ind,
                            bid,
                            max_neighbors=self.local_neighbors_per_elite,
                            max_shift=6
                        )
                    )
                    batches.append(
                        self.neighbor_shortage_window_alignment(
                            ind,
                            bid,
                            max_neighbors=self.local_neighbors_per_elite
                        )
                    )
                local_candidates.extend(self.take_neighbors_by_budget(
                    batches=batches,
                    budget=self.local_neighbors_per_elite
                ))
            else:
                for _ in range(self.local_neighbors_per_elite):
                    nei = self.mutate(
                        ind,
                        os_mut_rate=self.local_os_mutation_rate,
                        ms_mut_rate=self.local_ms_mutation_rate
                    )
                    nei.origin_task = "local"
                    local_candidates.append(nei)

        while len(local_candidates) < self.local_pop_size:
            base = self.rng.choice(elites).copy()
            nei = self.mutate(
                base,
                os_mut_rate=self.local_os_mutation_rate,
                ms_mut_rate=self.local_ms_mutation_rate
            )
            nei.origin_task = "local"
            local_candidates.append(nei)

        self.rng.shuffle(local_candidates)
        return local_candidates[:self.local_pop_size]

    # =========================================================
    # Offspring 生成
    # =========================================================

    def generate_main_offspring(self) -> List[Individual]:
        offspring = []

        remaining = self.remaining_budget()
        if remaining is not None and remaining <= 0:
            return offspring

        target_size = self.pop_size if remaining is None else min(self.pop_size, remaining)

        while len(offspring) < target_size:
            parent1 = self.tournament_select_main(self.main_population)
            parent2 = self.tournament_select_main(self.main_population)

            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            child1.origin_task = "main"
            child2.origin_task = "main"

            offspring.append(child1)
            if len(offspring) < target_size:
                offspring.append(child2)

        return offspring

    # 兼容旧接口：此处返回 CPAT offspring
    def generate_global_offspring(self) -> List[Individual]:
        return self.generate_critical_offspring()

    # =========================================================
    # 环境选择
    # =========================================================

    def refill_population_from_candidates(
        self,
        selected: List[Individual],
        candidates: List[Individual],
        target_size: int,
        sort_key
    ) -> List[Individual]:
        if target_size <= 0:
            return []
        if len(selected) >= target_size:
            return selected[:target_size]

        pool = [ind for ind in candidates if ind.makespan is not None and ind.shortage is not None]
        if not pool:
            return selected

        pool = sorted(pool, key=sort_key)
        idx = 0
        while len(selected) < target_size:
            selected.append(pool[idx % len(pool)].copy())
            idx += 1

        return selected

    def deduplicate_main_candidates_by_objectives(
        self,
        candidates: List[Individual]
    ) -> List[Individual]:
        unique = {}

        for ind in candidates:
            if ind.makespan is None or ind.shortage is None:
                continue

            key = (float(ind.makespan), float(ind.shortage))

            if key not in unique:
                unique[key] = ind
                continue

            old = unique[key]

            old_rank = old.rank if old.rank is not None else float("inf")
            new_rank = ind.rank if ind.rank is not None else float("inf")
            old_crowd = old.crowding_distance
            new_crowd = ind.crowding_distance

            old_key = (old_rank, -old_crowd, old.makespan, old.shortage)
            new_key = (new_rank, -new_crowd, ind.makespan, ind.shortage)

            if new_key < old_key:
                unique[key] = ind

        return [ind.copy() for ind in unique.values()]

    def environmental_select_main(self, candidates: List[Individual], target_size: int) -> List[Individual]:
        candidates = [ind for ind in candidates if ind.makespan is not None and ind.shortage is not None]
        if not candidates:
            return []

        original_candidates = candidates[:]
        self.assign_rank_and_crowding(candidates)
        candidates = self.deduplicate_main_candidates_by_objectives(candidates)
        fronts = self.assign_rank_and_crowding(candidates)

        new_population: List[Individual] = []

        for front in fronts:
            if len(new_population) + len(front) <= target_size:
                front_sorted = sorted(
                    front,
                    key=lambda ind: (ind.rank, -ind.crowding_distance, ind.makespan, ind.shortage)
                )
                new_population.extend(ind.copy() for ind in front_sorted)
            else:
                remaining = target_size - len(new_population)
                front_sorted = sorted(
                    front,
                    key=lambda ind: (-ind.crowding_distance, ind.makespan, ind.shortage)
                )
                new_population.extend(ind.copy() for ind in front_sorted[:remaining])
                break

        return self.refill_population_from_candidates(
            selected=new_population,
            candidates=original_candidates,
            target_size=target_size,
            sort_key=lambda ind: (
                ind.rank if ind.rank is not None else float("inf"),
                -ind.crowding_distance,
                ind.makespan,
                ind.shortage
            )
        )

    def environmental_select_critical(self, candidates: List[Individual], target_size: int) -> List[Individual]:
        candidates = [ind for ind in candidates if ind.makespan is not None and ind.shortage is not None]
        if not candidates:
            return []

        # 按 makespan, shortage 去重，保留排序靠前者
        unique = {}
        sorted_candidates = sorted(candidates, key=lambda ind: (ind.makespan, ind.shortage))
        for ind in sorted_candidates:
            key = (float(ind.makespan), float(ind.shortage))
            if key not in unique:
                unique[key] = ind

        selected = sorted(unique.values(), key=lambda ind: (ind.makespan, ind.shortage))[:target_size]
        return self.refill_population_from_candidates(
            selected=[ind.copy() for ind in selected],
            candidates=candidates,
            target_size=target_size,
            sort_key=lambda ind: (ind.makespan, ind.shortage)
        )

    def environmental_select_local(self, candidates: List[Individual], target_size: int) -> List[Individual]:
        candidates = [ind for ind in candidates if ind.makespan is not None and ind.shortage is not None]
        if not candidates:
            return []

        unique = {}
        sorted_candidates = sorted(candidates, key=lambda ind: (ind.shortage, ind.makespan))
        for ind in sorted_candidates:
            key = (float(ind.makespan), float(ind.shortage))
            if key not in unique:
                unique[key] = ind

        selected = sorted(unique.values(), key=lambda ind: (ind.shortage, ind.makespan))[:target_size]
        return self.refill_population_from_candidates(
            selected=[ind.copy() for ind in selected],
            candidates=candidates,
            target_size=target_size,
            sort_key=lambda ind: (ind.shortage, ind.makespan)
        )

    # 兼容旧接口
    def environmental_select_global(self, candidates: List[Individual], target_size: int) -> List[Individual]:
        return self.environmental_select_critical(candidates, target_size)

    def update_global_activity(self) -> None:
        self.global_active = False

    def get_current_gat_improve_ratio(self) -> Optional[float]:
        return None

    # =========================================================
    # 单代更新
    # =========================================================

    def run_one_generation(self, store_stats: bool = False) -> None:
        if not self.has_budget():
            return

        # 1) 三类 offspring 独立生成
        main_offspring = self.generate_main_offspring()
        if main_offspring:
            self.evaluate_population_main(main_offspring, store_stats=store_stats)
            main_offspring = [ind for ind in main_offspring if ind.makespan is not None and ind.shortage is not None]

        critical_offspring = []
        if self.has_budget():
            critical_offspring = self.generate_critical_offspring()
            if critical_offspring:
                self.evaluate_population_critical(critical_offspring, store_stats=store_stats)
                critical_offspring = [ind for ind in critical_offspring if ind.makespan is not None and ind.shortage is not None]

        local_offspring = []
        if self.has_budget():
            local_offspring = self.generate_local_offspring()
            if local_offspring:
                self.evaluate_population_main(local_offspring, store_stats=store_stats)
                local_offspring = [ind for ind in local_offspring if ind.makespan is not None and ind.shortage is not None]

        # 2) MT 环境选择：融合三类搜索结果
        main_candidates = (
            [ind.copy() for ind in self.main_population] +
            [ind.copy() for ind in main_offspring] +
            [ind.copy() for ind in critical_offspring] +
            [ind.copy() for ind in local_offspring]
        )
        main_candidates = [ind for ind in main_candidates if ind.makespan is not None and ind.shortage is not None]

        if main_candidates:
            self.main_population = self.environmental_select_main(main_candidates, self.pop_size)
            self.assign_rank_and_crowding(self.main_population)
            self.population = self.main_population
            self._update_best(self.main_population)

        # 3) CPAT 环境选择：按 makespan, shortage
        critical_candidates = (
            [ind.copy() for ind in self.critical_population if ind.makespan is not None and ind.shortage is not None] +
            [ind.copy() for ind in critical_offspring] +
            [ind.copy() for ind in main_offspring]
        )
        critical_candidates = [ind for ind in critical_candidates if ind.makespan is not None and ind.shortage is not None]

        if critical_candidates:
            self.critical_population = self.environmental_select_critical(critical_candidates, self.critical_pop_size)
            self.global_population = self.critical_population

        # 4) LAT 环境选择：按 shortage, makespan
        local_candidates = (
            [ind.copy() for ind in self.local_population if ind.makespan is not None and ind.shortage is not None] +
            [ind.copy() for ind in local_offspring] +
            [ind.copy() for ind in main_offspring]
        )
        local_candidates = [ind for ind in local_candidates if ind.makespan is not None and ind.shortage is not None]

        if local_candidates:
            self.local_population = self.environmental_select_local(local_candidates, self.local_pop_size)

    # =========================================================
    # 主循环
    # =========================================================

    def run(
        self,
        store_stats_init: bool = True,
        store_stats_generations: bool = False,
        verbose: bool = True
    ) -> Individual:
        # store_stats_* 保留给统一实验 runner 兼容；CPAT/LAT 邻域依赖 schedule/stats，
        # 因此本算法始终保存完整评价信息。
        self.n_evaluations = 0

        self.main_population = []
        self.critical_population = []
        self.local_population = []
        self.global_population = self.critical_population
        self.population = self.main_population
        self.best_individual = None

        self.history_best_fitness = []
        self.history_best_shortage = []
        self.history_eval_counts = []
        self.history_fronts = []
        self._last_snapshot_eval = -1

        self.global_active = False
        self.global_best_history = []

        self.initialize_populations()

        # 初始化 MT
        self.evaluate_population_main(self.main_population, store_stats=store_stats_init)
        self.main_population = [ind for ind in self.main_population if ind.makespan is not None and ind.shortage is not None]
        if not self.main_population:
            raise RuntimeError("初始化阶段预算不足，主种群未能完成任何个体评价")

        # 初始化 CPAT：真实 WIP 评价
        if self.has_budget() and self.critical_population:
            self.evaluate_population_critical(self.critical_population, store_stats=store_stats_init)
            self.critical_population = [
                ind for ind in self.critical_population
                if ind.makespan is not None and ind.shortage is not None
            ]
            self.critical_population = self.environmental_select_critical(self.critical_population, self.critical_pop_size)
            self.global_population = self.critical_population
        else:
            self.critical_population = []
            self.global_population = self.critical_population

        # 初始化 LAT：直接评价 initialize_populations() 中的启发式混合初始化个体。
        if self.has_budget() and self.local_population:
            self.evaluate_population_main(self.local_population, store_stats=store_stats_init)
            self.local_population = [
                ind for ind in self.local_population
                if ind.makespan is not None and ind.shortage is not None
            ]
            self.local_population = self.environmental_select_local(self.local_population, self.local_pop_size)
        else:
            self.local_population = []

        self.population = self.main_population
        self.assign_rank_and_crowding(self.main_population)
        self._update_best(self.main_population)

        init_best = self.get_best_individual(self.main_population)
        self.history_best_fitness = [float(init_best.makespan)]
        self.history_best_shortage = [float(init_best.shortage)]
        self.history_eval_counts = [int(self.n_evaluations)]

        self.record_front_snapshot(force=True)

        if verbose:
            print(
                f"[Init] rep_makespan = {init_best.makespan}, "
                f"rep_shortage = {init_best.shortage}, "
                f"rank = {init_best.rank}, "
                f"pareto_size = {len(self.get_pareto_front(self.main_population))}, "
                f"critical_pop = {len(self.critical_population)}, "
                f"local_pop = {len(self.local_population)}, "
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

            best = self.get_best_individual(self.main_population)
            self.history_best_fitness.append(float(best.makespan))
            self.history_best_shortage.append(float(best.shortage))
            self.history_eval_counts.append(int(self.n_evaluations))

            self.record_front_snapshot(force=False)

            if verbose:
                pareto_size = len(self.get_pareto_front(self.main_population))
                critical_best = None
                local_best = None

                if self.critical_population:
                    valid_critical = [ind for ind in self.critical_population if ind.makespan is not None]
                    if valid_critical:
                        critical_best = min(ind.makespan for ind in valid_critical)

                if self.local_population:
                    valid_local = [ind for ind in self.local_population if ind.shortage is not None]
                    if valid_local:
                        local_best = min(ind.shortage for ind in valid_local)

                print(
                    f"[Gen {gen}] rep_makespan = {best.makespan}, "
                    f"rep_shortage = {best.shortage}, "
                    f"rank = {best.rank}, "
                    f"pareto_size = {pareto_size}, "
                    f"critical_best_makespan = {critical_best}, "
                    f"local_best_shortage = {local_best}, "
                    f"evals = {self.n_evaluations}"
                )

        self.record_front_snapshot(force=True)

        if self.best_individual is None:
            raise RuntimeError("算法结束时 best_individual is None")

        return self.best_individual.copy()

    # =========================================================
    # 调试 / 展示
    # =========================================================

    def print_population_summary(self, population: Optional[List[Individual]] = None, top_k: int = 5) -> None:
        if population is None:
            population = self.main_population

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
                f"origin={ind.origin_task}, "
                f"OS_len={len(ind.OS)}, MS_len={len(ind.MS)}"
            )

    def print_best_summary(self) -> None:
        if self.best_individual is None:
            print("best individual is None")
            return

        ind = self.best_individual
        print("===== Best Main-Task Individual =====")
        print(f"rank     : {ind.rank}")
        print(f"crowd    : {ind.crowding_distance}")
        print(f"makespan : {ind.makespan}")
        print(f"shortage : {ind.shortage}")
        print(f"origin   : {ind.origin_task}")
        print(f"OS       : {ind.OS}")
        print(f"MS       : {ind.MS}")

        if ind.stats is not None:
            total_blocking = ind.stats["blocking"]["total_blocking_time"]
            print(f"blocking : {total_blocking}")
