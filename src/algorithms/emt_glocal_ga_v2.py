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
from src.solution.decoder_no_wip import NoWIPScheduler

@dataclass
class Individual:
    """
    个体表示：
    - OS: job-based operation sequence
    - MS: machine selection list（与 encoder.ms_index_order 对齐）
    """
    OS: List[str]
    MS: List[str]

    # ---------- 主任务（真实WIP）上的评价 ----------
    fitness: Optional[float] = None
    makespan: Optional[int] = None
    shortage: Optional[float] = None

    rank: Optional[int] = None
    crowding_distance: float = 0.0

    schedule: Optional[List[Dict[str, Any]]] = None
    buffer_trace: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None

    # ---------- 全局辅助任务（放松WIP）上的评价 ----------
    global_fitness: Optional[float] = None
    global_makespan: Optional[int] = None

    # ---------- 调试字段 ----------
    origin_task: Optional[str] = None  # "main" / "global" / "local" / None

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
    EMT-based 三任务调度算法（主任务 + 全局辅助任务 + 局部辅助任务）

    任务定义：
    1) Main Task (MT)
       - 真实 buffer / WIP 约束
       - 双目标：min makespan, min shortage

    2) Global Auxiliary Task (GAT)
        - 使用“完全忽略 WIP / buffer 约束”的解码器
        - 单目标：min makespan
        - 用于前期进行更激进的全局结构探索
        - 当 GAT 最优改进率过低时，关闭 GAT，进入后期阶段

    3) Local Auxiliary Task (LAT)
       - 围绕主种群精英构造邻域个体
       - 使用真实任务评价
       - 强化主种群周围的局部多样性与精细挖掘

    说明：
    - 这是一版“可落地、易调试”的 EMT 风格实现
    - 接口尽量与现有实验框架兼容
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
        global_pop_size: Optional[int] = None,
        local_pop_size: Optional[int] = None,
        
        # ----- GAT 阶段切换参数 -----
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
        if max_evaluations is not None and max_evaluations <= 0:
            raise ValueError("max_evaluations 必须 > 0")
        if snapshot_interval is not None and snapshot_interval <= 0:
            raise ValueError("snapshot_interval 必须 > 0")

        self.operations = operations
        self.buffers = buffers

        self.pop_size = pop_size
        self.global_pop_size = global_pop_size if global_pop_size is not None else pop_size
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

        self.gat_improve_window = gat_improve_window
        self.gat_improve_threshold = gat_improve_threshold

        self.local_elite_count = local_elite_count
        self.local_neighbors_per_elite = local_neighbors_per_elite
        self.local_os_mutation_rate = local_os_mutation_rate
        self.local_ms_mutation_rate = local_ms_mutation_rate

        self.rng = random.Random(seed)

        self.encoder = Encoder(self.operations, rng=self.rng)
        self.main_scheduler = StageBufferWIPScheduler(self.operations, self.buffers)
        self.global_scheduler = NoWIPScheduler(self.operations)

        # 三个种群
        self.main_population: List[Individual] = []
        self.global_population: List[Individual] = []
        self.local_population: List[Individual] = []

        self.population: List[Individual] = self.main_population  # 为兼容旧代码/展示接口
        self.best_individual: Optional[Individual] = None

        self.history_best_fitness: List[float] = []
        self.history_best_shortage: List[float] = []

        self.history_eval_counts: List[int] = []
        self.history_fronts: List[Dict[str, Any]] = []
        self._last_snapshot_eval: int = -1

        # GAT 阶段切换相关
        self.global_active: bool = True
        self.global_best_history: List[float] = []

    # =========================================================
    # 初始化相关
    # =========================================================
    def initialize_individual(self, origin_task: Optional[str] = None) -> Individual:
        os_seq = self.encoder.generate_random_os()
        ms_list = self.encoder.generate_random_ms()
        return Individual(OS=os_seq, MS=ms_list, origin_task=origin_task)

    def initialize_populations(self) -> None:
        self.main_population = [self.initialize_individual(origin_task="main") for _ in range(self.pop_size)]
        self.global_population = [self.initialize_individual(origin_task="global") for _ in range(self.global_pop_size)]
        self.local_population = [self.initialize_individual(origin_task="local") for _ in range(self.local_pop_size)]
        self.population = self.main_population

    def has_budget(self) -> bool:
        return self.max_evaluations is None or self.n_evaluations < self.max_evaluations

    def remaining_budget(self) -> Optional[int]:
        if self.max_evaluations is None:
            return None
        return max(0, self.max_evaluations - self.n_evaluations)

    # =========================================================
    # 评价相关：主任务 / 全局辅助任务
    # =========================================================

    def evaluate_individual(
        self,
        ind: Individual,
        store_stats: bool = True,
        task: str = "main"
    ) -> float:
        """
        默认对主任务评价，以兼容现有实验脚本。
        task:
        - "main": 真实约束任务
        - "global": 完全忽略 WIP / buffer 的全局辅助任务
        """
        if not self.has_budget():
            raise RuntimeError("Evaluation budget exhausted")

        ms_map = self.encoder.build_ms_map(ind.MS)

        if task == "main":
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

            self.n_evaluations += 1
            return ind.fitness

        elif task == "global":
            makespan, _, _ = self.global_scheduler.decode(
                os_seq=ind.OS,
                ms_map=ms_map
            )
            ind.global_makespan = makespan
            ind.global_fitness = float(makespan)

            self.n_evaluations += 1
            return ind.global_fitness

        else:
            raise ValueError(f"未知任务类型: {task}")

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
            self.evaluate_individual(ind, store_stats=store_stats, task="main")

        evaluated = [ind for ind in population if ind.makespan is not None and ind.shortage is not None]
        if evaluated:
            self.assign_rank_and_crowding(evaluated)
            self._update_best(evaluated)

    def evaluate_population_global(self, population: List[Individual]) -> None:
        if not population:
            raise ValueError("population 为空，无法评价全局辅助任务")

        for ind in population:
            if not self.has_budget():
                break
            self.evaluate_individual(ind, store_stats=False, task="global")

    def ensure_evaluated_on_main(
        self,
        population: List[Individual],
        store_stats: bool = True
    ) -> List[Individual]:
        """
        确保 population 中所有个体都具有主任务评价（makespan, shortage）。
        对尚未具有主任务评价的个体进行补评。
        返回已成功具有主任务评价的个体列表。
        """
        result = []

        for ind in population:
            if ind.makespan is None or ind.shortage is None:
                if not self.has_budget():
                    break
                self.evaluate_individual(ind, store_stats=store_stats, task="main")

            if ind.makespan is not None and ind.shortage is not None:
                result.append(ind)

        return result
    
    def ensure_evaluated_on_global(
        self,
        population: List[Individual]
    ) -> List[Individual]:
        """
        确保 population 中所有个体都具有全局辅助任务评价（global_makespan）。
        对尚未具有 GAT 评价的个体进行补评。
        返回已成功具有 GAT 评价的个体列表。
        """
        result = []

        for ind in population:
            if ind.global_makespan is None:
                if not self.has_budget():
                    break
                self.evaluate_individual(ind, store_stats=False, task="global")

            if ind.global_makespan is not None:
                result.append(ind)

        return result

    # =========================================================
    # 主任务：多目标工具函数
    # =========================================================

    def get_objectives(self, ind: Individual) -> Tuple[float, float]:
        if ind.makespan is None or ind.shortage is None:
            raise ValueError("个体尚未完成主任务双目标评价")
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

    def sort_population(self, population: Optional[List[Individual]] = None) -> List[Individual]:
        if population is None:
            population = self.main_population

        if any(ind.makespan is None or ind.shortage is None for ind in population):
            raise ValueError("存在未评价个体，不能排序")

        self.assign_rank_and_crowding(population)

        return sorted(
            population,
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

        valid_population = [
            ind for ind in population
            if ind.makespan is not None and ind.shortage is not None
        ]

        if not valid_population:
            return []

        fronts = self.assign_rank_and_crowding(valid_population)
        return [ind.copy() for ind in fronts[0]] if fronts else []

    def _update_best(self, population: Optional[List[Individual]] = None) -> None:
        if population is None:
            population = self.main_population

        current_best = self.get_best_individual(population)

        if self.best_individual is None:
            self.best_individual = current_best.copy()
            return

        if self.better(current_best, self.best_individual):
            self.best_individual = current_best.copy()

    def export_pareto_front_snapshot(
        self,
        population: Optional[List[Individual]] = None
    ) -> List[Dict[str, Any]]:
        """
        导出当前主任务 Pareto front 的精简快照，用于后续离线计算 GD/IGD 收敛曲线。
        这里不保存完整 schedule / stats，只保存目标值和编码，避免 JSON 过大。
        """
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

    def record_front_snapshot(
        self,
        force: bool = False
    ) -> None:
        """
        按评价次数记录当前主种群的 Pareto front 快照。
        - 若 snapshot_interval is None，则不记录
        - force=True 时强制记录（通常用于初始化结束 / 算法结束）
        """
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

        # 避免同一 eval_count 重复记录
        if self.history_fronts and self.history_fronts[-1]["eval_count"] == current_eval:
            return

        snapshot = {
            "eval_count": current_eval,
            "pareto_front": self.export_pareto_front_snapshot(self.main_population)
        }

        self.history_fronts.append(snapshot)
        self._last_snapshot_eval = current_eval

    # =========================================================
    # 选择
    # =========================================================

    def tournament_select_main(self, population: Optional[List[Individual]] = None) -> Individual:
        if population is None:
            population = self.main_population

        if len(population) < self.tournament_size:
            raise ValueError("population 数量小于 tournament_size")

        candidates = self.rng.sample(population, self.tournament_size)
        best = candidates[0]
        for cand in candidates[1:]:
            if self.better(cand, best):
                best = cand
        return best.copy()

    def tournament_select_global(self, population: Optional[List[Individual]] = None) -> Individual:
        if population is None:
            population = self.global_population

        if len(population) < self.tournament_size:
            raise ValueError("global population 数量小于 tournament_size")

        candidates = self.rng.sample(population, self.tournament_size)
        winner = min(candidates, key=lambda ind: ind.global_fitness)
        return winner.copy()

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
    # LAT：邻域生成
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
    
    def get_critical_shortage_buffers_from_main(
        self,
        ind: Individual,
        top_k: int = 2
    ) -> List[str]:
        """
        从主任务统计信息中找 shortage 最严重的 buffer
        """
        if ind.stats is None:
            return []

        shortage_dict = ind.stats["shortage"]["per_buffer_shortage_area"]
        if not shortage_dict:
            return []

        items = sorted(shortage_dict.items(), key=lambda x: x[1], reverse=True)
        items = [x for x in items if x[1] > 0]

        return [bid for bid, _ in items[:top_k]]
    
    def get_upstream_supply_ops_of_buffer(self, buffer_id: str) -> List[Tuple[str, int]]:
        """
        返回所有向该 buffer 放料的工序 (job, op_idx)
        即 buffer_out == buffer_id
        """
        result = []
        for job, ops in self.operations.items():
            for op_idx, op in enumerate(ops):
                if op.get("buffer_out", None) == buffer_id:
                    result.append((job, op_idx))
        return result


    def get_downstream_consume_ops_of_buffer(self, buffer_id: str) -> List[Tuple[str, int]]:
        """
        返回所有从该 buffer 取料的工序 (job, op_idx)
        即 buffer_in == buffer_id
        """
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
        """
        从 buffer_trace 中提取某个 buffer 的 shortage 区间
        返回 [(start, end, gap_level), ...]
        其中 gap_level = low_wip - level (>0)
        """
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

        # 最后一段延续到 makespan
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
        """
        返回 shortage 面积最大的区间 (start, end, gap_level)
        面积定义为 (end-start) * gap_level
        """
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
        """
        对向 buffer_id 供料的上游工序，尝试换到更快机器
        """
        neighbors = []
        supply_ops = self.get_upstream_supply_ops_of_buffer(buffer_id)

        for job, op_idx in supply_ops:
            target_idx = None
            for idx, (j, o) in enumerate(self.encoder.ms_index_order):
                if j == job and o == op_idx:
                    target_idx = idx
                    break

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
        """
        将向 buffer_id 供料的工序对应 job 在 OS 中前移，促进更早补货
        """
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

                nei = ind.copy()
                gene = nei.OS.pop(pos)
                nei.OS.insert(new_pos, gene)
                nei.origin_task = "local"
                self.reset_individual_evaluation(nei)
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
        """
        将从 buffer_id 取料的下游工序对应 job 在 OS 中后移，减缓抽空速度
        """
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

                nei = ind.copy()
                gene = nei.OS.pop(pos)
                nei.OS.insert(new_pos, gene)
                nei.origin_task = "local"
                self.reset_individual_evaluation(nei)
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
        """
        围绕最严重 shortage 时间窗进行对齐：
        - 将晚到的供料工序前移
        - 将早取的下游工序后移
        """
        if ind.schedule is None:
            return []

        critical_interval = self.get_most_critical_shortage_interval(ind, buffer_id)
        if critical_interval is None:
            return []

        t_start, t_end, _ = critical_interval
        neighbors = []

        # 1) 找晚到的供料工序：end > t_start
        late_supply_ops = []
        for rec in ind.schedule:
            job = rec["job"]
            op_idx = rec["op"]
            op_def = self.operations[job][op_idx]
            if op_def.get("buffer_out", None) == buffer_id and rec["end"] > t_start:
                late_supply_ops.append((job, op_idx, rec["end"]))

        late_supply_ops.sort(key=lambda x: x[2])  # 优先移动稍晚到的

        for job, op_idx, _ in late_supply_ops[:2]:
            pos = self.find_os_position_of_operation(ind.OS, job, op_idx)
            if pos is None:
                continue

            new_pos = max(0, pos - 4)
            nei = ind.copy()
            gene = nei.OS.pop(pos)
            nei.OS.insert(new_pos, gene)
            nei.origin_task = "local"
            self.reset_individual_evaluation(nei)
            neighbors.append(nei)

            if len(neighbors) >= max_neighbors:
                return neighbors

        # 2) 找 shortage 窗口内过早抽料的下游工序：start < t_end 且 buffer_in == buffer_id
        early_consume_ops = []
        for rec in ind.schedule:
            job = rec["job"]
            op_idx = rec["op"]
            op_def = self.operations[job][op_idx]
            if op_def.get("buffer_in", None) == buffer_id and rec["start"] < t_end:
                early_consume_ops.append((job, op_idx, rec["start"]))

        early_consume_ops.sort(key=lambda x: x[2])  # 优先处理最早抽料的

        for job, op_idx, _ in early_consume_ops[:2]:
            pos = self.find_os_position_of_operation(ind.OS, job, op_idx)
            if pos is None:
                continue

            new_pos = min(len(ind.OS) - 1, pos + 4)
            nei = ind.copy()
            gene = nei.OS.pop(pos)
            nei.OS.insert(new_pos, gene)
            nei.origin_task = "local"
            self.reset_individual_evaluation(nei)
            neighbors.append(nei)

            if len(neighbors) >= max_neighbors:
                return neighbors

        return neighbors

    
    def find_os_positions_of_job(self, os_seq: List[str], job: str) -> List[int]:
        return [i for i, g in enumerate(os_seq) if g == job]

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
            nei.origin_task = "local"
            self.reset_individual_evaluation(nei)
            neighbors.append(nei)

        return neighbors

    def generate_local_offspring(self) -> List[Individual]:
        """
        LAT 生成逻辑（shortage-aware）：
        - 以主种群精英为中心
        - 先识别 shortage 最严重的 buffer
        - 再围绕该 buffer 构造三类邻域：
            1) 上游供料加速
            2) 下游取料延缓
            3) shortage 时间窗对齐
        """
        if not self.main_population:
            return [self.initialize_individual(origin_task="local") for _ in range(self.local_pop_size)]

        elites = self.sort_population(self.main_population)[:min(self.local_elite_count, len(self.main_population))]
        local_candidates: List[Individual] = []

        # 1) 精英自身作为局部任务种子
        for ind in elites:
            clone = ind.copy()
            clone.origin_task = "local"
            local_candidates.append(clone)

        # 2) 围绕关键 shortage buffer 构造邻域
        for ind in elites:
            critical_buffers = self.get_critical_shortage_buffers_from_main(ind, top_k=2)

            if critical_buffers:
                for bid in critical_buffers:
                    # 邻域1：上游供料加速（机器重分配）
                    local_candidates.extend(
                        self.neighbor_ms_reassign_supply_to_buffer(
                            ind, bid, max_neighbors_per_op=1
                        )
                    )

                    # 邻域1扩展：上游供料前移
                    local_candidates.extend(
                        self.neighbor_os_forward_insert_supply_to_buffer(
                            ind, bid, max_neighbors=2, max_shift=6
                        )
                    )

                    # 邻域2：下游取料延缓
                    local_candidates.extend(
                        self.neighbor_os_backward_insert_consume_from_buffer(
                            ind, bid, max_neighbors=2, max_shift=6
                        )
                    )

                    # 邻域3：shortage 时间窗对齐
                    local_candidates.extend(
                        self.neighbor_shortage_window_alignment(
                            ind, bid, max_neighbors=2
                        )
                    )
            else:
                # 如果没有 shortage，就退化成轻微局部扰动
                for _ in range(3):
                    nei = self.mutate(
                        ind,
                        os_mut_rate=self.local_os_mutation_rate,
                        ms_mut_rate=self.local_ms_mutation_rate
                    )
                    nei.origin_task = "local"
                    local_candidates.append(nei)

        # 3) 不足则补齐
        while len(local_candidates) < self.local_pop_size:
            base = self.rng.choice(elites).copy()
            nei = self.mutate(
                base,
                os_mut_rate=self.local_os_mutation_rate,
                ms_mut_rate=self.local_ms_mutation_rate
            )
            nei.origin_task = "local"
            local_candidates.append(nei)

        # 4) 若过多，随机打散后截断
        self.rng.shuffle(local_candidates)
        local_candidates = local_candidates[:self.local_pop_size]

        return local_candidates

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

    def generate_global_offspring(self) -> List[Individual]:
        if not self.global_active:
            return []

        offspring = []

        remaining = self.remaining_budget()
        if remaining is not None and remaining <= 0:
            return offspring

        target_size = self.global_pop_size if remaining is None else min(self.global_pop_size, remaining)

        while len(offspring) < target_size:
            parent1 = self.tournament_select_global(self.global_population)
            parent2 = self.tournament_select_global(self.global_population)

            child1, child2 = self.crossover(parent1, parent2)

            # GAT 偏全局：OS/MS 都保留基础扰动
            child1 = self.mutate(child1, os_mut_rate=self.os_mutation_rate, ms_mut_rate=self.ms_mutation_rate)
            child2 = self.mutate(child2, os_mut_rate=self.os_mutation_rate, ms_mut_rate=self.ms_mutation_rate)

            child1.origin_task = "global"
            child2.origin_task = "global"

            offspring.append(child1)
            if len(offspring) < target_size:
                offspring.append(child2)

        return offspring

    # =========================================================
    # 环境选择
    # =========================================================

    def environmental_select_main(self, candidates: List[Individual], target_size: int) -> List[Individual]:
        # 先做一次 rank / crowding 赋值，便于去重时挑代表个体
        self.assign_rank_and_crowding(candidates)

        # 按主任务目标点去重，避免大量重复 (makespan, shortage) 占满种群
        candidates = self.deduplicate_main_candidates_by_objectives(candidates)

        # 去重后重新计算 rank / crowding
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

        return new_population

    def environmental_select_global(self, candidates: List[Individual], target_size: int) -> List[Individual]:
        if any(ind.global_makespan is None for ind in candidates):
            raise ValueError("global candidates 存在未评价个体")

        candidates_sorted = sorted(candidates, key=lambda ind: ind.global_makespan)
        return [ind.copy() for ind in candidates_sorted[:target_size]]
    
    def deduplicate_main_candidates_by_objectives(
        self,
        candidates: List[Individual]
    ) -> List[Individual]:
        """
        按主任务目标 (makespan, shortage) 去重。
        同一目标点只保留一个个体。

        保留策略：
        - 优先保留 rank 更小的
        - 若 rank 相同，保留 crowding_distance 更大的
        - 若仍相同，保留 makespan 更小、shortage 更小的
        """
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

    # =========================================================
    # 全局辅助任务停更检测
    # =========================================================

    def update_global_activity(self) -> None:
        """
        根据 GAT 最优 makespan 的相对改进率决定是否关闭 GAT。
        """
        if not self.global_active:
            return

        if not self.global_population:
            return

        valid_global = [
            ind.global_makespan
            for ind in self.global_population
            if ind.global_makespan is not None
        ]
        if not valid_global:
            return

        current_best = float(min(valid_global))
        self.global_best_history.append(current_best)

        w = self.gat_improve_window
        if len(self.global_best_history) < w + 1:
            return

        old_best = self.global_best_history[-(w + 1)]
        new_best = self.global_best_history[-1]
        improve_ratio = (old_best - new_best) / max(1.0, old_best)

        if improve_ratio <= self.gat_improve_threshold:
            self.global_active = False
            self.global_population = []
            # GAT 已完成前期全局探索使命，后续不再进化

    def get_current_gat_improve_ratio(self) -> Optional[float]:
        """
        返回当前基于窗口的 GAT 最优改进率。
        若历史长度不足，则返回 None。
        """
        w = self.gat_improve_window
        if len(self.global_best_history) < w + 1:
            return None

        old_best = self.global_best_history[-(w + 1)]
        new_best = self.global_best_history[-1]
        return (old_best - new_best) / max(1.0, old_best)

    # =========================================================
    # 单代更新
    # =========================================================

    def run_one_generation(self, store_stats: bool = False, generation: int = 1) -> None:
        if not self.has_budget():
            return

        # =====================================================
        # 1) 三个任务独立生成 offspring
        # =====================================================
        main_offspring = self.generate_main_offspring()
        if main_offspring:
            self.evaluate_population_main(main_offspring, store_stats=store_stats)
            main_offspring = [
                ind for ind in main_offspring
                if ind.makespan is not None and ind.shortage is not None
            ]

        global_offspring = self.generate_global_offspring()
        if global_offspring:
            self.evaluate_population_global(global_offspring)
            global_offspring = [
                ind for ind in global_offspring
                if ind.global_makespan is not None
            ]

        local_offspring = []
        if self.has_budget():
            local_offspring = self.generate_local_offspring()
            if local_offspring:
                self.evaluate_population_main(local_offspring, store_stats=store_stats)
                local_offspring = [
                    ind for ind in local_offspring
                    if ind.makespan is not None and ind.shortage is not None
                ]

        # =====================================================
        # 2) MT 环境选择：P1 <- select(P1 + O1 + O2 + O3)
        #    其中 O2 需要先补主任务评价
        # =====================================================
        o2_for_main = []
        if global_offspring and self.has_budget():
            o2_for_main = self.ensure_evaluated_on_main(
                [ind.copy() for ind in global_offspring],
                store_stats=store_stats
            )

        main_candidates = (
            [ind.copy() for ind in self.main_population] +
            [ind.copy() for ind in main_offspring] +
            o2_for_main +
            [ind.copy() for ind in local_offspring]
        )

        main_candidates = [
            ind for ind in main_candidates
            if ind.makespan is not None and ind.shortage is not None
        ]

        if main_candidates:
            self.main_population = self.environmental_select_main(main_candidates, self.pop_size)
            self.assign_rank_and_crowding(self.main_population)
            self.population = self.main_population
            self._update_best(self.main_population)

    # =====================================================
    # 3) GAT 环境选择：P2 <- select(P2 + O2 + O1 + O3)
    #    其中 O1、O3 需要先补 GAT 评价
    # =====================================================
        if self.global_active:
            o1_for_global = []
            o3_for_global = []

            if main_offspring and self.has_budget():
                o1_for_global = self.ensure_evaluated_on_global(
                    [ind.copy() for ind in main_offspring]
                )

            if local_offspring and self.has_budget():
                o3_for_global = self.ensure_evaluated_on_global(
                    [ind.copy() for ind in local_offspring]
                )

            global_candidates = (
                [ind.copy() for ind in self.global_population if ind.global_makespan is not None] +
                [ind.copy() for ind in global_offspring] +
                o1_for_global +
                o3_for_global
            )

            global_candidates = [
                ind for ind in global_candidates
                if ind.global_makespan is not None
            ]

            if global_candidates:
                self.global_population = self.environmental_select_global(global_candidates, self.global_pop_size)
                self.update_global_activity()

        # =====================================================
        # 4) LAT 环境选择：P3 <- select(P3 + O3 + O1 + O2)
        #    其中 O2 需要先补主任务评价
        # =====================================================
        o2_for_local = []
        if global_offspring and self.has_budget():
            o2_for_local = self.ensure_evaluated_on_main(
                [ind.copy() for ind in global_offspring],
                store_stats=store_stats
            )

        local_candidates = (
            [ind.copy() for ind in self.local_population if ind.makespan is not None and ind.shortage is not None] +
            [ind.copy() for ind in local_offspring] +
            [ind.copy() for ind in main_offspring] +
            o2_for_local
        )

        local_candidates = [
            ind for ind in local_candidates
            if ind.makespan is not None and ind.shortage is not None
        ]

        if local_candidates:
            self.local_population = self.environmental_select_main(local_candidates, self.local_pop_size)
            self.assign_rank_and_crowding(self.local_population)

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

        self.main_population = []
        self.global_population = []
        self.local_population = []
        self.population = self.main_population

        self.best_individual = None

        self.history_best_fitness = []
        self.history_best_shortage = []
        self.history_eval_counts = []
        self.history_fronts = []
        self._last_snapshot_eval = -1

        self.global_active = True
        self.global_best_history = []

        self.initialize_populations()
        # ---------- 初始化评价 ----------
        self.evaluate_population_main(self.main_population, store_stats=store_stats_init)
        self.main_population = [ind for ind in self.main_population if ind.makespan is not None and ind.shortage is not None]
        if not self.main_population:
            raise RuntimeError("初始化阶段预算不足，主种群未能完成任何个体评价")

        if self.has_budget():
            self.evaluate_population_global(self.global_population)
            self.global_population = [ind for ind in self.global_population if ind.global_makespan is not None]

        if self.has_budget():
            self.local_population = self.generate_local_offspring()
            self.evaluate_population_main(self.local_population, store_stats=store_stats_init)
            self.local_population = [
                ind for ind in self.local_population
                if ind.makespan is not None and ind.shortage is not None
            ]
        else:
            self.local_population = []

        self.population = self.main_population
        self.assign_rank_and_crowding(self.main_population)
        if self.local_population:
            self.assign_rank_and_crowding(self.local_population)

        self.update_global_activity()

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
                f"global_active = {self.global_active}, "
                f"gat_improve = {self.get_current_gat_improve_ratio()}, "
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

            self.run_one_generation(store_stats=store_stats_generations, generation=gen)

            if self.n_evaluations == prev_evals:
                break

            best = self.get_best_individual(self.main_population)
            self.history_best_fitness.append(float(best.makespan))
            self.history_best_shortage.append(float(best.shortage))
            self.history_eval_counts.append(int(self.n_evaluations))

            self.record_front_snapshot(force=False)

            if verbose:
                pareto_size = len(self.get_pareto_front(self.main_population))
                global_best = None
                if self.global_active and self.global_population:
                    valid_global = [
                        ind.global_makespan
                        for ind in self.global_population
                        if ind.global_makespan is not None
                    ]
                    if valid_global:
                        global_best = min(valid_global)

                gat_improve = self.get_current_gat_improve_ratio()

                print(
                    f"[Gen {gen}] rep_makespan = {best.makespan}, "
                    f"rep_shortage = {best.shortage}, "
                    f"rank = {best.rank}, "
                    f"pareto_size = {pareto_size}, "
                    f"global_best = {global_best}, "
                    f"gat_improve = {gat_improve}, "
                    f"global_active = {self.global_active}, "
                    f"evals = {self.n_evaluations}"
                )
        self.record_front_snapshot(force=True)
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