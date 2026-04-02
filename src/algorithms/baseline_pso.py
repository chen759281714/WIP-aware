from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import random
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.solution.encoder import Encoder
from src.solution.decoder import StageBufferWIPScheduler


@dataclass
class Particle:
    """
    粒子表示：
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

    # 粒子历史最优
    pbest_OS: Optional[List[str]] = None
    pbest_MS: Optional[List[str]] = None
    pbest_fitness: Optional[float] = None
    pbest_makespan: Optional[int] = None

    def copy(self) -> "Particle":
        return Particle(
            OS=self.OS[:],
            MS=self.MS[:],
            fitness=self.fitness,
            makespan=self.makespan,
            schedule=[rec.copy() for rec in self.schedule] if self.schedule is not None else None,
            buffer_trace={k: v[:] for k, v in self.buffer_trace.items()} if self.buffer_trace is not None else None,
            stats=self.stats.copy() if self.stats is not None else None,
            pbest_OS=self.pbest_OS[:] if self.pbest_OS is not None else None,
            pbest_MS=self.pbest_MS[:] if self.pbest_MS is not None else None,
            pbest_fitness=self.pbest_fitness,
            pbest_makespan=self.pbest_makespan,
        )


class BaselinePSO:
    """
    基础离散 PSO（适配 OS/MS 双层编码）

    设计思路：
    - 不定义连续速度
    - 采用“向 pbest / gbest 学习”的概率更新方式
    - OS 通过目标导向 swap + 随机扰动更新
    - MS 通过逐位向 pbest / gbest 靠拢 + 小概率随机变异更新

    目标：
    - 先作为一个稳定、可运行、可对比的 baseline
    """

    def __init__(
        self,
        operations: Dict[str, List[Dict[str, Any]]],
        buffers: Dict[str, Dict[str, Any]],
        swarm_size: int = 100,
        n_iterations: int = 100,
        c1_os: float = 0.4,
        c2_os: float = 0.4,
        c1_ms: float = 0.4,
        c2_ms: float = 0.4,
        os_mutation_rate: float = 0.2,
        ms_mutation_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        if swarm_size <= 0:
            raise ValueError("swarm_size 必须 > 0")
        if n_iterations <= 0:
            raise ValueError("n_iterations 必须 > 0")
        for name, value in [
            ("c1_os", c1_os),
            ("c2_os", c2_os),
            ("c1_ms", c1_ms),
            ("c2_ms", c2_ms),
            ("os_mutation_rate", os_mutation_rate),
            ("ms_mutation_rate", ms_mutation_rate),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} 必须在 [0,1] 内")

        self.operations = operations
        self.buffers = buffers

        self.swarm_size = swarm_size
        self.n_iterations = n_iterations
        self.c1_os = c1_os
        self.c2_os = c2_os
        self.c1_ms = c1_ms
        self.c2_ms = c2_ms
        self.os_mutation_rate = os_mutation_rate
        self.ms_mutation_rate = ms_mutation_rate
        self.seed = seed

        self.rng = random.Random(seed)

        self.encoder = Encoder(self.operations, rng=self.rng)
        self.scheduler = StageBufferWIPScheduler(self.operations, self.buffers)

        self.swarm: List[Particle] = []
        self.best_particle: Optional[Particle] = None
        self.gbest_OS: Optional[List[str]] = None
        self.gbest_MS: Optional[List[str]] = None
        self.gbest_fitness: Optional[float] = None
        self.gbest_makespan: Optional[int] = None

        self.history_best_fitness: List[float] = []

    # =========================
    # 初始化
    # =========================

    def initialize_particle(self) -> Particle:
        os_seq = self.encoder.generate_random_os()
        ms_list = self.encoder.generate_random_ms()
        return Particle(OS=os_seq, MS=ms_list)

    def initialize_swarm(self) -> List[Particle]:
        self.swarm = [self.initialize_particle() for _ in range(self.swarm_size)]
        return self.swarm

    # =========================
    # 评价
    # =========================

    def evaluate_particle(self, particle: Particle, store_stats: bool = True) -> float:
        ms_map = self.encoder.build_ms_map(particle.MS)

        makespan, schedule, buffer_trace = self.scheduler.decode(
            os_seq=particle.OS,
            ms_map=ms_map
        )

        particle.makespan = makespan
        particle.fitness = float(makespan)
        particle.schedule = schedule
        particle.buffer_trace = buffer_trace

        if store_stats:
            particle.stats = self.scheduler.analyze(
                schedule=schedule,
                buffer_trace=buffer_trace,
                makespan=makespan
            )
        else:
            particle.stats = None

        return particle.fitness

    def evaluate_swarm(
        self,
        swarm: Optional[List[Particle]] = None,
        store_stats: bool = True
    ) -> None:
        if swarm is None:
            swarm = self.swarm

        if not swarm:
            raise ValueError("swarm 为空，无法评价")

        for p in swarm:
            self.evaluate_particle(p, store_stats=store_stats)
            self.update_pbest(p)

        self.update_gbest(swarm)

    # =========================
    # pbest / gbest 更新
    # =========================

    def update_pbest(self, particle: Particle) -> None:
        if particle.fitness is None:
            raise ValueError("particle 未评价，无法更新 pbest")

        if particle.pbest_fitness is None or particle.fitness < particle.pbest_fitness:
            particle.pbest_OS = particle.OS[:]
            particle.pbest_MS = particle.MS[:]
            particle.pbest_fitness = particle.fitness
            particle.pbest_makespan = particle.makespan

    def update_gbest(self, swarm: Optional[List[Particle]] = None) -> None:
        if swarm is None:
            swarm = self.swarm

        if any(p.fitness is None for p in swarm):
            raise ValueError("存在未评价粒子，无法更新 gbest")

        current_best = min(swarm, key=lambda p: p.fitness)

        if self.gbest_fitness is None or current_best.fitness < self.gbest_fitness:
            self.gbest_OS = current_best.OS[:]
            self.gbest_MS = current_best.MS[:]
            self.gbest_fitness = current_best.fitness
            self.gbest_makespan = current_best.makespan

            self.best_particle = current_best.copy()

    # =========================
    # 工具函数
    # =========================

    def reset_particle_evaluation(self, particle: Particle) -> None:
        particle.fitness = None
        particle.makespan = None
        particle.schedule = None
        particle.buffer_trace = None
        particle.stats = None

    def find_first_mismatch_index(self, source: List[str], target: List[str]) -> Optional[int]:
        for i, (a, b) in enumerate(zip(source, target)):
            if a != b:
                return i
        return None

    def find_swap_index_for_gene(self, seq: List[str], start_idx: int, gene: str) -> Optional[int]:
        for j in range(start_idx + 1, len(seq)):
            if seq[j] == gene:
                return j
        return None

    # =========================
    # OS 更新
    # =========================

    def learn_from_target_os_once(self, current_os: List[str], target_os: List[str]) -> List[str]:
        """
        朝 target_os 靠近一步：
        - 找到第一个不一致的位置 i
        - 在 current_os 后面找到 target_os[i] 对应的同 job 出现
        - 做一次 swap
        """
        new_os = current_os[:]

        idx = self.find_first_mismatch_index(new_os, target_os)
        if idx is None:
            return new_os

        target_gene = target_os[idx]
        swap_idx = self.find_swap_index_for_gene(new_os, idx, target_gene)

        if swap_idx is None:
            return new_os

        new_os[idx], new_os[swap_idx] = new_os[swap_idx], new_os[idx]
        return new_os

    def mutate_os(self, os_seq: List[str]) -> List[str]:
        """
        混合 OS 扰动：
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

    def update_os(self, particle: Particle) -> List[str]:
        if particle.pbest_OS is None:
            raise ValueError("particle.pbest_OS is None")
        if self.gbest_OS is None:
            raise ValueError("self.gbest_OS is None")

        new_os = particle.OS[:]

        if self.rng.random() < self.c1_os:
            new_os = self.learn_from_target_os_once(new_os, particle.pbest_OS)

        if self.rng.random() < self.c2_os:
            new_os = self.learn_from_target_os_once(new_os, self.gbest_OS)

        new_os = self.mutate_os(new_os)

        return new_os

    # =========================
    # MS 更新
    # =========================

    def mutate_ms(self, ms_list: List[str]) -> List[str]:
        """
        半贪婪 MS 变异：
        - 70% 选更快机器
        - 30% 随机合法机器
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

    def update_ms(self, particle: Particle) -> List[str]:
        if particle.pbest_MS is None:
            raise ValueError("particle.pbest_MS is None")
        if self.gbest_MS is None:
            raise ValueError("self.gbest_MS is None")

        new_ms = particle.MS[:]

        for idx, (job, op) in enumerate(self.encoder.ms_index_order):
            r = self.rng.random()
            machine_dict = self.operations[job][op]["machines"]
            legal_machines = list(machine_dict.keys())

            if r < self.c1_ms:
                candidate = particle.pbest_MS[idx]
                if candidate in legal_machines:
                    new_ms[idx] = candidate
            elif r < self.c1_ms + self.c2_ms:
                candidate = self.gbest_MS[idx]
                if candidate in legal_machines:
                    new_ms[idx] = candidate

        new_ms = self.mutate_ms(new_ms)

        return new_ms

    # =========================
    # 粒子更新
    # =========================

    def update_particle(self, particle: Particle) -> Particle:
        new_particle = particle.copy()

        new_particle.OS = self.update_os(particle)
        new_particle.MS = self.update_ms(particle)

        self.reset_particle_evaluation(new_particle)
        return new_particle

    def update_swarm_positions(self) -> None:
        self.swarm = [self.update_particle(p) for p in self.swarm]

    # =========================
    # 主循环
    # =========================

    def run_one_iteration(self, store_stats: bool = False) -> None:
        self.update_swarm_positions()
        self.evaluate_swarm(self.swarm, store_stats=store_stats)

    def run(
        self,
        store_stats_init: bool = True,
        store_stats_iterations: bool = False,
        verbose: bool = True
    ) -> Particle:
        self.initialize_swarm()
        self.evaluate_swarm(self.swarm, store_stats=store_stats_init)

        self.history_best_fitness = [self.gbest_fitness]

        if verbose:
            print(f"[Init] best fitness = {self.gbest_fitness}, makespan = {self.gbest_makespan}")

        for it in range(1, self.n_iterations + 1):
            self.run_one_iteration(store_stats=store_stats_iterations)

            self.history_best_fitness.append(self.gbest_fitness)

            if verbose:
                print(f"[Iter {it}] best fitness = {self.gbest_fitness}, makespan = {self.gbest_makespan}")

        return self.best_particle.copy()

    # =========================
    # 调试 / 展示
    # =========================

    def sort_swarm(self, swarm: Optional[List[Particle]] = None) -> List[Particle]:
        if swarm is None:
            swarm = self.swarm

        if any(p.fitness is None for p in swarm):
            raise ValueError("存在未评价粒子，不能排序")

        return sorted(swarm, key=lambda p: p.fitness)

    def print_swarm_summary(self, swarm: Optional[List[Particle]] = None, top_k: int = 5) -> None:
        if swarm is None:
            swarm = self.swarm

        if not swarm:
            print("swarm is empty")
            return

        sorted_swarm = self.sort_swarm(swarm)
        k = min(top_k, len(sorted_swarm))

        print(f"Swarm size: {len(sorted_swarm)}")
        print(f"Top {k} particles:")
        for i in range(k):
            p = sorted_swarm[i]
            print(
                f"[{i}] fitness={p.fitness}, makespan={p.makespan}, "
                f"OS_len={len(p.OS)}, MS_len={len(p.MS)}"
            )

    def print_best_summary(self) -> None:
        if self.best_particle is None:
            print("best particle is None")
            return

        p = self.best_particle
        print("===== Best Particle =====")
        print(f"fitness  : {p.fitness}")
        print(f"makespan : {p.makespan}")
        print(f"OS       : {p.OS}")
        print(f"MS       : {p.MS}")

        if p.stats is not None:
            total_blocking = p.stats["blocking"]["total_blocking_time"]
            print(f"blocking : {total_blocking}")