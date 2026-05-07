from __future__ import annotations

from typing import List
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.algorithms.emt_glocal_ga_v2 import EMTGLocalGAV2, Individual


class EMTGLocalGAV2_NoGAT(EMTGLocalGAV2):
    """
    NoGAT（MT + LAT）

    ✔ 完全移除 Global Task
    ✔ 保留 Main + Local
    ✔ 重写 run_one_generation（核心）
    ✔ 不再调用任何 global 逻辑
    """

    # =========================
    # 初始化：彻底移除 GAT
    # =========================
    def initialize_populations(self) -> None:
        self.main_population = [
            self.initialize_individual(origin_task="main")
            for _ in range(self.pop_size)
        ]

        self.local_population = [
            self.initialize_individual(origin_task="local")
            for _ in range(self.local_pop_size)
        ]

        self.global_population = []  # 完全不用
        self.global_active = False

        self.population = self.main_population

    # =========================
    # 禁用所有 GAT 相关接口
    # =========================
    def generate_global_offspring(self):
        return []

    def evaluate_population_global(self, population):
        return

    def ensure_evaluated_on_global(self, population):
        return []

    def environmental_select_global(self, candidates, target_size):
        return []

    def update_global_activity(self):
        self.global_active = False

    # =========================
    # ⭐ 核心：重写一代进化（无 GAT）
    # =========================
    def run_one_generation(self, store_stats: bool = False, generation: int = 1) -> None:
        if not self.has_budget():
            return

        # =========================
        # 1) MT offspring
        # =========================
        main_offspring = self.generate_main_offspring()

        if main_offspring:
            self.evaluate_population_main(main_offspring, store_stats=store_stats)
            main_offspring = [
                ind for ind in main_offspring
                if ind.makespan is not None and ind.shortage is not None
            ]

        # =========================
        # 2) LAT offspring
        # =========================
        local_offspring = []
        if self.has_budget():
            local_offspring = self.generate_local_offspring()

            if local_offspring:
                self.evaluate_population_main(local_offspring, store_stats=store_stats)
                local_offspring = [
                    ind for ind in local_offspring
                    if ind.makespan is not None and ind.shortage is not None
                ]

        # =========================
        # 3) MT 环境选择（无 GAT）
        # =========================
        main_candidates = (
            [ind.copy() for ind in self.main_population] +
            [ind.copy() for ind in main_offspring] +
            [ind.copy() for ind in local_offspring]
        )

        main_candidates = [
            ind for ind in main_candidates
            if ind.makespan is not None and ind.shortage is not None
        ]

        if main_candidates:
            self.main_population = self.environmental_select_main(
                main_candidates,
                self.pop_size
            )
            self.assign_rank_and_crowding(self.main_population)
            self.population = self.main_population
            self._update_best(self.main_population)

        # =========================
        # 4) LAT 环境选择（无 GAT）
        # =========================
        local_candidates = (
            [ind.copy() for ind in self.local_population] +
            [ind.copy() for ind in local_offspring] +
            [ind.copy() for ind in main_offspring]
        )

        local_candidates = [
            ind for ind in local_candidates
            if ind.makespan is not None and ind.shortage is not None
        ]

        if local_candidates:
            self.local_population = self.environmental_select_main(
                local_candidates,
                self.local_pop_size
            )
            self.assign_rank_and_crowding(self.local_population)

    # =========================
    # ⭐ 重写 run（去掉 GAT 初始化）
    # =========================
    def run(
        self,
        store_stats_init: bool = True,
        store_stats_generations: bool = False,
        verbose: bool = True
    ):
        self.n_evaluations = 0

        self.main_population = []
        self.local_population = []
        self.global_population = []

        self.population = self.main_population
        self.best_individual = None

        self.history_best_fitness = []
        self.history_best_shortage = []
        self.history_eval_counts = []
        self.history_fronts = []
        self._last_snapshot_eval = -1

        # 初始化
        self.initialize_populations()

        # 主任务初始化评价
        self.evaluate_population_main(self.main_population, store_stats=store_stats_init)
        self.main_population = [
            ind for ind in self.main_population
            if ind.makespan is not None and ind.shortage is not None
        ]

        if not self.main_population:
            raise RuntimeError("初始化失败：main_population 为空")

        # 初始化 LAT
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

        init_best = self.get_best_individual(self.main_population)

        self.history_best_fitness = [float(init_best.makespan)]
        self.history_best_shortage = [float(init_best.shortage)]
        self.history_eval_counts = [int(self.n_evaluations)]

        self.record_front_snapshot(force=True)

        # =========================
        # 主循环
        # =========================
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

            self.run_one_generation(
                store_stats=store_stats_generations,
                generation=gen
            )

            if self.n_evaluations == prev_evals:
                break

            best = self.get_best_individual(self.main_population)

            self.history_best_fitness.append(float(best.makespan))
            self.history_best_shortage.append(float(best.shortage))
            self.history_eval_counts.append(int(self.n_evaluations))

            self.record_front_snapshot(force=False)

            if verbose:
                print(
                    f"[NoGAT Gen {gen}] "
                    f"makespan={best.makespan}, "
                    f"shortage={best.shortage}, "
                    f"pareto={len(self.get_pareto_front())}, "
                    f"evals={self.n_evaluations}"
                )

        self.record_front_snapshot(force=True)

        return self.best_individual.copy()