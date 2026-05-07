from __future__ import annotations

from typing import List
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.algorithms.emt_glocal_ga_v2 import EMTGLocalGAV2, Individual


class EMTGLocalGAV2_NoLAT(EMTGLocalGAV2):
    """
    NoLAT（MT + GAT）

    ✔ 完全移除 Local Task
    ✔ 保留 Main + Global
    ✔ 重写 run_one_generation
    ✔ 不再调用任何 LAT 逻辑
    """

    # =========================
    # 初始化：去掉 LAT
    # =========================
    def initialize_populations(self) -> None:
        self.main_population = [
            self.initialize_individual(origin_task="main")
            for _ in range(self.pop_size)
        ]

        self.global_population = [
            self.initialize_individual(origin_task="global")
            for _ in range(self.global_pop_size)
        ]

        self.local_population = []  # 不存在 LAT

        self.population = self.main_population
        self.global_active = True

    # =========================
    # 禁用 LAT
    # =========================
    def generate_local_offspring(self):
        return []

    # =========================
    # ⭐ 核心：重写一代进化（无 LAT）
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
        # 2) GAT offspring
        # =========================
        global_offspring = self.generate_global_offspring()

        if global_offspring:
            self.evaluate_population_global(global_offspring)
            global_offspring = [
                ind for ind in global_offspring
                if ind.global_makespan is not None
            ]

        # =========================
        # 3) MT 环境选择（无 LAT）
        # =========================
        o2_for_main = []
        if global_offspring and self.has_budget():
            o2_for_main = self.ensure_evaluated_on_main(
                [ind.copy() for ind in global_offspring],
                store_stats=store_stats
            )

        main_candidates = (
            [ind.copy() for ind in self.main_population] +
            [ind.copy() for ind in main_offspring] +
            o2_for_main
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
        # 4) GAT 环境选择（无 LAT）
        # =========================
        if self.global_active:
            o1_for_global = []

            if main_offspring and self.has_budget():
                o1_for_global = self.ensure_evaluated_on_global(
                    [ind.copy() for ind in main_offspring]
                )

            global_candidates = (
                [ind.copy() for ind in self.global_population if ind.global_makespan is not None] +
                [ind.copy() for ind in global_offspring] +
                o1_for_global
            )

            global_candidates = [
                ind for ind in global_candidates
                if ind.global_makespan is not None
            ]

            if global_candidates:
                self.global_population = self.environmental_select_global(
                    global_candidates,
                    self.global_pop_size
                )
                self.update_global_activity()

    # =========================
    # ⭐ 重写 run（去掉 LAT 初始化）
    # =========================
    def run(
        self,
        store_stats_init: bool = True,
        store_stats_generations: bool = False,
        verbose: bool = True
    ):
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

        # 初始化
        self.initialize_populations()

        # MT 初始化评价
        self.evaluate_population_main(self.main_population, store_stats=store_stats_init)
        self.main_population = [
            ind for ind in self.main_population
            if ind.makespan is not None and ind.shortage is not None
        ]

        if not self.main_population:
            raise RuntimeError("初始化失败：main_population 为空")

        # GAT 初始化评价
        if self.has_budget():
            self.evaluate_population_global(self.global_population)
            self.global_population = [
                ind for ind in self.global_population
                if ind.global_makespan is not None
            ]
        else:
            self.global_population = []

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
                global_best = None
                if self.global_population:
                    valid = [
                        ind.global_makespan
                        for ind in self.global_population
                        if ind.global_makespan is not None
                    ]
                    if valid:
                        global_best = min(valid)

                print(
                    f"[NoLAT Gen {gen}] "
                    f"makespan={best.makespan}, "
                    f"shortage={best.shortage}, "
                    f"pareto={len(self.get_pareto_front())}, "
                    f"global_best={global_best}, "
                    f"evals={self.n_evaluations}"
                )

        self.record_front_snapshot(force=True)

        return self.best_individual.copy()