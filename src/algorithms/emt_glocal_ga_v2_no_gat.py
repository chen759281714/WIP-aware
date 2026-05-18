from __future__ import annotations

from typing import Any

from src.algorithms.emt_glocal_ga_v2 import EMTGLocalGAV2


class EMTGLocalGAV2_NoGAT(EMTGLocalGAV2):
    """
    消融版本：MT + LAT，移除当前主算法中的 CPAT/BACP 辅助种群。

    类名保留 NoGAT 是为了兼容既有实验脚本；在当前版本中，
    旧 GAT 已被 CPAT 替代，因此该类实际用于验证 CPAT 的有效性。

    LAT 保持与主算法一致：WIP-aware shortage-oriented global auxiliary
    population，使用真实 WIP 评价、shortage-first selection、buffer-aware
    crossover 和 shortage-guided mutation。
    """

    def __init__(self, *args: Any, **kwargs: Any):
        # 实验脚本会给 NoGAT 传 global_pop_size=0；父类当前不接受 0。
        # 这里保留该消融语义，在进入父类前改成合法占位值，随后初始化时清空 CPAT。
        if kwargs.get("global_pop_size", None) == 0:
            kwargs["global_pop_size"] = 1
        super().__init__(*args, **kwargs)
        self.critical_pop_size = 0
        self.global_pop_size = 0
        self.critical_migration_count = 0

    def initialize_populations(self) -> None:
        self.main_population = [
            self.initialize_individual(origin_task="main")
            for _ in range(self.pop_size)
        ]
        self.critical_population = []
        self.global_population = self.critical_population
        self.local_population = [
            self.initialize_local_individual(i)
            for i in range(self.local_pop_size)
        ]
        self.population = self.main_population
        self.global_active = False

    def generate_critical_offspring(self):
        return []

    def generate_bacp_guided_neighbors(self, ind):
        return []

    def select_critical_migration_candidates(self, critical_offspring, main_population):
        return []

    def evaluate_population_critical(self, population, store_stats: bool = True) -> None:
        return

    def environmental_select_critical(self, candidates, target_size: int):
        return []

    def run_one_generation(self, store_stats: bool = False) -> None:
        """
        只执行 MT + LAT 协同进化，不生成、不评价、不融合 CPAT/BACP 个体。
        LAT offspring 由父类当前的 shortage-oriented global GA 生成。
        """
        if not self.has_budget():
            return

        main_offspring = self.generate_main_offspring()
        if main_offspring:
            self.evaluate_population_main(main_offspring, store_stats=store_stats)
            main_offspring = [
                ind for ind in main_offspring
                if ind.makespan is not None and ind.shortage is not None
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

        local_migrants = self.select_local_migration_candidates(
            local_offspring,
            self.main_population
        )

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
            selected_main = self.environmental_select_main(main_candidates, self.pop_size)
            self.main_population = self.apply_protected_migration(
                selected_main,
                local_migrants,
                self.pop_size
            )
            self.assign_rank_and_crowding(self.main_population)
            self.population = self.main_population
            self._update_best(self.main_population)

        local_candidates = (
            [ind.copy() for ind in self.local_population if ind.makespan is not None and ind.shortage is not None] +
            [ind.copy() for ind in local_offspring] +
            [ind.copy() for ind in main_offspring]
        )
        local_candidates = [
            ind for ind in local_candidates
            if ind.makespan is not None and ind.shortage is not None
        ]

        if local_candidates:
            self.local_population = self.environmental_select_local(
                local_candidates,
                self.local_pop_size
            )
