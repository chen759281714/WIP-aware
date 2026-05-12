from __future__ import annotations

from typing import Any

from src.algorithms.emt_glocal_ga_v2 import EMTGLocalGAV2


class EMTGLocalGAV2_NoGAT(EMTGLocalGAV2):
    """
    消融版本：MT + LAT。

    当前主算法中旧 GAT 已被 CPAT 替代，因此 NoGAT 表示移除 CPAT，
    只保留主种群和 shortage-aware local auxiliary population。
    """

    def __init__(self, *args: Any, **kwargs: Any):
        # 实验脚本会给 NoGAT 传 global_pop_size=0；父类当前不接受 0。
        # 这里保留该消融语义，在进入父类前改成合法占位值，随后初始化时清空 CPAT。
        if kwargs.get("global_pop_size", None) == 0:
            kwargs["global_pop_size"] = 1
        super().__init__(*args, **kwargs)
        self.critical_pop_size = 0
        self.global_pop_size = 0

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

    def generate_global_offspring(self):
        return []

    def evaluate_population_critical(self, population, store_stats: bool = True) -> None:
        return

    def evaluate_population_global(self, population) -> None:
        return

    def environmental_select_critical(self, candidates, target_size: int):
        return []

    def environmental_select_global(self, candidates, target_size: int):
        return []

    def update_global_activity(self) -> None:
        self.global_active = False

    def run_one_generation(self, store_stats: bool = False) -> None:
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
            self.main_population = self.environmental_select_main(main_candidates, self.pop_size)
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
