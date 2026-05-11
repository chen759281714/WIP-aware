from __future__ import annotations

from typing import Any

from src.algorithms.emt_glocal_ga_v2 import EMTGLocalGAV2


class EMTGLocalGAV2_NoLAT(EMTGLocalGAV2):
    """
    消融版本：MT + CPAT。

    当前主算法中 CPAT 替代了旧 GAT，因此 NoLAT 表示移除
    shortage-aware local auxiliary population，只保留主种群和 CPAT。
    """

    def __init__(self, *args: Any, **kwargs: Any):
        # 实验脚本会给 NoLAT 传 local_pop_size=0；父类当前不接受 0。
        # 这里保留该消融语义，在进入父类前改成合法占位值，随后初始化时清空 LAT。
        if kwargs.get("local_pop_size", None) == 0:
            kwargs["local_pop_size"] = 1
        super().__init__(*args, **kwargs)
        self.local_pop_size = 0

    def initialize_populations(self) -> None:
        self.main_population = [
            self.initialize_individual(origin_task="main")
            for _ in range(self.pop_size)
        ]
        self.critical_population = [
            self.initialize_individual(origin_task="critical")
            for _ in range(self.critical_pop_size)
        ]
        self.global_population = self.critical_population
        self.local_population = []
        self.population = self.main_population
        self.global_active = False

    def generate_local_offspring(self):
        return []

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

        critical_offspring = []
        if self.has_budget():
            critical_offspring = self.generate_critical_offspring()
            if critical_offspring:
                self.evaluate_population_critical(critical_offspring, store_stats=store_stats)
                critical_offspring = [
                    ind for ind in critical_offspring
                    if ind.makespan is not None and ind.shortage is not None
                ]

        main_candidates = (
            [ind.copy() for ind in self.main_population] +
            [ind.copy() for ind in main_offspring] +
            [ind.copy() for ind in critical_offspring]
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

        critical_candidates = (
            [ind.copy() for ind in self.critical_population if ind.makespan is not None and ind.shortage is not None] +
            [ind.copy() for ind in critical_offspring] +
            [ind.copy() for ind in main_offspring]
        )
        critical_candidates = [
            ind for ind in critical_candidates
            if ind.makespan is not None and ind.shortage is not None
        ]

        if critical_candidates:
            self.critical_population = self.environmental_select_critical(
                critical_candidates,
                self.critical_pop_size
            )
            self.global_population = self.critical_population
