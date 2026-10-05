"""P_M uses the existing basic mutation instead of MGS."""

from typing import Optional

from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation


class WIPGraphDualPopulationWithoutMGS(WIPGraphDualPopulation):
    algorithm_variant = "wo_mgs"
    pm_specialized_kind = "BasicMutation"
    pm_archive_guidance_enabled = False

    def archive_guide(self) -> Optional[Individual]:
        return None

    def vary_makespan(self, seed: Individual) -> Individual:
        self._last_pm_move_kind = "basic_mutation"
        return self.basic_mutation(seed)
