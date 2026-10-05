"""P_S uses the existing basic mutation instead of SIS."""

from typing import Sequence

from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation


class WIPGraphDualPopulationWithoutSIS(WIPGraphDualPopulation):
    algorithm_variant = "wo_sis"
    ps_specialized_kind = "BasicMutation"
    ps_archive_guidance_enabled = False

    def vary_shortage(self, x: Individual, guide_pool: Sequence[Individual],
                      archive_assisted: bool = False) -> Individual:
        self._last_ps_guide_source = "self"
        self._last_ps_move_kind = "basic_mutation"
        return self.basic_mutation(x)
