"""Disable archive guidance while retaining both specialized searches."""

from typing import Optional, Sequence

from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation


class WIPGraphDualPopulationWithoutAC(WIPGraphDualPopulation):
    algorithm_variant = "wo_ac"
    pm_archive_guidance_enabled = False
    ps_archive_guidance_enabled = False

    def archive_guide(self) -> Optional[Individual]:
        return None

    def vary_shortage(self, x: Individual, guide_pool: Sequence[Individual],
                      archive_assisted: bool = False) -> Individual:
        return super().vary_shortage(x, guide_pool, archive_assisted=False)
