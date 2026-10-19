"""Disable A-to-P_M seed guidance, not the shared Pareto archive."""

from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation


class WIPGraphDualPopulationWithoutAC(WIPGraphDualPopulation):
    def __init__(self, operations, buffers, *, variant="HLNS_WO_AC", **kwargs):
        super().__init__(operations, buffers, variant=variant, **kwargs)

    @property
    def algorithm_variant(self):
        return "HLNS_WO_AC"

    @property
    def pm_archive_guidance_enabled(self):
        return False

    def archive_guide(self):
        return None
