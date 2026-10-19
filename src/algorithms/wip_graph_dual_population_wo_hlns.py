"""Replace P_S H-LNS by global BasicMutation, retaining its branch draw."""

from src.algorithms.wip_graph_dual_population import Individual, WIPGraphDualPopulation


class WIPGraphDualPopulationWithoutHLNS(WIPGraphDualPopulation):
    def __init__(self, operations, buffers, *, variant="HLNS_WO_HLNS", **kwargs):
        super().__init__(operations, buffers, variant=variant, **kwargs)

    @property
    def algorithm_variant(self):
        return "HLNS_WO_HLNS"

    @property
    def ps_specialized_kind(self):
        return "BasicMutation"

    def vary_shortage(self, x: Individual) -> Individual:
        self._last_ps_move_kind = "basic_mutation"
        if self.rng.random() < self.p_mut:
            return self.basic_mutation(x)
        return self.basic_mutation(x)
