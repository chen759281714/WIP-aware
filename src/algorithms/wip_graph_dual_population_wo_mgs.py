"""Remove only MGS; keep the original P_M archive seed pipeline."""

from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation


class WIPGraphDualPopulationWithoutMGS(WIPGraphDualPopulation):
    def __init__(self, operations, buffers, *, variant="HLNS_WO_MGS", **kwargs):
        super().__init__(operations, buffers, variant=variant, **kwargs)

    @property
    def algorithm_variant(self):
        return "HLNS_WO_MGS"

    @property
    def pm_specialized_kind(self):
        return "BasicMutation"

    def _vary_makespan_specialized(self, seed):
        self._last_pm_selected_tags = {"unblocking": False, "wip": False}
        self._last_pm_selected_action = None
        self._last_pm_move_kind = "basic_mutation"
        return self.basic_mutation(seed)
