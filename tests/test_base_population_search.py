import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.problem.instance_generator import (
    InstanceSpec,
    generate_fms_wip_instance_auto_caps,
)
from src.algorithms.base_population_search import BasePopulationSearch


def test_base_population_search_run():
    spec = InstanceSpec(
        num_stages=3,
        machines_per_stage=2,
        n_jobs=10,
        buffer_caps=[1, 1],   # 占位
        pt_profile="mid_bottleneck",
        seed=1,
        os_repeat=20,
    )

    spec, operations, buffers, _, _ = generate_fms_wip_instance_auto_caps(spec)

    search = BasePopulationSearch(
            operations=operations,
            buffers=buffers,
            pop_size=6,
            n_generations=5,
            crossover_rate=0.8,
            os_mutation_rate=0.2,
            ms_mutation_rate=0.1,
            tournament_size=2,
            seed=42,
            use_local_search=False,
            # ls_apply_mode="elite"
            # elite_ls_count=2,
            # ls_max_tries=4,
        )

    best = search.run(
        store_stats_init=True,
        store_stats_generations=False,
        verbose=True
    )

    assert best is not None
    assert best.fitness is not None
    assert best.makespan is not None
    assert len(search.history_best_fitness) == 6  # init + 5 generations

    print("\nHistory best fitness:", search.history_best_fitness)
    search.print_best_summary()


if __name__ == "__main__":
    test_base_population_search_run()
    print("✅ base population search run test passed")