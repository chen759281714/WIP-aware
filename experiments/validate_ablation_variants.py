"""Small mechanism smoke test for the Full algorithm and three ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_compare_experiments import ALGORITHMS


VARIANTS = (
    ("Full", "WIPGraphDualPopulation", True, True, True, True),
    ("w/o MGS", "WIPGraphDualPopulationWithoutMGS", False, True, False, True),
    ("w/o SIS", "WIPGraphDualPopulationWithoutSIS", True, False, True, False),
    ("w/o AC", "WIPGraphDualPopulationWithoutAC", True, True, False, False),
)


def run_smoke_case(instance: dict, algorithm_name: str, seed: int,
                   N: int, N_A: int, FE_max: int) -> dict:
    search = ALGORITHMS[algorithm_name](instance["operations"], instance["buffers"],
                                        N=N, N_A=N_A, FE_max=FE_max, seed=seed)
    archive = search.run()
    diag = search.diagnostics
    cooperation = diag["cooperation"]
    result = {
        "algorithm": algorithm_name,
        "seed": seed,
        "FE": search.n_evaluations,
        "archive_size": len(archive),
        "min_makespan": min(ind.makespan for ind in archive),
        "min_shortage": min(ind.shortage for ind in archive),
        "MGS_calls": len(diag["pm_action_sets"]),
        "SIS_calls": diag["shortage_diagnosis"]["calls"],
        "PM_archive_seeds_used": cooperation["pm_usable"],
        "PS_archive_guide_calls": cooperation["ps_probes"],
        "PS_archive_guides_used": cooperation["ps_valid_guide"],
        "BasicMutation_calls": sum(
            counts["generated"]
            for group in ("pm_moves", "ps_moves")
            for kind, counts in diag[group].items()
            if kind == "basic_mutation" or kind.startswith("fallback_basic_mutation")
        ),
    }
    for key in ("algorithm_variant", "pm_specialized_kind", "ps_specialized_kind",
                "pm_archive_guidance_enabled", "ps_archive_guidance_enabled"):
        result[key] = diag[key]
    assert result["FE"] == FE_max
    assert len(search.P_M) == len(search.P_S) == N
    assert 1 <= len(archive) <= N_A
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path,
                        default=ROOT / "data/final_benchmark/instances/WIPHFSP_S1_P1_B1.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--N-A", type=int, default=20)
    parser.add_argument("--fe-max", type=int, default=300)
    args = parser.parse_args()
    instance = json.loads(args.instance.read_text(encoding="utf-8"))
    for label, name, mgs, sis, archive_pm, archive_ps in VARIANTS:
        for seed in args.seeds:
            result = run_smoke_case(instance, name, seed, args.N, args.N_A, args.fe_max)
            assert (result["MGS_calls"] > 0) == mgs
            assert (result["SIS_calls"] > 0) == sis
            assert result["pm_archive_guidance_enabled"] == archive_pm
            assert result["ps_archive_guidance_enabled"] == archive_ps
            if archive_pm:
                assert result["PM_archive_seeds_used"] > 0
            else:
                assert result["PM_archive_seeds_used"] == 0
            if archive_ps:
                assert result["PS_archive_guide_calls"] > 0
                assert result["PS_archive_guides_used"] > 0
            else:
                assert result["PS_archive_guide_calls"] == 0
            print(f"{label}, seed={seed}: {json.dumps(result, ensure_ascii=False, sort_keys=True)}",
                  flush=True)


if __name__ == "__main__":
    main()
