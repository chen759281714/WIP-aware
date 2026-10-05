# Frozen SIS correctness and validation

This directory records the correction of the P_S relinking layer. The historical
stage-wise implementation is not an equivalence baseline for this correction.

## Formal rules implemented

- Seed diagnosis, phi, H, and machine-related evidence remain fixed.
- OS discrepancy includes every operation of a different job, across all stages.
- Every precedence-feasible gap in the global OS is considered.
- OS insertion must strictly reduce the target operation's discrepancy.
- Gap priority is `(D_after, absolute displacement, insertion index)`.
- Initial units are unique `(operation, OS/MS)` identities stored in a tuple.
- Unit priority is descending phi, ascending operation key, then OS before MS.
- Each initial unit may be applied at most once; unused OS units are rechecked.
- K is `max(1, ceil(eta * initial_unit_count))`; empty units trigger fallback.
- Self-guide prefers actionable strictly lower shortage, then equal shortage.
  Within that eligibility group it uses the first candidate in population order.
- Archive guides use non-extreme filtering and the existing sparse tournament.
  They are not filtered or preferentially grouped by shortage relative to the seed.
- Intermediate atomic moves do not decode. The outer evaluation counts one FE.

## Correctness tests

`tests/test_ps_relinking.py` adds 26 tests, including PS-1 through PS-20,
an independent exhaustive-gap oracle, deterministic ties, unavailable units,
invalid guide machines, fallback, and repeatability. Three existing test methods
were updated to follow the new unit semantics. All 66 relevant unittest tests
passed. The five standalone encoder/decoder function tests also passed.

Full unittest discovery encountered four pre-existing import errors for the
missing `src.problem.instance_generator`. Pytest is unavailable in the Python
runtime used here; no packages were installed.

## Behavior validation before map reuse

`behavior_correctness.json` uses S1/S4/S8 P1_B1, seeds 7 and 23, N=3, N_A=12,
200 FE, and the algorithm's default cooperation settings (T_coop=10, gamma_A=0.2).

| Size | Seed | Applied OS | Applied MS | Total OS discrepancy reduction |
| --- | ---: | ---: | ---: | ---: |
| S1 | 7 | 287 | 90 | 2017 |
| S1 | 23 | 387 | 70 | 3665 |
| S4 | 7 | 685 | 291 | 17752 |
| S4 | 23 | 865 | 458 | 23388 |
| S8 | 7 | 2373 | 1608 | 89632 |
| S8 | 23 | 1921 | 1081 | 62275 |

Every applied OS unit strictly reduced D. All twelve recorded violation counters
were zero, including repeated units, non-H targets, non-initial units, illegal
insertions, machine-evidence violations, and intermediate FE changes.

## Performance equivalence after correctness

The correct pre-reuse algorithm was snapshotted at
`/tmp/wip_sis_correct_baseline.py`. Optimization only reuses position maps within
one candidate analysis or one unchanged working-child state. No persistent
individual cache was added.

`performance_full.json` records per-FE comparisons for S1/S4/S8, seeds 7/23,
200 FE, N=3, N_A=12, T_coop=2, gamma_A=0.4, PYTHONHASHSEED=0. All comparisons
passed, including child genotypes/objectives, full event/provenance signatures,
selected parents/guides, replacements, populations, archive, diagnostics, and RNG.
All behavior violation counters of the optimized runs were also zero.

`timing_full.json` measures uninstrumented runs under the same parameters.
Validation and signatures are computed after stopping the timer.

| Size | Seed | Before (s) | After (s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| S1 | 7 | 0.939 | 0.913 | 1.029x |
| S1 | 23 | 1.020 | 0.976 | 1.044x |
| S8 | 7 | 36.118 | 33.265 | 1.086x |
| S8 | 23 | 31.802 | 29.295 | 1.086x |

These short-run timings are single measurements, not a formal performance study.

## Ablations

`variant_wo_mgs.json`, `variant_wo_sis.json`, and `variant_wo_ac.json` record
per-FE comparisons on S1 P1_B1, seeds 7/23, 100 FE. Every comparison passed.
The variant source files were unchanged. Existing isolation tests also verify
that w/o SIS never calls diagnosis/unit construction, and w/o AC never uses
archive guidance. Full and w/o MGS inherit the new SIS; w/o AC inherits its
self-guide path.

No 240-run pre-ablation experiment was started.
