# Full FE convergence audit

## A. Execution
Completed 60/60 runs; actual FE=60000 each; total FE=3600000.
Parameters: {'N': 50, 'N_A': 100, 'p_mut': 0.1, 'rho': 0.8, 'T_coop': 10, 'FE_max': 60000}. Sum of run elapsed times: 422496.58 seconds (not wall time).
Existing diagnostic_callback reused after archive updates; no algorithm/encoder/decoder edits.
FE_max only caps evaluations and final batch allocation. All candidate checkpoints are complete batches.
Resume skips complete source-validated runs; interrupted runs restart from initialization.

## B. Fixed-scale metrics
Each instance uses all initialization/checkpoint archives across all five seeds for fixed bounds and empirical reference.
NHV=HV/1.21, reference=(1.1,1.1). IGD reference is empirical, not the true Pareto front.
Extremes may belong to different schedules. Cross-instance raw extremes are not used as normalized curves.
|FE|NHV mean (SD)|IGD mean (SD)|R_H mean|R_I mean|R_C mean|R_S mean|four-limit pass rate|
|---:|---:|---:|---:|---:|---:|---:|---:|
|5000|0.682587 (0.118766)|0.214052 (0.087363)|35.30%|72.87%|4.15%|26.34%|0.0%|
|10000|0.751659 (0.089622)|0.159754 (0.062960)|20.82%|64.48%|2.66%|19.42%|0.0%|
|15000|0.786551 (0.077237)|0.132859 (0.051647)|14.97%|58.19%|1.90%|15.82%|0.0%|
|20000|0.813555 (0.067256)|0.113675 (0.043536)|10.86%|51.99%|1.36%|12.93%|0.0%|
|25000|0.831098 (0.062603)|0.101724 (0.041686)|8.41%|46.20%|0.93%|10.93%|0.0%|
|30000|0.847209 (0.057570)|0.092117 (0.036821)|6.24%|40.95%|0.67%|8.20%|0.0%|
|35000|0.857904 (0.053618)|0.084486 (0.032893)|4.85%|36.19%|0.49%|7.18%|1.7%|
|40000|0.867857 (0.050432)|0.076511 (0.030546)|3.60%|29.78%|0.36%|5.67%|5.0%|
|45000|0.877137 (0.047105)|0.069095 (0.027763)|2.46%|23.13%|0.27%|4.26%|8.3%|
|50000|0.885920 (0.045185)|0.066526 (0.027958)|1.42%|18.51%|0.16%|2.22%|16.7%|
|55000|0.892843 (0.043752)|0.061897 (0.030389)|0.62%|10.64%|0.06%|0.85%|38.3%|
|60000|0.898314 (0.043178)|0.057304 (0.031004)|0.00%|0.00%|0.00%|0.00%|100.0%|

Pass rates at 55000/60000 have insufficient follow-up and do not qualify a budget.

## C. Predeclared decision
{
  "limits": {
    "nhv": 0.01,
    "igd": 0.05,
    "makespan": 0.01,
    "shortage": 0.03
  },
  "overall_min_rate": 0.85,
  "size_min_rate": 0.8,
  "window_width_FE": 5000,
  "window_count": 2,
  "windows": "Peak improvement over all 1000-FE samples, not just endpoints; same limits and coverage.",
  "instance_guard": "Every instance mean residual NHV<=1% and IGD<=5%, including both windows.",
  "instance_earliest": "At least 4/5 seeds pass and instance mean NHV/IGD guard, including both windows.",
  "late_event": "A 5000-FE window passes all four limits, followed later by a window that fails at least one.",
  "candidate_FE": [
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    45000,
    50000
  ],
  "reference": "All snapshots including initialization, all five seeds; empirical only."
}
Recommended FE: None; evidence: NO_VALIDATED_BUDGET_WITHIN_OBSERVATION.
If none qualifies, 60000 is only a provisional comparison cap, NOT a validated convergence budget.
|FE|overall|S1|S4|S8|instance guard|window 1|window 2|qualified|
|---:|---:|---:|---:|---:|---|---|---|---|
|10000|0.0%|0.0%|0.0%|0.0%|False|False|False|False|
|15000|0.0%|0.0%|0.0%|0.0%|False|False|False|False|
|20000|0.0%|0.0%|0.0%|0.0%|False|False|False|False|
|25000|0.0%|0.0%|0.0%|0.0%|False|False|False|False|
|30000|0.0%|0.0%|0.0%|0.0%|False|False|False|False|
|35000|1.7%|5.0%|0.0%|0.0%|False|False|False|False|
|40000|5.0%|10.0%|5.0%|0.0%|False|False|False|False|
|45000|8.3%|15.0%|10.0%|0.0%|False|False|False|False|
|50000|16.7%|25.0%|15.0%|10.0%|False|False|False|False|

## D. Instance boundaries
[
  {
    "instance": "WIPHFSP_S1_P1_B1",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S1_P2_B2",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S1_P3_B3",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S1_P4_B1",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S4_P1_B2",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S4_P4_B2",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S8_P2_B1",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S8_P3_B2",
    "earliest_qualified_FE": null
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "earliest_qualified_FE": null
  }
]
Failures at FE=50000:
[
  {
    "instance": "WIPHFSP_S1_P1_B1",
    "seed": 0,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.01324634833359256,
    "R_igd": 0.05466625958156436,
    "R_makespan": 0.0,
    "R_shortage": 0.07971014492753623
  },
  {
    "instance": "WIPHFSP_S1_P1_B1",
    "seed": 1,
    "failed_metrics": [
      "shortage"
    ],
    "R_nhv": 0.0038656882078246845,
    "R_igd": 0.02483691684842212,
    "R_makespan": 0.0,
    "R_shortage": 0.0392156862745098
  },
  {
    "instance": "WIPHFSP_S1_P1_B1",
    "seed": 2,
    "failed_metrics": [
      "igd",
      "shortage"
    ],
    "R_nhv": 0.008983199245522148,
    "R_igd": 0.7887510764133316,
    "R_makespan": 0.0,
    "R_shortage": 0.06796116504854369
  },
  {
    "instance": "WIPHFSP_S1_P1_B1",
    "seed": 3,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.005038536305202938,
    "R_igd": 0.3466415094451406,
    "R_makespan": 0.0,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S1_P2_B2",
    "seed": 0,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.010543356923484392,
    "R_igd": 0.10213778078101185,
    "R_makespan": 0.0,
    "R_shortage": 0.002600780234070221
  },
  {
    "instance": "WIPHFSP_S1_P2_B2",
    "seed": 1,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.010414574147982214,
    "R_igd": 0.4030710717813327,
    "R_makespan": 0.0,
    "R_shortage": 0.06419400855920114
  },
  {
    "instance": "WIPHFSP_S1_P2_B2",
    "seed": 2,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.013236095864497015,
    "R_igd": 0.08922272607008508,
    "R_makespan": 0.0,
    "R_shortage": 0.0203527815468114
  },
  {
    "instance": "WIPHFSP_S1_P2_B2",
    "seed": 4,
    "failed_metrics": [
      "igd",
      "shortage"
    ],
    "R_nhv": 0.009662529763418434,
    "R_igd": 0.4623617498628587,
    "R_makespan": 0.0,
    "R_shortage": 0.03494176372712146
  },
  {
    "instance": "WIPHFSP_S1_P3_B3",
    "seed": 0,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.013072711657015894,
    "R_igd": 0.19269839249796922,
    "R_makespan": 0.0,
    "R_shortage": 0.022709475332811275
  },
  {
    "instance": "WIPHFSP_S1_P3_B3",
    "seed": 2,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.011468657733512204,
    "R_igd": 0.3996465774077601,
    "R_makespan": 0.0,
    "R_shortage": 0.010512483574244415
  },
  {
    "instance": "WIPHFSP_S1_P3_B3",
    "seed": 3,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.022141721988024606,
    "R_igd": 0.477162867437219,
    "R_makespan": 0.0,
    "R_shortage": 0.01675041876046901
  },
  {
    "instance": "WIPHFSP_S1_P3_B3",
    "seed": 4,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.004223160871642797,
    "R_igd": 0.051175405906495615,
    "R_makespan": 0.0,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S1_P4_B1",
    "seed": 1,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.001220482411623517,
    "R_igd": 0.0863214691040091,
    "R_makespan": 0.0,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S1_P4_B1",
    "seed": 3,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.0019493876186524007,
    "R_igd": 0.14780698907198528,
    "R_makespan": 0.0,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S1_P4_B1",
    "seed": 4,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.002215528236967874,
    "R_igd": 0.2002120052375934,
    "R_makespan": 0.0,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S4_P1_B2",
    "seed": 0,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.006185787862333558,
    "R_igd": 0.6398641873089541,
    "R_makespan": 0.0,
    "R_shortage": 0.012877939529675251
  },
  {
    "instance": "WIPHFSP_S4_P1_B2",
    "seed": 1,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.022800202323346733,
    "R_igd": 0.18917420718220973,
    "R_makespan": 0.006485849056603774,
    "R_shortage": 0.018867924528301886
  },
  {
    "instance": "WIPHFSP_S4_P1_B2",
    "seed": 2,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.016036931472787955,
    "R_igd": 0.16364341284717293,
    "R_makespan": 0.0,
    "R_shortage": 0.04157303370786517
  },
  {
    "instance": "WIPHFSP_S4_P1_B2",
    "seed": 3,
    "failed_metrics": [
      "nhv",
      "shortage"
    ],
    "R_nhv": 0.017711483437350573,
    "R_igd": 0.04988887796069178,
    "R_makespan": 0.0,
    "R_shortage": 0.050920910075839654
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "seed": 0,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.022313000034787575,
    "R_igd": 0.23487787743567895,
    "R_makespan": 0.0,
    "R_shortage": 0.030299204645878045
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "seed": 1,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.034226587145519134,
    "R_igd": 0.15371616873107782,
    "R_makespan": 0.0,
    "R_shortage": 0.039425651009978094
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "seed": 2,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.013818240350320914,
    "R_igd": 0.11440853526683485,
    "R_makespan": 0.0,
    "R_shortage": 0.014195213514781792
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "seed": 3,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.03392135867117882,
    "R_igd": 0.10081782441256532,
    "R_makespan": 0.0,
    "R_shortage": 0.042421462967209354
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "seed": 4,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.005726022732584799,
    "R_igd": 0.09259140229603512,
    "R_makespan": 0.0,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "seed": 0,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.0016923016037474182,
    "R_igd": 0.7257702621943511,
    "R_makespan": 0.0,
    "R_shortage": 0.013761467889908258
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "seed": 1,
    "failed_metrics": [
      "nhv",
      "igd",
      "makespan"
    ],
    "R_nhv": 0.04551732106330793,
    "R_igd": 0.11033190369344081,
    "R_makespan": 0.017846862406447898,
    "R_shortage": 0.020356234096692113
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "seed": 2,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.008562192589613337,
    "R_igd": 0.14030648451440864,
    "R_makespan": 0.0017513134851138354,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "seed": 3,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.03764351148932749,
    "R_igd": 0.5563099287040809,
    "R_makespan": 0.004626951995373048,
    "R_shortage": 0.21301775147928995
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "seed": 4,
    "failed_metrics": [
      "nhv",
      "igd",
      "makespan",
      "shortage"
    ],
    "R_nhv": 0.03564099294145697,
    "R_igd": 0.07220599468756768,
    "R_makespan": 0.01099537037037037,
    "R_shortage": 0.08303249097472924
  },
  {
    "instance": "WIPHFSP_S4_P4_B2",
    "seed": 0,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.012508543901261786,
    "R_igd": 0.15360096218794606,
    "R_makespan": 0.0,
    "R_shortage": 0.052099533437014
  },
  {
    "instance": "WIPHFSP_S4_P4_B2",
    "seed": 2,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.004300404592292888,
    "R_igd": 0.2734340117507679,
    "R_makespan": 0.0,
    "R_shortage": 0.011863224005582694
  },
  {
    "instance": "WIPHFSP_S4_P4_B2",
    "seed": 3,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.00956887384892154,
    "R_igd": 0.1391376049147997,
    "R_makespan": 0.0045841519318926,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "seed": 0,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.020903225806451146,
    "R_igd": 0.2811571450176789,
    "R_makespan": 0.0,
    "R_shortage": 0.0346553715825953
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "seed": 1,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.010903316711685962,
    "R_igd": 0.07549451652751428,
    "R_makespan": 0.00035486160397445,
    "R_shortage": 0.01347305389221557
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "seed": 2,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.006875617736015331,
    "R_igd": 0.10622382260234721,
    "R_makespan": 0.00035298270384751147,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "seed": 3,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.01232848524642502,
    "R_igd": 0.29808050505130235,
    "R_makespan": 0.0,
    "R_shortage": 0.013805522208883553
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "seed": 4,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.019545157404096087,
    "R_igd": 0.10829488697848552,
    "R_makespan": 0.0,
    "R_shortage": 0.03596861798430899
  },
  {
    "instance": "WIPHFSP_S8_P2_B1",
    "seed": 1,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.012688139194492574,
    "R_igd": 0.10252965138808354,
    "R_makespan": 0.0,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S8_P2_B1",
    "seed": 2,
    "failed_metrics": [
      "nhv",
      "igd",
      "shortage"
    ],
    "R_nhv": 0.06546006962960106,
    "R_igd": 0.407388830630005,
    "R_makespan": 0.005239030779305829,
    "R_shortage": 0.06392557022809124
  },
  {
    "instance": "WIPHFSP_S8_P2_B1",
    "seed": 3,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.019401088865563187,
    "R_igd": 0.10339169635204294,
    "R_makespan": 0.006140917905623788,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S8_P2_B1",
    "seed": 4,
    "failed_metrics": [
      "nhv",
      "igd",
      "makespan"
    ],
    "R_nhv": 0.035232480069278954,
    "R_igd": 0.27657370948721427,
    "R_makespan": 0.0107421875,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S8_P3_B2",
    "seed": 0,
    "failed_metrics": [
      "nhv"
    ],
    "R_nhv": 0.017201793929414778,
    "R_igd": 0.03413571375729263,
    "R_makespan": 0.0024204702627939143,
    "R_shortage": 0.011728567439262776
  },
  {
    "instance": "WIPHFSP_S8_P3_B2",
    "seed": 2,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.031552460997902654,
    "R_igd": 0.1295484393281877,
    "R_makespan": 0.007306889352818371,
    "R_shortage": 0.009762711864406779
  },
  {
    "instance": "WIPHFSP_S8_P3_B2",
    "seed": 3,
    "failed_metrics": [
      "nhv"
    ],
    "R_nhv": 0.010124680802566545,
    "R_igd": 0.0,
    "R_makespan": 0.0006894174422612892,
    "R_shortage": 0.004105090311986864
  },
  {
    "instance": "WIPHFSP_S8_P3_B2",
    "seed": 4,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.02838949253453535,
    "R_igd": 0.11179021474832845,
    "R_makespan": 0.0,
    "R_shortage": 0.024653085326247416
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "seed": 0,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.008757165895896226,
    "R_igd": 0.2429955720596748,
    "R_makespan": 0.0,
    "R_shortage": 0.007759616972098399
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "seed": 1,
    "failed_metrics": [
      "igd"
    ],
    "R_nhv": 0.0047752822957732935,
    "R_igd": 0.19201311077991137,
    "R_makespan": 0.0006414368184733803,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "seed": 2,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.010106655568419816,
    "R_igd": 0.19490243331551765,
    "R_makespan": 0.0,
    "R_shortage": 0.0
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "seed": 3,
    "failed_metrics": [
      "nhv",
      "igd"
    ],
    "R_nhv": 0.012139182110076937,
    "R_igd": 0.0636538494011508,
    "R_makespan": 0.0,
    "R_shortage": 0.01851275668948351
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "seed": 4,
    "failed_metrics": [
      "nhv",
      "igd",
      "makespan"
    ],
    "R_nhv": 0.022185138452323123,
    "R_igd": 0.45275738838360186,
    "R_makespan": 0.011457670273711012,
    "R_shortage": 0.007712082262210797
  }
]
## E. Late improvements
Quiet-then-resurgent windows: 64.
See late_improvement_events.csv; this is observed trajectory behavior, not causal proof.
Final 55000-60000 windows:
[
  {
    "instance": "WIPHFSP_S1_P1_B1",
    "seed": 0,
    "R_nhv": 0.0015014453678943297,
    "R_igd": 0.021584600179162118,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S1_P1_B1",
    "seed": 1,
    "R_nhv": 0.003819592107220016,
    "R_igd": 0.01987787165738387,
    "R_makespan": 0.0,
    "R_shortage": 0.0392156862745098,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S1_P1_B1",
    "seed": 2,
    "R_nhv": 0.0026050027652006116,
    "R_igd": 0.6048769676147401,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S1_P1_B1",
    "seed": 3,
    "R_nhv": 0.0019474245211041853,
    "R_igd": 0.13113556907687973,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S1_P1_B1",
    "seed": 4,
    "R_nhv": 0.0005629516187684819,
    "R_igd": 0.00438655643725899,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S1_P2_B2",
    "seed": 0,
    "R_nhv": 0.0035806099782830524,
    "R_igd": 0.03475281820820154,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S1_P2_B2",
    "seed": 1,
    "R_nhv": 0.0010813926386922911,
    "R_igd": 0.023077800353281802,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S1_P2_B2",
    "seed": 2,
    "R_nhv": 0.005432249855339201,
    "R_igd": 0.04733509640036144,
    "R_makespan": 0.0,
    "R_shortage": 0.008241758241758242,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S1_P2_B2",
    "seed": 3,
    "R_nhv": 0.0033726461985210177,
    "R_igd": 0.027832534134336895,
    "R_makespan": 0.0,
    "R_shortage": 0.013761467889908258,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S1_P2_B2",
    "seed": 4,
    "R_nhv": 0.0027685817765885005,
    "R_igd": 0.24238057301444227,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S1_P3_B3",
    "seed": 0,
    "R_nhv": 0.007995986832738902,
    "R_igd": 0.12450320320584776,
    "R_makespan": 0.0,
    "R_shortage": 0.022709475332811275,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S1_P3_B3",
    "seed": 1,
    "R_nhv": 0.00441433509320463,
    "R_igd": 0.0,
    "R_makespan": 0.0,
    "R_shortage": 0.016751638747268753,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S1_P3_B3",
    "seed": 2,
    "R_nhv": 0.006007077907979913,
    "R_igd": 0.3284686748284282,
    "R_makespan": 0.0,
    "R_shortage": 0.010512483574244415,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S1_P3_B3",
    "seed": 3,
    "R_nhv": 0.011164637457461313,
    "R_igd": 0.3504332051688262,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S1_P3_B3",
    "seed": 4,
    "R_nhv": 0.001512052789412327,
    "R_igd": 0.014020683978271624,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S1_P4_B1",
    "seed": 0,
    "R_nhv": 7.364434180582361e-06,
    "R_igd": 0.0,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S1_P4_B1",
    "seed": 1,
    "R_nhv": 0.0009083895795887105,
    "R_igd": 0.20284327086368154,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S1_P4_B1",
    "seed": 2,
    "R_nhv": 0.0001988142001984235,
    "R_igd": 0.016112779677186825,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S1_P4_B1",
    "seed": 3,
    "R_nhv": 0.0008634542918021703,
    "R_igd": 0.11943226459358883,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S1_P4_B1",
    "seed": 4,
    "R_nhv": 0.0009958749072584412,
    "R_igd": 0.095470416232208,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P1_B2",
    "seed": 0,
    "R_nhv": 0.004536520904416724,
    "R_igd": 0.32136949086484495,
    "R_makespan": 0.0,
    "R_shortage": 0.00731981981981982,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P1_B2",
    "seed": 1,
    "R_nhv": 0.006596881139797407,
    "R_igd": 0.060396267023135504,
    "R_makespan": 0.0,
    "R_shortage": 0.0004928536224741252,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P1_B2",
    "seed": 2,
    "R_nhv": 0.010310957154181037,
    "R_igd": 0.09639464053310412,
    "R_makespan": 0.0,
    "R_shortage": 0.03342776203966006,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P1_B2",
    "seed": 3,
    "R_nhv": 0.011420766808601412,
    "R_igd": 0.02639837307658809,
    "R_makespan": 0.0,
    "R_shortage": 0.04,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P1_B2",
    "seed": 4,
    "R_nhv": 0.001016532391422886,
    "R_igd": 0.0059302452121142895,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "seed": 0,
    "R_nhv": 0.016137851296087115,
    "R_igd": 0.10223461939207122,
    "R_makespan": 0.0,
    "R_shortage": 0.02772151898734177,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "seed": 1,
    "R_nhv": 0.005703249503924608,
    "R_igd": 0.09076407780637091,
    "R_makespan": 0.0,
    "R_shortage": 0.007792860734037205,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "seed": 2,
    "R_nhv": 0.004545331285649311,
    "R_igd": 0.027451823717434657,
    "R_makespan": 0.0,
    "R_shortage": 0.010829899941141849,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "seed": 3,
    "R_nhv": 0.025373401260030726,
    "R_igd": 0.036468844266440426,
    "R_makespan": 0.0,
    "R_shortage": 0.042421462967209354,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P2_B3",
    "seed": 4,
    "R_nhv": 0.0021220413535105614,
    "R_igd": 0.06005816218587695,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "seed": 0,
    "R_nhv": 0.0014162599888648858,
    "R_igd": 0.719115804471859,
    "R_makespan": 0.0,
    "R_shortage": 0.013761467889908258,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "seed": 1,
    "R_nhv": 0.038407138259212305,
    "R_igd": 0.04818874981094412,
    "R_makespan": 0.017846862406447898,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "seed": 2,
    "R_nhv": 0.001436853245091727,
    "R_igd": 0.03674615594903495,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "seed": 3,
    "R_nhv": 0.010001569537970887,
    "R_igd": 0.18099809407732956,
    "R_makespan": 0.003474232773595831,
    "R_shortage": 0.00894187779433681,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P3_B1",
    "seed": 4,
    "R_nhv": 0.010511475764658917,
    "R_igd": 0.057697551735763034,
    "R_makespan": 0.0011689070718877848,
    "R_shortage": 0.045112781954887216,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P4_B2",
    "seed": 0,
    "R_nhv": 0.00838316682438993,
    "R_igd": 0.034679755093431006,
    "R_makespan": 0.0,
    "R_shortage": 0.04914196567862714,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S4_P4_B2",
    "seed": 1,
    "R_nhv": 0.003650865432746743,
    "R_igd": 0.01277830033405201,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S4_P4_B2",
    "seed": 2,
    "R_nhv": 0.0005348827869740681,
    "R_igd": 0.025948875054251124,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S4_P4_B2",
    "seed": 3,
    "R_nhv": 0.0011162581217529467,
    "R_igd": 0.03607624967229984,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S4_P4_B2",
    "seed": 4,
    "R_nhv": 0.001398383678548403,
    "R_igd": 0.03866381092812254,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "seed": 0,
    "R_nhv": 0.0114910418406496,
    "R_igd": 0.0910715018209576,
    "R_makespan": 0.0,
    "R_shortage": 0.022485053288276578,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "seed": 1,
    "R_nhv": 0.007439106594584261,
    "R_igd": 0.07362472272722283,
    "R_makespan": 0.0,
    "R_shortage": 0.009767092411720512,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "seed": 2,
    "R_nhv": 0.0029580226782408322,
    "R_igd": 0.07159800210248023,
    "R_makespan": 0.00035298270384751147,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "seed": 3,
    "R_nhv": 0.007157306498488282,
    "R_igd": 0.1955996648082605,
    "R_makespan": 0.0,
    "R_shortage": 0.013805522208883553,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P1_B3",
    "seed": 4,
    "R_nhv": 0.002690825326276484,
    "R_igd": 0.03820375914309392,
    "R_makespan": 0.0,
    "R_shortage": 0.003990522509041028,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S8_P2_B1",
    "seed": 0,
    "R_nhv": 0.0007178525610714106,
    "R_igd": 0.007356970385361763,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S8_P2_B1",
    "seed": 1,
    "R_nhv": 0.004941987349523947,
    "R_igd": 0.09670654321630241,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P2_B1",
    "seed": 2,
    "R_nhv": 0.0299406611257976,
    "R_igd": 0.3494746320551917,
    "R_makespan": 0.0,
    "R_shortage": 0.03346761698171676,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P2_B1",
    "seed": 3,
    "R_nhv": 0.002126098387010867,
    "R_igd": 0.01409797299346253,
    "R_makespan": 0.0006499837504062399,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S8_P2_B1",
    "seed": 4,
    "R_nhv": 0.030037734092545242,
    "R_igd": 0.1646653603927603,
    "R_makespan": 0.010420058612829698,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P3_B2",
    "seed": 0,
    "R_nhv": 0.010641786171676453,
    "R_igd": 0.03138457753727052,
    "R_makespan": 0.0024204702627939143,
    "R_shortage": 0.0026771875440326897,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P3_B2",
    "seed": 1,
    "R_nhv": 0.0005104065273461655,
    "R_igd": 0.0021013005867376714,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S8_P3_B2",
    "seed": 2,
    "R_nhv": 0.010702623630707373,
    "R_igd": 0.03208912629199083,
    "R_makespan": 0.0,
    "R_shortage": 0.009762711864406779,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P3_B2",
    "seed": 3,
    "R_nhv": 0.0,
    "R_igd": 0.0,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S8_P3_B2",
    "seed": 4,
    "R_nhv": 0.003866741824019734,
    "R_igd": 0.07284909052188818,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "seed": 0,
    "R_nhv": 0.005928117841168693,
    "R_igd": 0.1336457104645493,
    "R_makespan": 0.0,
    "R_shortage": 0.007431874483897605,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "seed": 1,
    "R_nhv": 0.0019475365555474728,
    "R_igd": 0.049416894245406656,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": true
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "seed": 2,
    "R_nhv": 0.004677529337506208,
    "R_igd": 0.14930991577344888,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "seed": 3,
    "R_nhv": 0.0007953565341054942,
    "R_igd": 0.05121987400876639,
    "R_makespan": 0.0,
    "R_shortage": 0.0,
    "pass": false
  },
  {
    "instance": "WIPHFSP_S8_P4_B3",
    "seed": 4,
    "R_nhv": 0.00801729116211723,
    "R_igd": 0.33093545560454396,
    "R_makespan": 0.0016072002571520412,
    "R_shortage": 0.007712082262210797,
    "pass": false
  }
]

## F. Interpretation boundaries
Instances, not seeds, are the independent benchmark units. Size groups contain four instances each.
Mean/SD and coverage are descriptive; no global optimality or behavior beyond 60000 is established.
No mechanisms, parameters or instance inclusion rules were adapted after viewing outcomes.
Any observed plateau is conditional on fixed parameters and the sampled seeds/instances.
Source hashes before/after are in fe_recommendation.json; wall time is in run_manifest.json.
