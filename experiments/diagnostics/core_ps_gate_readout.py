"""Read-only descriptive exports after the 72-run gate; no additional evaluations.

Run: PYTHONHASHSEED=0 python3 -m experiments.diagnostics.core_ps_gate_readout
Never changes the preregistered classifier, normalization or any run source.
"""

import json
import statistics
from collections import Counter, defaultdict

from experiments import run_core_ps_integration_gate as g


def average(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def export():
    fingerprint,payload=g.source_registration()
    summary=json.loads((g.ROOT/"analysis/core_ps_gate_summary.json").read_text())
    if summary["fingerprint"]!=fingerprint or summary["formal_runs"]!=72:
        raise AssertionError("Complete source-matched gate required")
    records=[]
    for instance in g.PRIMARY+g.SECONDARY:
        for variant in g.VARIANTS:
            for seed in g.SEEDS:
                record=g.load_run(g.r.Task(instance,variant,seed),fingerprint)
                if record is None:
                    raise AssertionError("Never read out an incomplete gate")
                records.append(record)
    details=[]; origins=[]; objectives={}
    for record in records:
        events=record["events"]
        entry={k:record[k] for k in ("instance","variant","seed","set")}
        for label,selected in (("H_only",[e for e in events if e["restricted"]]),
                               ("specialized_branch",[e for e in events if e["branch"]=="specialized"]),
                               ("mutation_branch",[e for e in events if e["branch"]=="mutation"])):
            n=len(selected)
            entry.update({label+"_generated":n,
                          label+"_accepted":sum(e["accepted"] for e in selected),
                          label+"_archive_entries":sum(e["archive_entry"] for e in selected),
                          label+"_acceptance_rate":average(e["accepted"] for e in selected),
                          label+"_archive_entry_rate":average(e["archive_entry"] for e in selected),
                          label+"_parent_gain_mean":average(e["parent_gain"] for e in selected),
                          label+"_accepted_parent_gain_sum":sum(e["parent_gain"] for e in selected if e["accepted"]),
                          label+"_accepted_replacement_gain_sum":sum(e["replacement_gain"] for e in selected),
                          label+"_parent_improvement_rate":average(e["parent_gain"]>0 for e in selected),
                          label+"_OS":sum(e["move_type"]=="OS" for e in selected),
                          label+"_MS":sum(e["move_type"]=="MS" for e in selected)})
        high=[e for e in events if e.get("H_size") is not None]
        entry.update(H_mean=average(e["H_size"] for e in high),
                     H_median=statistics.median(e["H_size"] for e in high) if high else None,
                     H_ratio_mean=average(e["H_ratio"] for e in high),
                     H_mass_mean=average(e["H_mass"] for e in high),
                     H_coverage_mean=average(e["H_coverage"] for e in high),
                     parent_shortage_mean=average(e["S_parent"] for e in events),
                     fallback_count=sum(bool(e["fallback_reason"]) for e in events),
                     fallback_fraction=average(bool(e["fallback_reason"]) for e in events),
                     H_only_MS_fraction=entry["H_only_MS"]/entry["H_only_generated"] if entry["H_only_generated"] else None,
                     wall_seconds=record["runtime_seconds"],**record["runtime"])
        details.append(entry)
        objectives[f"{record['instance']}/{record['variant']}/seed{record['seed']}"]=g.r.nondominated_filter(g.r.validated_points(record["final_archive"]))
        counts=Counter(point["origin"] for point in record["final_archive"])
        origins.extend({**{k:record[k] for k in ("instance","variant","seed","set")},
                        "origin":origin,"genotypes":count,"fraction":count/len(record["final_archive"])}
                       for origin,count in sorted(counts.items()))
    g.save("mechanism_details.csv",details)
    g.save("archive_origin_contribution.csv",origins)
    g.r.write_json_atomic(g.ROOT/"analysis/final_nondominated_objective_sets.json",objectives)
    groups=defaultdict(list)
    for row in details:
        groups[(row["instance"],row["variant"])].append(row)
    means=[]
    numeric=[k for k in details[0] if k not in ("instance","variant","seed","set")]
    for (instance,variant),values in sorted(groups.items()):
        means.append({"instance":instance,"variant":variant,"set":values[0]["set"],
                      **{key:average(z[key] for z in values) for key in numeric}})
    g.save("mechanism_instance_mean.csv",means)
    metrics=g.r.build_run_metrics({g.r.Task(z["instance"],z["variant"],z["seed"]):z for z in records})
    paired=g.instance_pairs(g.pair_metrics(metrics))
    index={(z["instance"],z["metric"]):z["benefit"] for z in paired}
    localization=[{**z,**{"benefit_"+metric:index[(z["instance"],metric)] for metric in g.METRICS}}
                  for z in means if z["variant"]=="core_ps"]
    g.save("gain_localization.csv",localization)
    correlations=[]
    for feature in ("H_ratio_mean","H_only_MS_fraction","parent_shortage_mean","fallback_fraction"):
        for metric in g.METRICS:
            selected=[z for z in localization if z["set"]=="primary_unseen" and z[feature] is not None]
            correlations.append({"set":"primary_unseen","instances":len(selected),"feature":feature,
                                 "metric":metric,"spearman_descriptive":g.r.sis_spearman(
                                     [z[feature] for z in selected],[z["benefit_"+metric] for z in selected]),
                                 "warning":"Post-run descriptive association; divergent parents, not causal and not a tuning rule"})
    g.save("descriptive_associations.csv",correlations)
    runtime=[]
    by={(z["instance"],z["variant"]):z for z in means}
    for instance in g.PRIMARY+g.SECONDARY:
        a,b=by[(instance,"basic_ps")],by[(instance,"core_ps")]
        runtime.append({"instance":instance,"set":a["set"],"basic_wall_mean":a["wall_seconds"],
                        "core_wall_mean":b["wall_seconds"],"core_basic_wall_ratio":b["wall_seconds"]/a["wall_seconds"],
                        "core_diagnosis_mean":b["diagnosis_seconds"],"core_H_mean":b["H_seconds"]})
    g.save("runtime_instance_pairs.csv",runtime)
    observed={variant:{field:average(z[field] for z in details if z["variant"]==variant and z["set"]=="primary_unseen")
                       for field in numeric} for variant in g.VARIANTS}
    g.r.write_json_atomic(g.ROOT/"analysis/readout_details.json",{
        "fingerprint":fingerprint,"additional_FE":0,"classification_unchanged":summary["classification"],
        "primary_mechanism_run_means":observed,"descriptive_associations":correlations,
        "accepted_gain_definition":"parent gain can be negative for accepted offspring; nearest-incumbent replacement gain is strictly positive when accepted",
        "archive_contribution":"First evaluated genotype origin, not proof of causal contribution; objective sets separately deduplicated"})
    g.check_sources(payload)
    print("Core-PS readout complete: no new decode, no classifier change",flush=True)
    return summary,observed,localization


if __name__=="__main__":
    export()
