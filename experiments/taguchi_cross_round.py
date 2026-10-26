"""Saved-round validation and offline common-scale comparison for Taguchi runs.

The runner supplies its existing pure metric functions. This module never imports
an optimizer/decoder and never writes inside an input round.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import platform
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


class RoundValidationError(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or []


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError(f"Unsafe directory name: {name!r}")
    return name


def results_base(api):
    relative = Path(api.RESULTS_DIR)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RESULTS_DIR must be project-relative")
    base = (api.ROOT/relative).resolve()
    allowed = (api.ROOT/"experiments/results/taguchi_tuning").resolve()
    if base != allowed and allowed not in base.parents:
        raise ValueError("Taguchi results must remain under the dedicated results directory")
    return base


def validate_saved_config(api, config, name):
    """Validate an old design against its OWN levels, never current globals."""
    if config["schema"] != api.SCHEMA or config["round"] != name or name == "cross_round_analysis":
        raise ValueError("Round/schema identity mismatch")
    if type(config["FE_max"]) is not int or config["FE_max"] < 1 or config["archive_ratio"] != 2:
        raise ValueError("Invalid saved budget/archive rule")
    for label in ("seeds", "instances"):
        values = config[label]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"Invalid saved {label}")
    if any(type(seed) is not int for seed in config["seeds"]):
        raise ValueError("Saved seed IDs must be integers")
    for name in config["instances"]:
        api.dimensions(name)
    levels = config["factor_levels"]
    if set(levels) != set(api.FACTORS):
        raise ValueError("Saved design must have exactly the four formal factors")
    for factor in api.FACTORS:
        values = levels[factor]
        if (len(values) != 4 or len(set(values)) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise ValueError(f"Invalid saved levels: {factor}")
        if factor in ("N", "T_coop") and any(type(v) is not int or v < 1 for v in values):
            raise ValueError("Invalid saved N/T_coop")
    if not all(0 <= v <= 1 for v in levels["p_mut"]) or not all(0 < v <= 1 for v in levels["rho"]):
        raise ValueError("Invalid saved mutation/coverage levels")
    if config["FE_max"] < 2*max(levels["N"]):
        raise ValueError("Saved budget cannot cover initialization")
    array = config["orthogonal_array"]
    if len(array) != 16 or {r["run_id"] for r in array} != {f"{i:02d}" for i in range(1, 17)}:
        raise ValueError("Saved L16 run IDs invalid")
    api.validate_orthogonality([tuple(r[l+"_level"] for l in api.LETTERS) for r in array])
    for row in array:
        for factor, letter in zip(api.FACTORS, api.LETTERS):
            if row[factor] != levels[factor][row[letter+"_level"]-1]:
                raise ValueError("Saved OA level-to-parameter mapping mismatch")
        if row["N_A"] != 2*row["N"]:
            raise ValueError("Saved N_A must equal 2N")
    if len({tuple(r[k] for k in api.FACTORS) for r in array}) != len(array):
        raise ValueError("Duplicate measured parameter combinations in one design")
    if (config["hv_reference_point"] != [1.1, 1.1] or config["hv_normalization_factor"] != 1.21
            or config["fair_archive_limit"] != 40):
        raise ValueError("Unsupported metric reference/NHV divisor/fair archive definition")
    if set(config["instance_hashes"]) != set(config["instances"]):
        raise ValueError("Missing saved instance identities")
    sources = config["source_identity"]["sources"]
    if not set(api.SOURCES) <= set(sources):
        raise ValueError("Missing core algorithm/encoder/decoder source identities")
    for value in list(config["instance_hashes"].values()) + [sources[k] for k in api.SOURCES]:
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("Invalid SHA256 source identifier")


def input_digest(api, root, tasks):
    h = hashlib.sha256()
    paths = [root/name for name in ("config.json", "orthogonal_array.csv", "manifest.json")]
    paths += [api.raw_path(root, task) for task in tasks]
    for path in sorted(paths):
        h.update(str(path.relative_to(root)).encode())
        h.update(bytes.fromhex(api.file_sha(path)))
    return h.hexdigest()


def load_saved_round(api, base, name):
    """Read/validate only; caller decides where any derived files may be written."""
    safe_name(name)
    if name == "cross_round_analysis":
        raise ValueError("Reserved input round name")
    root = (base/name).resolve()
    if root.parent != base.resolve():
        raise ValueError("Input round escapes results directory")
    if (root/".session.lock").exists():
        raise RoundValidationError(f"{name}: input round is locked; wait for its writer or verify a stale lock")
    try:
        stored = json.loads((root/"config.json").read_text())
        config = stored["experiment"]
        validate_saved_config(api, config, name)
        if stored["experiment_fingerprint"] != api.digest(config):
            raise ValueError("Saved configuration fingerprint mismatch")
        with (root/"orthogonal_array.csv").open(encoding="utf-8-sig", newline="") as handle:
            array = list(csv.DictReader(handle))
        if array != [{k: str(v) for k, v in r.items()} for r in config["orthogonal_array"]]:
            raise ValueError("Saved CSV orthogonal table differs from saved configuration")
        tasks = api.build_tasks(config)
        records, errors = api.scan_records(root, tasks, config)
        if errors:
            raise RoundValidationError(f"{name}: {len(errors)}/{len(tasks)} missing/invalid tasks", errors)
        manifest = json.loads((root/"manifest.json").read_text())
        if (manifest.get("experiment_fingerprint") != stored["experiment_fingerprint"] or
                manifest.get("expected") != len(tasks) or manifest.get("valid_completed") != len(tasks) or
                manifest.get("complete") is not True or
                manifest.get("actual_FE_completed") != len(tasks)*config["FE_max"]):
            raise ValueError("Manifest does not confirm all expected source-valid completed tasks")
        return {"name": name, "root": root, "config": config, "tasks": tasks, "records": records,
                "manifest": manifest, "input_sha256": input_digest(api, root, tasks),
                "config_sha256": api.file_sha(root/"config.json")}
    except RoundValidationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise RoundValidationError(f"{name}: {exc}") from exc


def validate_common_conditions(api, rounds):
    first = rounds[0]["config"]
    for item in rounds[1:]:
        config = item["config"]
        for key in ("FE_max", "archive_ratio", "hv_reference_point", "hv_normalization_factor", "fair_archive_limit", "schema"):
            if config[key] != first[key]:
                raise ValueError(f"Rounds differ in required common condition: {key}")
        if set(config["instances"]) != set(first["instances"]):
            raise ValueError("Rounds have different instance sets; default cross-round ranking refused")
        if config["instance_hashes"] != first["instance_hashes"]:
            raise ValueError("Problem instance contents differ between rounds")
        for path in api.SOURCES:
            if config["source_identity"]["sources"][path] != first["source_identity"]["sources"][path]:
                raise ValueError(f"Core implementation differs between rounds: {path}")
    return {"objectives": ["makespan", "shortage"], "directions": ["min", "min"],
            "normalization": "per-instance union of every raw archive across compared rounds",
            "gd": "sqrt(sum(min_distance_squared))/archive_set_size",
            "igd": "mean_reference_nearest_distance", "hv_reference": [1.1, 1.1], "nhv_divisor": 1.21,
            "fair_archive_limit": 40,
            "identical_seed_sets": all(set(r["config"]["seeds"]) == set(first["seeds"]) for r in rounds),
            "seed_sets": {r["name"]: r["config"]["seeds"] for r in rounds},
            "python_versions": {r["name"]: r["config"]["source_identity"].get("python") for r in rounds},
            "runner_hash_policy": "Each raw record matches its own stored config; whole-runner hashes need not match across rounds.",
            "metric_contract": "hlns-taguchi-l16-v1 documented formulas, recomputed by one shared metric implementation"}


def measured_summary(api, runs, rounds):
    instances = api.summarize(runs, ("round_name", "run_id", "instance", "size", "profile", "buffer"))
    observations = [{**{k: row[k] for k in ("round_name", "run_id", "instance", "size", "profile", "buffer")},
                     **{k: row["mean_"+k] for k in api.METRICS}} for row in instances]
    combinations = api.summarize(observations, ("round_name", "run_id"))
    params = {(item["name"], r["run_id"]): {k: r[k] for k in (*api.FACTORS, "N_A")}
              for item in rounds for r in item["config"]["orthogonal_array"]}
    for row in instances:
        row.update(params[row["round_name"], row["run_id"]])
        row["n_seeds"] = row["n_observations"]
    for row in combinations:
        row.update(params[row["round_name"], row["run_id"]])
        row["mean_runtime"] = row["mean_runtime_seconds"]
        row["rank_global"] = 1+sum(r["mean_nhv"] > row["mean_nhv"] for r in combinations)
        row["rank_within_round"] = 1+sum(r["round_name"] == row["round_name"] and r["mean_nhv"] > row["mean_nhv"] for r in combinations)
        row["rank_global_nhv40"] = 1+sum(r["mean_nhv40"] > row["mean_nhv40"] for r in combinations)
    return instances, sorted(combinations, key=lambda r: (r["rank_global"], r["round_name"], r["run_id"]))


def choose_pairs(api, combinations, rounds):
    # Stable representative for exact metric ties; all tied top rows remain in tables.
    best = {item["name"]: min((r for r in combinations if r["round_name"] == item["name"]),
                              key=lambda r: (-r["mean_nhv"], r["run_id"])) for item in rounds}
    pairs = [((left["name"], best[left["name"]]["run_id"]), (right["name"], best[right["name"]]["run_id"]))
             for left, right in itertools.combinations(rounds, 2)]
    pairs.extend(api.COMPARE_PAIRS)
    measured = {(r["round_name"], r["run_id"]) for r in combinations}
    checked = []
    for pair in pairs:
        if len(pair) != 2 or any(len(key) != 2 for key in pair):
            raise ValueError("COMPARE_PAIRS requires ((baseline_round, run_id), (candidate_round, run_id))")
        left, right = map(tuple, pair)
        if left not in measured or right not in measured or left == right:
            raise ValueError(f"Comparison must name two measured configurations: {pair}")
        if (left, right) not in checked:
            checked.append((left, right))
    return best, checked


def round_factor_effects(api, instances, rounds):
    """Separate OA main effects per round, evaluated on the common scale."""
    result = []
    for item in rounds:
        rows = [r for r in instances if r["round_name"] == item["name"]]
        for factor, letter in zip(api.FACTORS, api.LETTERS):
            for index, value in enumerate(item["config"]["factor_levels"][factor], 1):
                ids = {r["run_id"] for r in item["config"]["orthogonal_array"] if r[letter+"_level"] == index}
                selected = [r for r in rows if r["run_id"] in ids]
                effect = {"round_name": item["name"], "factor": factor, "level_index": index,
                          "level_value": value, "mean_nhv": st.mean(r["mean_nhv"] for r in selected)}
                for size in sorted({r["size"] for r in rows}):
                    effect["mean_nhv_"+size] = st.mean(r["mean_nhv"] for r in selected if r["size"] == size)
                result.append(effect)
    return result


def paired_comparisons(api, runs, instances, pairs, rounds):
    lookup = {(r["round_name"], r["run_id"], r["instance"], r["seed"]): r for r in runs}
    ilookup = {(r["round_name"], r["run_id"], r["instance"]): r for r in instances}
    configs = {r["name"]: r["config"] for r in rounds}
    seed_rows, instance_rows, summaries = [], [], []
    for idx, (left, right) in enumerate(pairs, 1):
        label = f"pair_{idx:02d}"
        common = sorted(set(configs[left[0]]["seeds"]) & set(configs[right[0]]["seeds"]))
        fully_paired = set(configs[left[0]]["seeds"]) == set(configs[right[0]]["seeds"])
        header = {"pair_id": label, "baseline_round": left[0], "baseline_run_id": left[1],
                  "candidate_round": right[0], "candidate_run_id": right[1],
                  "seed_pairing_available": bool(common), "common_seed_count": len(common),
                  "full_seed_pairing": fully_paired,
                  "comparison_unit": "instance_mean_using_each_round_own_seeds"}
        for name in configs[left[0]]["instances"]:
            a, b = ilookup[left+(name,)], ilookup[right+(name,)]
            row = {**header, "instance": name, **api.dimensions(name),
                   "baseline_seed_count": len(configs[left[0]]["seeds"]),
                   "candidate_seed_count": len(configs[right[0]]["seeds"]), "common_seed_count": len(common)}
            for metric in api.METRICS:
                row["baseline_"+metric] = a["mean_"+metric]
                row["candidate_"+metric] = b["mean_"+metric]
                row["delta_"+metric] = b["mean_"+metric]-a["mean_"+metric]
            instance_rows.append(row)
            for seed in common:
                x, y = lookup[left+(name, seed)], lookup[right+(name, seed)]
                seed_rows.append({**header, "instance": name, "seed": seed, **api.dimensions(name),
                                  **{"delta_"+k: y[k]-x[k] for k in api.METRICS}})
        group = [r for r in instance_rows if r["pair_id"] == label]
        d = [r["delta_nhv"] for r in group]
        summary = {**header, "n_instances": len(group), "paired_seed_records": len(group)*len(common),
                   "mean_delta_nhv": st.mean(d), "median_delta_nhv": st.median(d),
                   "wins": sum(v > api.EPS for v in d), "ties": sum(abs(v) <= api.EPS for v in d),
                   "losses": sum(v < -api.EPS for v in d),
                   "loo_min_mean_delta_nhv": min(st.mean(d[:j]+d[j+1:]) for j in range(len(d))) if len(d)>1 else None,
                   "loo_max_mean_delta_nhv": max(st.mean(d[:j]+d[j+1:]) for j in range(len(d))) if len(d)>1 else None}
        for metric in api.METRICS:
            for prefix in ("baseline_", "candidate_", "delta_"):
                summary["mean_"+prefix+metric] = st.mean(r[prefix+metric] for r in group)
        summaries.append(summary)
    grouped = {}
    for dimension in ("size", "profile", "buffer"):
        groups = defaultdict(list)
        for row in instance_rows:
            groups[row["pair_id"], row[dimension]].append(row)
        values = []
        for (pair_id, group_name), group in sorted(groups.items()):
            r = {k: group[0][k] for k in ("pair_id", "baseline_round", "baseline_run_id", "candidate_round", "candidate_run_id",
                                         "seed_pairing_available", "common_seed_count", "full_seed_pairing", "comparison_unit")}
            r.update({dimension: group_name, "n_instances": len(group),
                      "wins": sum(g["delta_nhv"]>api.EPS for g in group),
                      "ties": sum(abs(g["delta_nhv"])<=api.EPS for g in group),
                      "losses": sum(g["delta_nhv"]< -api.EPS for g in group)})
            for metric in api.METRICS:
                for prefix in ("baseline_", "candidate_", "delta_"):
                    r["mean_"+prefix+metric] = st.mean(g[prefix+metric] for g in group)
            values.append(r)
        grouped[dimension] = values
    return seed_rows, instance_rows, summaries, grouped


def compare_mode(api):
    """Validate every input before emitting any ranking, metrics or figures."""
    names = api.COMPARE_ROUNDS
    if not isinstance(names, (list, tuple)) or len(names)<2 or len(set(names))!=len(names):
        raise ValueError("COMPARE_ROUNDS requires at least two unique round names")
    base = results_base(api)
    output = (base/"cross_round_analysis"/safe_name(api.COMPARE_OUTPUT_NAME)).resolve()
    if output.parent != (base/"cross_round_analysis").resolve() or base not in output.parents:
        raise ValueError("Cross-round output escapes its dedicated directory")
    rounds, errors = [], []
    for name in names:
        try:
            rounds.append(load_saved_round(api, base, name))
        except (RoundValidationError, ValueError) as exc:
            errors.append({"round": name, "error": str(exc), "details": getattr(exc, "details", [])})
    common = None
    if not errors:
        try:
            common = validate_common_conditions(api, rounds)
        except ValueError as exc:
            errors.append({"common_conditions": str(exc)})
    validation = {"valid": not errors, "errors": errors, "rounds": names,
                  "inputs": [{"round": r["name"], "expected": len(r["tasks"]), "valid_completed": len(r["records"]),
                              "instances": len(r["config"]["instances"]), "seeds": len(r["config"]["seeds"]),
                              "FE_max": r["config"]["FE_max"], "input_sha256": r["input_sha256"],
                              "config_sha256": r["config_sha256"], "stored_source_identity": r["config"]["source_identity"]} for r in rounds],
                  "common_conditions": common, "checked_utc": api.utc_now()}
    output.mkdir(parents=True, exist_ok=True)
    api.atomic_json(output/"validation_report.json", validation)
    if errors:
        for error in errors:
            print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        raise RoundValidationError(f"Cross-round comparison refused; see {output/'validation_report.json'}. No optimization or comparison report generated.", errors)
    pool = defaultdict(list)
    for item in rounds:
        for record in item["records"].values():
            pool[record["instance"]].extend(record["final_archive_objectives"])
    bounds_json, references, bases = {}, {}, {}
    for name, points in pool.items():
        bounds = [(min(p[j] for p in points), max(p[j] for p in points)) for j in (0, 1)]
        front = api.nondominated(points)
        normalized_reference = [api.normalize(p, bounds) for p in front]
        bounds_json[name] = {"makespan": bounds[0], "shortage": bounds[1],
                             "degenerate_objectives": [k for k, (lo, hi) in zip(("makespan", "shortage"), bounds) if lo==hi],
                             "rounds": list(names)}
        references[name] = {"raw": front, "normalized": normalized_reference, "rounds": list(names)}
        bases[name] = bounds, normalized_reference
    runs = []
    for item in rounds:
        config = item["config"]
        for task in item["tasks"]:
            r = item["records"][task["run_id"], task["instance"], task["seed"]]
            bounds, reference = bases[task["instance"]]
            front = api.nondominated([api.normalize(p, bounds) for p in r["final_archive_objectives"]])
            fair = api.fair_truncate(front, config["fair_archive_limit"])
            hv = api.compute_hv(front, config["hv_reference_point"])
            runs.append({"round_name": item["name"], **task, **api.dimensions(task["instance"]),
                         "raw_hv": hv, "nhv": hv/config["hv_normalization_factor"],
                         "nhv40": api.compute_hv(fair, config["hv_reference_point"])/config["hv_normalization_factor"],
                         "gd": api.compute_gd(front, reference), "igd": api.compute_igd(front, reference),
                         "best_makespan": min(p[0] for p in r["final_archive_objectives"]),
                         "best_shortage": min(p[1] for p in r["final_archive_objectives"]),
                         "archive_size": r["final_archive_size"], "unique_objective_count": len(front),
                         "fair_archive_size": len(fair), "runtime_seconds": r["runtime_seconds"], "actual_FE": r["actual_FE"]})
    instances, combinations = measured_summary(api, runs, rounds)
    best, pairs = choose_pairs(api, combinations, rounds)
    effects = round_factor_effects(api, instances, rounds)
    seeds, differences, summaries, grouped = paired_comparisons(api, runs, instances, pairs, rounds)
    if any(input_digest(api, r["root"], r["tasks"]) != r["input_sha256"] for r in rounds):
        validation.update(valid=False, errors=[{"error": "Input data changed during offline evaluation; retry after stopping writers"}])
        api.atomic_json(output/"validation_report.json", validation)
        raise RoundValidationError("Input data changed; refusing derived ranking")
    config_out = {"rounds": names, "comparison_pairs": pairs, "best_measured_rows": best,
                  "metric_contract": common, "analysis_python": platform.python_version(),
                  "analysis_runner_sha256": api.file_sha(api.__file__), "analysis_helper_sha256": api.file_sha(__file__),
                  "input_fingerprints": {r["name"]: api.digest(r["config"]) for r in rounds},
                  "total_valid_runs": len(runs), "created_utc": api.utc_now()}
    api.atomic_json(output/"config.json", config_out)
    api.atomic_json(output/"normalization_bounds.json", bounds_json)
    api.atomic_json(output/"reference_fronts.json", references)
    for name, rows in (("cross_round_run_metrics", runs), ("cross_round_instance_metrics", instances),
                       ("cross_round_combination_summary", combinations), ("best_combination_comparison", summaries),
                       ("paired_seed_differences", seeds), ("instance_nhv_differences", differences),
                       ("round_best_combinations", [best[name] for name in names]),
                       ("pairwise_round_comparisons", summaries), ("paired_instance_differences", differences),
                       ("seed_pairing_summary", [{k: r[k] for k in
                            ("pair_id", "baseline_round", "baseline_run_id", "candidate_round", "candidate_run_id",
                             "seed_pairing_available", "common_seed_count", "full_seed_pairing", "paired_seed_records", "comparison_unit")}
                            for r in summaries]),
                       ("round_factor_effects_on_common_scale", effects)):
        if rows:
            api.save_csv(output/(name+".csv"), rows)
        else:
            # No overlapping seeds: record an empty table, never invent pairings.
            (output/(name+".csv")).write_text("pair_id,instance,seed,seed_pairing_available,common_seed_count,delta_nhv\n", encoding="utf-8")
    for field, filename in (("size", "size_group_comparison"), ("profile", "profile_group_comparison"), ("buffer", "buffer_group_comparison")):
        api.save_csv(output/(filename+".csv"), grouped[field])
    draw_comparison_figures(output/"figures", combinations, differences, instances, best, names)
    write_cross_report(api, output/"cross_round_report.md", validation, combinations, best, summaries, grouped, runs, effects, rounds, differences)
    print(f"Offline comparison complete: {output}")


def draw_comparison_figures(directory, combinations, differences, instances, best, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    directory.mkdir(parents=True, exist_ok=True)
    palette = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(names)}
    fig, ax = plt.subplots(figsize=(max(12, len(combinations)*.28), 7))
    labels = [r["round_name"]+":"+r["run_id"] for r in combinations]
    ax.bar(range(len(labels)), [r["mean_nhv"] for r in combinations], color=[palette[r["round_name"]] for r in combinations])
    ax.annotate("Best measured: "+labels[0], (0, combinations[0]["mean_nhv"]),
                xytext=(12, 8), textcoords="offset points", fontsize=9)
    ax.legend(handles=[Patch(color=palette[name], label=name) for name in names], frameon=False)
    ax.set_xticks(range(len(labels)), labels, rotation=90); ax.set_ylabel("Mean NHV (common per-instance scale)")
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory/f"cross_round_nhv_ranking.{ext}", dpi=300)
    plt.close(fig)
    sizes = sorted({r["size"] for r in instances})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = list(range(len(sizes)))
    width = .8/len(names)
    for i, name in enumerate(names):
        rid = best[name]["run_id"]
        values = [st.mean(r["mean_nhv"] for r in instances if r["round_name"] == name
                          and r["run_id"] == rid and r["size"] == size) for size in sizes]
        ax.bar([x+(i-(len(names)-1)/2)*width for x in xs], values, width,
               label=name+":"+rid, color=palette[name])
    ax.set_xticks(xs, sizes); ax.set_ylabel("Mean NHV (common scale)"); ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory/f"round_best_by_size.{ext}", dpi=300)
    plt.close(fig)
    used = set()
    for pair_id in sorted({r["pair_id"] for r in differences}):
        group = [r for r in differences if r["pair_id"] == pair_id]
        first = group[0]
        left = first["baseline_round"]+":"+first["baseline_run_id"]
        right = first["candidate_round"]+":"+first["candidate_run_id"]
        filename = first["candidate_round"]+"_vs_"+first["baseline_round"]+"_instance_differences"
        if filename in used:
            filename += "_"+pair_id
        used.add(filename)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(group)), [r["delta_nhv"] for r in group],
               color=["#24736b" if r["delta_nhv"]>=0 else "#b34747" for r in group])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(len(group)), [r["instance"] for r in group], rotation=65, ha="right")
        ax.set_ylabel("Delta NHV: candidate - baseline")
        ax.set_title(right+" minus "+left+"; instance seed means")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(directory/f"{filename}.{ext}", dpi=300)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(names)), [best[name]["mean_nhv"] for name in names], color=[palette[name] for name in names])
    ax.set_xticks(range(len(names)), [name+":"+best[name]["run_id"] for name in names])
    ax.set_ylabel("Best measured Mean NHV (common scale)")
    ax.set_title("Parameter-selection results, NOT independent confirmation", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory/f"best_nhv_by_round.{ext}", dpi=300)
    plt.close(fig)


def write_cross_report(api, path, validation, combinations, best, summaries, grouped, runs, effects, rounds, differences):
    lines = ["# 跨轮田口实测配置统一评价报告", "", "## 数据完整性"]
    for r in validation["inputs"]:
        lines.append(f"- {r['round']}：{r['expected']}条预期任务，{r['valid_completed']}条有效，"
                     f"{r['instances']}实例、{r['seeds']}种子，每次{r['FE_max']} FE；来源摘要{r['input_sha256']}。")
    for item in rounds:
        lines.append(f"- {item['name']}独立L16水平：{item['config']['factor_levels']}；种子：{item['config']['seeds']}。")
    total_runs = sum(r["valid_completed"] for r in validation["inputs"])
    total_fe = sum(r["valid_completed"]*r["FE_max"] for r in validation["inputs"])
    lines.append(f"共{len(rounds)}轮、{total_runs}次有效运行、{len(combinations)}个实测配置记录、{total_fe} FE。")
    lines += ["原始结果逐轮按各自保存配置验证，不用当前轮的参数水平验证旧轮；核心算法/编码器/解码器及实例哈希跨轮一致。",
              "实验脚本自身版本允许不同；这不等于忽略原始任务各自的指纹校验。Python版本等来源元数据见validation_report.json。",
              "", "## 统一评价口径",
              "每实例汇总所有参与轮次的全部原始目标档案，固定共同归一化界限与经验非支配参考前沿。",
              "HV参考点(1.1,1.1)，NHV=HV/1.21；GD=sqrt(sum(d²))/|A|，IGD为参考点最近距离均值。",
              "先种子等权、再实例等权。NHV/HV越大越好；GD/IGD越小越好。raw_hv仍是归一化目标空间面积。",
              "退化维度归一化为0并记录；经验参考前沿不是数学最优前沿。不同轮原报告NHV不可直接相减，本报告也不覆盖旧报告。",
              "", "## 全部实测配置统一排名",
              "|全局名次|轮次/行|N/N_A|p_mut|rho|T_coop|NHV|NHV_40|GD|IGD|平均秒|",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in combinations:
        lines.append(f"|{r['rank_global']}|{r['round_name']}/{r['run_id']}|{r['N']}/{r['N_A']}|{r['p_mut']}|{r['rho']}|{r['T_coop']}|"
                     f"{r['mean_nhv']:.6f}|{r['mean_nhv40']:.6f}|{r['mean_gd']:.6f}|{r['mean_igd']:.6f}|{r['mean_runtime']:.2f}|")
    winner = combinations[0]
    lines += [f"统一尺度全局最佳实测配置：{winner['round_name']}/{winner['run_id']}；并列时按轮次名和行号稳定展示，排名保留并列。",
              "", "## 各轮最佳实测行及配对差异"]
    for name, r in best.items():
        lines.append(f"- {name}：行{r['run_id']}，参数N={r['N']}、N_A={r['N_A']}、p_mut={r['p_mut']}、rho={r['rho']}、T_coop={r['T_coop']}，"
                     f"NHV={r['mean_nhv']:.6f}、NHV_40={r['mean_nhv40']:.6f}、GD={r['mean_gd']:.6f}、IGD={r['mean_igd']:.6f}，平均耗时{r['mean_runtime']:.2f}秒。")
    for r in summaries:
        lines.append(f"- {r['pair_id']}：{r['candidate_round']}/{r['candidate_run_id']}减{r['baseline_round']}/{r['baseline_run_id']}，"
                     f"实例平均ΔNHV={r['mean_delta_nhv']:.6f}，中位数={r['median_delta_nhv']:.6f}，胜/平/负={r['wins']}/{r['ties']}/{r['losses']}，"
                     f"留一实例平均差范围=[{r['loo_min_mean_delta_nhv']},{r['loo_max_mean_delta_nhv']}]；完整种子配对={r['full_seed_pairing']}，"
                     f"seed_pairing_available={r['seed_pairing_available']}，公共种子数={r['common_seed_count']}，"
                     f"实际共有{r['paired_seed_records']}条共同种子配对。")
        if not r["seed_pairing_available"]:
            lines.append("  本比较无公共种子：只进行同实例各自种子均值的配对，不按种子编号或列表下标伪造运行级配对。")
        lines.append(f"  平均ΔNHV_40={r['mean_delta_nhv40']:.6f}、ΔGD={r['mean_delta_gd']:.6f}、ΔIGD={r['mean_delta_igd']:.6f}；"
                     f"平均耗时差={r['mean_delta_runtime_seconds']:.2f}秒。")
        failures = [r2["instance"] for r2 in differences if r2["pair_id"] == r["pair_id"] and r2["delta_nhv"] < -api.EPS]
        lines.append(f"  NHV退化实例：{failures}；总体NHV与IGD方向"
                     + ("一致。" if r['mean_delta_nhv']*r['mean_delta_igd'] < 0 else "不构成严格同向改善，需分别解读。"))
        subset = [row for row in differences if row["pair_id"] == r["pair_id"]]
        for label, selected in (("P2", [row for row in subset if row["profile"] == "P2"]),
                                ("非P2", [row for row in subset if row["profile"] != "P2"]),
                                ("S8", [row for row in subset if row["size"] == "S8"])):
            if selected:
                delta = st.mean(row["delta_nhv"] for row in selected)
                wins = sum(row["delta_nhv"] > api.EPS for row in selected)
                losses = sum(row["delta_nhv"] < -api.EPS for row in selected)
                lines.append(f"  {label}边界检查：{len(selected)}实例，平均ΔNHV={delta:.6f}，"
                             f"胜/平/负={wins}/{len(selected)-wins-losses}/{losses}；"
                             "总体正均值不能代替非P2或S8改善的证据。")
    if not validation["common_conditions"]["identical_seed_sets"]:
        lines.append("警告：各轮种子集合不完全一致，实例差值基于各轮各自全部种子均值，不属于完整同种子配对；种子差值表只列实际交集。")
    lines += ["全部delta字段统一为候选减基线：NHV正值表示改善，GD/IGD正值表示退化。每个seed不是一个独立测试实例。",
              "", "## 规模、负荷结构和缓冲等级"]
    for field in ("size", "profile", "buffer"):
        for r in grouped[field]:
            lines.append(f"- {r['pair_id']} {r[field]}：ΔNHV={r['mean_delta_nhv']:.6f}，ΔNHV_40={r['mean_delta_nhv40']:.6f}，"
                         f"ΔGD={r['mean_delta_gd']:.6f}，ΔIGD={r['mean_delta_igd']:.6f}；胜/平/负={r['wins']}/{r['ties']}/{r['losses']}。")
    lines += ["任何分组负ΔNHV均应作为失败边界保留，尤其不能用总体正均值掩盖S8退化。指标方向不一致时不能声称一致改善。",
              "", "## 各轮独立正交设计的N响应及其他因素稳定性",
              "下述主效应在各轮自己的L16内部计算，但使用本次共同尺度；不是合并各轮正交表，也不覆盖各轮plot主效应。"]
    previous = None
    for item in rounds:
        name = item["name"]
        selected = {}
        for factor in api.FACTORS:
            group = [r for r in effects if r["round_name"] == name and r["factor"] == factor]
            top = min(group, key=lambda r: (-r["mean_nhv"], r["level_index"]))
            selected[factor] = top["level_value"]
            lines.append(f"- {name} {factor}："+"；".join(f"{r['level_value']}→{r['mean_nhv']:.6f}" for r in group)
                         +f"；所选最好水平={top['level_value']}（水平{top['level_index']}）。")
        nrows = [r for r in effects if r["round_name"] == name and r["factor"] == "N"]
        nbest = min(nrows, key=lambda r: (-r["mean_nhv"], r["level_index"]))
        lines.append(f"  {name} N最佳所选水平是否内部（水平2/3）：{nbest['level_index'] in (2,3)}。并列及小差距不等于显著最优。")
        if any(r["level_value"] == 10 for r in nrows):
            anchor = next(r for r in nrows if r["level_value"] == 10)
            for r in nrows:
                if r["level_value"] == 10:
                    continue
                text = f"  {name} N={r['level_value']}减N=10：总体ΔNHV={r['mean_nhv']-anchor['mean_nhv']:.6f}"
                for size in sorted({x["size"] for x in runs if x["round_name"] == name}):
                    text += f"；{size}={r['mean_nhv_'+size]-anchor['mean_nhv_'+size]:.6f}"
                lines.append(text+"。正值表示该N水平主效应更好，不是单独N的因果效应。")
        tested = [r["run_id"] for r in item["config"]["orthogonal_array"] if all(r[k] == selected[k] for k in api.FACTORS)]
        lines.append(f"  {name}本共同尺度下主效应候选：{selected}，N_A={2*selected['N']}；"
                     f"是否已作为完整组合实测={bool(tested)}，对应行={tested}。未实测时没有真实响应，需确认。")
        if previous is not None:
            old_name, old_selected, old_config = previous
            for factor in api.FACTORS[1:]:
                same_levels = old_config["factor_levels"][factor] == item["config"]["factor_levels"][factor]
                lines.append(f"  {old_name}→{name} {factor}：水平集合相同={same_levels}；最佳{old_selected[factor]}→{selected[factor]}。"
                             "变化可能来自N范围、实例差异或交互，不能直接归结为机制失效。")
        previous = name, selected, item["config"]
    lines.append("各轮最佳实测配置的N依次为："+" → ".join(f"{name}:{best[name]['N']}" for name in best)
                 +"。这只是不同候选集合的筛选轨迹，不证明小N独立有利。")
    lines += ["N_A=2N；N=5对应档案上限10。N改变种群、档案容量、固定FE内轮数及按轮协作机会，不能视为其余条件完全固定的N因果效应。",
              "", "## 档案容量与公平性"]
    changed = [r['round_name']+":"+r['run_id'] for r in combinations if r['rank_global']!=r['rank_global_nhv40']]
    lines.append(f"统一40点截断后全局排名发生变化的配置：{changed}。完整排名及辅助排名均保留在汇总CSV。")
    for name in best:
        subset = [r for r in runs if r["round_name"] == name]
        lines.append(f"- {name}：原始档案达到容量上限的运行{sum(r['archive_size']==r['N_A'] for r in subset)}/{len(subset)}；"
                     f"唯一非支配目标点超过40的运行{sum(r['unique_objective_count']>40 for r in subset)}。")
    lines += ["确定性截断在统一归一化空间进行，保护两极端点。它不消除档案容量对AC及整个搜索轨迹的影响；N与N_A仍为联合配置。",
              "", "## 结论与局限",
              "各轮最佳行和全局最佳行均为真实运行过的配置；未运行的田口主效应推荐组合不参与实测排名、也不被赋予虚构响应。",
              "本报告只比较统一尺度下的实测性能，不把两个不同水平的L16合并成新正交表，不拼接跨轮主效应。",
              "若选择某个组合冻结，应结合总体、S8和其他分组边界，并在冻结前进行独立确认。",
              "各轮最佳组合均经事后参数筛选；这些是探索性比较，不证明全局最优参数，不是独立确认试验。",
              "同一实例多个种子不视作独立问题实例。本报告不输出未经多重选择校正的显著性宣称。",
              "重复使用调参实例和种子不能称为独立验证；跨服务器/并发环境的耗时也不等同于严格公平单线程速度。",
              "不同种子集合的实例均值比较同时包含配置差异与随机抽样差异，不能当作同随机序列条件下的配置因果比较。",
              "同名完整参数组合若跨轮重复，保留各来源分别评价，不静默去重。所有轮次最佳行、全部两两差值均提供图表。",
              "本报告不自动创建下一轮或执行主效应推荐组合确认；未实测推荐组合只能标为待确认。"]
    if "round4" in best:
        lines.append("Round 4定位为最终正式L16，Round 1～3是水平探索依据；即使Round 4未超过历史候选也如实保留，"
                     "不自动生成Round 5。新种子不能替代未参与调参实例上的泛化验证。")
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
