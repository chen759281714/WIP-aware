"""Read-only batch snapshots and a predeclared Full FE convergence audit.

Resume skips complete runs, never restores a search from objective snapshots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics as st
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.algorithms.wip_graph_dual_population import WIPGraphDualPopulation
from experiments import run_pre_ablation as metrics

INSTANCES = (
    'WIPHFSP_S1_P1_B1', 'WIPHFSP_S1_P2_B2', 'WIPHFSP_S1_P3_B3', 'WIPHFSP_S1_P4_B1',
    'WIPHFSP_S4_P1_B2', 'WIPHFSP_S4_P2_B3', 'WIPHFSP_S4_P3_B1', 'WIPHFSP_S4_P4_B2',
    'WIPHFSP_S8_P1_B3', 'WIPHFSP_S8_P2_B1', 'WIPHFSP_S8_P3_B2', 'WIPHFSP_S8_P4_B3',
)
SEEDS = tuple(range(5))
PARAMS = dict(N=50, N_A=100, p_mut=0.1, rho=0.8, T_coop=10, FE_max=60000)
INTERVAL = 1000
CANDIDATES = tuple(range(10000, 50001, 5000))
LIMITS = dict(nhv=0.01, igd=0.05, makespan=0.01, shortage=0.03)
EPS = 1e-12
OUT = ROOT / 'experiments/results/fe_convergence'
SOURCE_FILES = (
    'src/algorithms/wip_graph_dual_population.py', 'src/solution/encoder.py',
    'src/solution/decoder.py', 'experiments/run_pre_ablation.py',
    'experiments/run_fe_convergence.py',
)
POLICY = {
    'limits': LIMITS, 'overall_min_rate': 0.85, 'size_min_rate': 0.80,
    'window_width_FE': 5000, 'window_count': 2,
    'windows': 'Peak improvement over all 1000-FE samples, not just endpoints; same limits and coverage.',
    'instance_guard': 'Every instance mean residual NHV<=1% and IGD<=5%, including both windows.',
    'instance_earliest': 'At least 4/5 seeds pass and instance mean NHV/IGD guard, including both windows.',
    'late_event': 'A 5000-FE window passes all four limits, followed later by a window that fails at least one.',
    'candidate_FE': CANDIDATES, 'reference': 'All snapshots including initialization, all five seeds; empirical only.',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {p: sha(ROOT / p) for p in SOURCE_FILES}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        with tmp.open('w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def csv_save(path, rows):
    if not rows:
        path.write_text('no_events\n')
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def objectives(engine):
    return [[float(x.makespan), float(x.shortage)] for x in engine.A]


class SnapshotObserver:
    """Observe only after initialization or the formal batch archive update."""
    def __init__(self, instance, seed, interval=INTERVAL):
        self.instance, self.seed, self.interval = instance, seed, interval
        self.started = time.perf_counter()
        self.snapshots = []

    def __call__(self, engine):
        fe = engine.n_evaluations
        if self.snapshots and fe % self.interval:
            return
        if self.snapshots and fe == self.snapshots[-1]['FE']:
            return
        points = objectives(engine)
        self.snapshots.append(dict(
            instance=self.instance, seed=self.seed, FE=fe, archive_size=len(points),
            best_makespan=min(x for x, y in points), best_shortage=min(y for x, y in points),
            archive_objectives=points, elapsed_seconds=time.perf_counter()-self.started))


def identity(instance, seed, hashes):
    return dict(schema='full-fe-convergence-v1', instance=instance, seed=seed,
                algorithm_variant='HLNS_FULL', parameters=dict(PARAMS, seed=seed),
                sources=hashes, instance_sha256=sha(ROOT / 'data/final_benchmark/instances' / (instance+'.json')),
                python_hash_seed='0', checkpoint_interval=INTERVAL)


def validate_record(record, expected):
    if record['identity'] != expected or record['actual_FE'] != PARAMS['FE_max']:
        raise ValueError('Identity, source, parameters or final FE mismatch')
    if record['actual_parameters'] != dict(PARAMS, seed=expected['seed']):
        raise ValueError('Actual constructor parameters mismatch')
    if record['population_sizes'] != {'P_M': 50, 'P_S': 50}:
        raise ValueError('Population size mismatch')
    if record['mechanisms'] != ['MGS', 'HLNS', True, False]:
        raise ValueError('Full mechanism mismatch')
    snaps = record['snapshots']
    if [r['FE'] for r in snaps] != [100] + list(range(1000, 60001, 1000)):
        raise ValueError('Missing, duplicate or unordered checkpoints')
    last_elapsed = -1
    for snap in snaps:
        pts = snap['archive_objectives']
        if (snap['instance'], snap['seed']) != (expected['instance'], expected['seed']):
            raise ValueError('Snapshot task mismatch')
        if not pts or len(pts) != snap['archive_size'] or len(pts) > PARAMS['N_A']:
            raise ValueError('Invalid archive size')
        if any(len(p) != 2 or any(not math.isfinite(z) or z < 0 for z in p) for p in pts):
            raise ValueError('Invalid objective')
        if metrics.nondominated_filter(list(map(tuple, pts))) != sorted(set(map(tuple, pts))):
            raise ValueError('Archive contains dominated points')
        if snap['best_makespan'] != min(p[0] for p in pts) or snap['best_shortage'] != min(p[1] for p in pts):
            raise ValueError('Extremes mismatch')
        if snap['elapsed_seconds'] < last_elapsed:
            raise ValueError('Elapsed time out of order')
        last_elapsed = snap['elapsed_seconds']
    if snaps[-1]['archive_objectives'] != record['final_archive']:
        raise ValueError('Final checkpoint differs from actual archive')


def run_task(instance, seed, output, hashes):
    if source_hashes() != hashes:
        raise ValueError('Sources changed during execution')
    expected = identity(instance, seed, hashes)
    path = Path(output) / 'raw' / instance / ('seed_%d.json' % seed)
    if path.exists():
        try:
            saved = json.loads(path.read_text())
            validate_record(saved, expected)
            return dict(instance=instance, seed=seed, status='skipped', seconds=saved['runtime_seconds'])
        except (ValueError, KeyError, TypeError):
            # Preserve invalid records for inspection, never accept a partial run.
            os.replace(path, path.with_suffix('.invalid.%d.json' % time.time_ns()))
    data = json.loads((ROOT / 'data/final_benchmark/instances' / (instance+'.json')).read_text())
    engine = WIPGraphDualPopulation(data['operations'], data['buffers'], variant='HLNS_FULL', **dict(PARAMS, seed=seed))
    observer = SnapshotObserver(instance, seed)
    engine.run(diagnostic_callback=observer)
    actual = {k: getattr(engine, k) for k in PARAMS}
    actual['seed'] = seed
    record = dict(identity=expected, actual_parameters=actual,
                  population_sizes={'P_M': len(engine.P_M), 'P_S': len(engine.P_S)},
                  mechanisms=[engine.pm_specialized_kind, engine.ps_specialized_kind,
                              engine.pm_archive_guidance_enabled, engine.ps_archive_guidance_enabled],
                  actual_FE=engine.n_evaluations, runtime_seconds=time.perf_counter()-observer.started,
                  snapshots=observer.snapshots, final_archive=objectives(engine),
                  completed_utc=datetime.now(timezone.utc).isoformat())
    if source_hashes() != hashes:
        raise ValueError('Sources changed during execution; refusing result')
    validate_record(record, expected)
    atomic_json(path, record)
    return dict(instance=instance, seed=seed, status='completed', seconds=record['runtime_seconds'])


def residual(rows, index, stop=None):
    current = rows[index]
    future = rows[index:] if stop is None else [r for r in rows[index:] if r['FE'] <= stop]
    result = {}
    for key in LIMITS:
        v = current[key]
        best = max(r[key] for r in future) if key == 'nhv' else min(r[key] for r in future)
        gain = best-v if key == 'nhv' else v-best
        result['R_'+key] = max(0.0, gain) / max(abs(v), EPS)
    result['pass'] = all(result['R_'+k] <= limit+EPS for k, limit in LIMITS.items())
    return result


def summarize(rows, dimensions):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[d] for d in dimensions)].append(r)
    output = []
    for keys, group in sorted(groups.items()):
        r = dict(zip(dimensions, keys))
        r['n_runs'] = len(group)
        for key in ('nhv', 'igd', 'makespan', 'shortage', 'relative_makespan', 'relative_shortage',
                    'elapsed_seconds', 'R_nhv', 'R_igd', 'R_makespan', 'R_shortage'):
            values = [g[key] for g in group]
            r[key+'_mean'] = st.mean(values)
            r[key+'_std'] = st.stdev(values) if len(values) > 1 else 0.0
        r['threshold_pass_rate'] = st.mean(int(g['pass']) for g in group)
        output.append(r)
    return output


def coverage_ok(group):
    overall = st.mean(int(g['pass']) for g in group) >= 0.85-EPS
    sizes = {size: st.mean(int(g['pass']) for g in group if g['size']==size) for size in ('S1','S4','S8')}
    instances = defaultdict(list)
    for g in group:
        instances[g['instance']].append(g)
    guard = all(st.mean(r['R_'+k] for r in rs) <= LIMITS[k]+EPS
                for rs in instances.values() for k in ('nhv','igd'))
    return overall and min(sizes.values()) >= 0.80-EPS and guard, sizes, guard


def analyze(output, hashes):
    records = []
    for instance in INSTANCES:
        for seed in SEEDS:
            path = output/'raw'/instance/('seed_%d.json' % seed)
            record = json.loads(path.read_text())
            validate_record(record, identity(instance, seed, hashes))
            records.append(record)
    directory = output/'analysis'
    directory.mkdir(parents=True, exist_ok=True)
    refs, rows, trajectories = {}, [], {}
    for instance in INSTANCES:
        runs = [r for r in records if r['identity']['instance']==instance]
        pool = [tuple(p) for run in runs for snap in run['snapshots'] for p in snap['archive_objectives']]
        bounds = metrics.pooled_bounds(pool)
        ref = [metrics.normalize_point(p, bounds) for p in metrics.nondominated_filter(pool)]
        refs[instance] = dict(bounds=bounds, reference_front=ref)
        for run in runs:
            trajectory = []
            baseline = next(snap for snap in run['snapshots'] if snap['FE']==1000)
            for snap in run['snapshots']:
                front = [metrics.normalize_point(p, bounds) for p in snap['archive_objectives']]
                r = dict(instance=instance, size=instance.split('_')[1], seed=run['identity']['seed'],
                         FE=snap['FE'], archive_size=snap['archive_size'],
                         nhv=metrics.compute_hv_2d(front)/1.21, igd=metrics.compute_igd(front, ref),
                         makespan=snap['best_makespan'], shortage=snap['best_shortage'],
                         relative_makespan=snap['best_makespan']/max(baseline['best_makespan'],EPS),
                         relative_shortage=snap['best_shortage']/max(baseline['best_shortage'],EPS),
                         elapsed_seconds=snap['elapsed_seconds'])
                trajectory.append(r)
            for j, r in enumerate(trajectory):
                r.update(residual(trajectory,j))
            trajectories[(instance,run['identity']['seed'])] = trajectory
            rows.extend(trajectory)
    atomic_json(directory/'fixed_references.json', refs)
    csv_save(directory/'checkpoint_metrics.csv', rows)
    overall = summarize(rows, ['FE'])
    instances = summarize(rows, ['instance','size','FE'])
    sizes = summarize(rows, ['size','FE'])
    csv_save(directory/'overall_convergence_summary.csv', overall)
    csv_save(directory/'instance_checkpoint_summary.csv', instances)
    csv_save(directory/'size_group_convergence_summary.csv', sizes)
    candidates, windows = [], []
    per_instance = []
    for fe in CANDIDATES:
        group = [r for r in rows if r['FE']==fe]
        qualified, rates, guard = coverage_ok(group)
        window_groups = []
        for offset in (0,5000):
            start = fe+offset
            wg = []
            for (name,seed), trajectory in trajectories.items():
                j = next(j for j,r in enumerate(trajectory) if r['FE']==start)
                r = dict(instance=name,size=name.split('_')[1],seed=seed, candidate_FE=fe,
                         start_FE=start,end_FE=start+5000,**residual(trajectory,j,start+5000))
                wg.append(r)
                windows.append(r)
            window_groups.append(wg)
            qualified = qualified and coverage_ok(wg)[0]
        candidates.append(dict(FE=fe, qualified=qualified,
            threshold_pass_rate=st.mean(int(r['pass']) for r in group),
            **{size+'_pass_rate': value for size,value in rates.items()}, instance_guard=guard,
            window_1_pass=coverage_ok(window_groups[0])[0], window_2_pass=coverage_ok(window_groups[1])[0],
            **{k+'_mean':st.mean(r[k] for r in group) for k in ('nhv','igd','R_nhv','R_igd','R_makespan','R_shortage')}))
        for name in INSTANCES:
            sub = [r for r in group if r['instance']==name]
            instance_ok = st.mean(int(r['pass']) for r in sub)>=.8-EPS
            instance_ok &= all(st.mean(r['R_'+k] for r in sub)<=LIMITS[k]+EPS for k in ('nhv','igd'))
            for wg in window_groups:
                ws = [r for r in wg if r['instance']==name]
                instance_ok &= st.mean(int(r['pass']) for r in ws)>=.8-EPS
                instance_ok &= all(st.mean(r['R_'+k] for r in ws)<=LIMITS[k]+EPS for k in ('nhv','igd'))
            per_instance.append(dict(instance=name,FE=fe,qualified=bool(instance_ok),
                                     pass_seeds=';'.join(str(r['seed']) for r in sub if r['pass']),
                                     failed_seeds=';'.join(str(r['seed']) for r in sub if not r['pass']),
                                     **{k+'_mean':st.mean(r[k] for r in sub) for k in ('R_nhv','R_igd','R_makespan','R_shortage')}))
    csv_save(directory/'candidate_fe_comparison.csv', candidates)
    csv_save(directory/'candidate_window_checks.csv', windows)
    csv_save(directory/'instance_candidate_checks.csv', per_instance)
    earliest = []
    for name in INSTANCES:
        valid = [r['FE'] for r in per_instance if r['instance']==name and r['qualified']]
        earliest.append(dict(instance=name, earliest_qualified_FE=min(valid) if valid else None))
    csv_save(directory/'instance_convergence_summary.csv', earliest)
    late = []
    for (name,seed), trajectory in trajectories.items():
        quiet = None
        for start in range(5000,60000,5000):
            j = next(j for j,r in enumerate(trajectory) if r['FE']==start)
            gains = residual(trajectory,j,start+5000)
            if gains['pass']:
                if quiet is None:
                    quiet = start
            else:
                if quiet is not None:
                    late.append(dict(instance=name,seed=seed,quiet_start_FE=quiet,
                                     resurgence_start_FE=start,resurgence_end_FE=start+5000,**gains))
                quiet = None
    csv_save(directory/'late_improvement_events.csv', late)
    valid = [r['FE'] for r in candidates if r['qualified']]
    recommended = min(valid) if valid else None
    audit_fe = recommended or 50000
    failures = [dict(instance=r['instance'],seed=r['seed'],
                     failed_metrics=[k for k,v in LIMITS.items() if r['R_'+k]>v+EPS],
                     **{k:r[k] for k in r if k.startswith('R_')}) for r in rows if r['FE']==audit_fe and not r['pass']]
    final_window = []
    for (name,seed),trajectory in trajectories.items():
        j = next(j for j,r in enumerate(trajectory) if r['FE']==55000)
        final_window.append(dict(instance=name,seed=seed,**residual(trajectory,j,60000)))
    decision = dict(recommended_FE=recommended,
        evidence='PASSED_PREDECLARED_CRITERIA' if valid else 'NO_VALIDATED_BUDGET_WITHIN_OBSERVATION',
        provisional_FE_if_immediate_comparison_required=None if valid else 60000,
        policy=POLICY, candidates=candidates, instance_earliest=earliest,
        failures_at_FE=audit_fe, failed_trajectories=failures, final_55000_60000_window=final_window,
        late_event_count=len(late), formal_FE=sum(r['actual_FE'] for r in records),
        total_run_seconds=sum(r['runtime_seconds'] for r in records),
        reference_note='Empirical front, not true Pareto optimum; no claim beyond 60000 FE.',
        sources_before=hashes,sources_after=source_hashes())
    atomic_json(directory/'fe_recommendation.json', decision)
    make_figures(directory, overall, sizes, instances, recommended)
    write_report(directory, decision, overall, sizes, earliest, records)
    return decision


def make_figures(directory, overall, sizes, instances, recommended):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/wip_fe_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_dir = directory/'figures'
    figure_dir.mkdir(exist_ok=True)
    def panel(ax, group, key, label):
        group = [r for r in group if r['FE']>=1000]
        xs = [r['FE'] for r in group]
        ys = [r[key+'_mean'] for r in group]
        line, = ax.plot(xs, ys, label=label)
        sd = [r[key+'_std'] for r in group]
        ax.fill_between(xs,[max(0,y-z) for y,z in zip(ys,sd)],
                        [y+z for y,z in zip(ys,sd)],color=line.get_color(),alpha=.12)
        ax.set_xlabel('FE'); ax.grid(alpha=.2)
        if recommended:
            ax.axvline(recommended,color='black',linestyle='--',linewidth=1)
    for key in ('nhv','igd','relative_makespan','relative_shortage'):
        fig,ax=plt.subplots(figsize=(8,4.5))
        panel(ax,overall,key,'All instances')
        ax.set_ylabel(key + (' (1000 FE = 1 per run)' if key.startswith('relative') else ''))
        ax.set_title('Mean +/- sample SD; descriptive, not confidence interval')
        fig.tight_layout();fig.savefig(figure_dir/(key+'_overall.png'),dpi=150);plt.close(fig)
        if key in ('nhv','igd'):
            fig,ax=plt.subplots(figsize=(8,4.5))
            for size in ('S1','S4','S8'):
                panel(ax,[r for r in sizes if r['size']==size],key,size)
            ax.set_ylabel(key);ax.legend();fig.tight_layout()
            fig.savefig(figure_dir/(key+'_by_size.png'),dpi=150);plt.close(fig)
    for name in INSTANCES:
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        group=[r for r in instances if r['instance']==name]
        for ax,key in zip(axs.flat,('nhv','igd','makespan','shortage')):
            panel(ax,group,key,name);ax.set_ylabel(key)
        fig.suptitle(name+' (5 seeds, mean +/- SD)');fig.tight_layout()
        fig.savefig(figure_dir/(name+'_convergence.png'),dpi=150);plt.close(fig)


def write_report(directory, decision, overall, sizes, earliest, records):
    lines=['# Full FE convergence audit', '',
           '## A. Execution',
           f'Completed {len(records)}/60 runs; actual FE=60000 each; total FE={decision["formal_FE"]}.',
           f'Parameters: {PARAMS}. Sum of run elapsed times: {decision["total_run_seconds"]:.2f} seconds (not wall time).',
           'Existing diagnostic_callback reused after archive updates; no algorithm/encoder/decoder edits.',
           'FE_max only caps evaluations and final batch allocation. All candidate checkpoints are complete batches.',
           'Resume skips complete source-validated runs; interrupted runs restart from initialization.',
           '', '## B. Fixed-scale metrics',
           'Each instance uses all initialization/checkpoint archives across all five seeds for fixed bounds and empirical reference.',
           'NHV=HV/1.21, reference=(1.1,1.1). IGD reference is empirical, not the true Pareto front.',
           'Extremes may belong to different schedules. Cross-instance raw extremes are not used as normalized curves.',
           '|FE|NHV mean (SD)|IGD mean (SD)|R_H mean|R_I mean|R_C mean|R_S mean|four-limit pass rate|',
           '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in overall:
        if r['FE']%5000==0:
            lines.append(f'|{r["FE"]}|{r["nhv_mean"]:.6f} ({r["nhv_std"]:.6f})|{r["igd_mean"]:.6f} ({r["igd_std"]:.6f})|'
                         f'{r["R_nhv_mean"]:.2%}|{r["R_igd_mean"]:.2%}|{r["R_makespan_mean"]:.2%}|{r["R_shortage_mean"]:.2%}|{r["threshold_pass_rate"]:.1%}|')
    lines += ['', 'Pass rates at 55000/60000 have insufficient follow-up and do not qualify a budget.',
              '', '## C. Predeclared decision',json.dumps(POLICY,indent=2),
              f'Recommended FE: {decision["recommended_FE"]}; evidence: {decision["evidence"]}.',
              'If none qualifies, 60000 is only a provisional comparison cap, NOT a validated convergence budget.',
              '|FE|overall|S1|S4|S8|instance guard|window 1|window 2|qualified|',
              '|---:|---:|---:|---:|---:|---|---|---|---|']
    for r in decision['candidates']:
        lines.append(f'|{r["FE"]}|{r["threshold_pass_rate"]:.1%}|{r["S1_pass_rate"]:.1%}|{r["S4_pass_rate"]:.1%}|{r["S8_pass_rate"]:.1%}|{r["instance_guard"]}|{r["window_1_pass"]}|{r["window_2_pass"]}|{r["qualified"]}|')
    lines += ['', '## D. Instance boundaries',json.dumps(earliest,indent=2),
              f'Failures at FE={decision["failures_at_FE"]}:',json.dumps(decision['failed_trajectories'],indent=2),
              '## E. Late improvements',f'Quiet-then-resurgent windows: {decision["late_event_count"]}.',
              'See late_improvement_events.csv; this is observed trajectory behavior, not causal proof.',
              'Final 55000-60000 windows:',json.dumps(decision['final_55000_60000_window'],indent=2),
              '', '## F. Interpretation boundaries',
              'Instances, not seeds, are the independent benchmark units. Size groups contain four instances each.',
              'Mean/SD and coverage are descriptive; no global optimality or behavior beyond 60000 is established.',
              'No mechanisms, parameters or instance inclusion rules were adapted after viewing outcomes.',
              'Any observed plateau is conditional on fixed parameters and the sampled seeds/instances.',
              'Source hashes before/after are in fe_recommendation.json; wall time is in run_manifest.json.']
    (directory/'fe_convergence_report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['all','run','analyze'],default='all')
    args=parser.parse_args()
    if os.environ.get('PYTHONHASHSEED')!='0':
        os.environ['PYTHONHASHSEED']='0'
        os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve())]+sys.argv[1:])
    if tuple(metrics.SELECTED_INSTANCES)!=INSTANCES:
        raise ValueError('Selected instances differ from current pre-ablation')
    OUT.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes()
    config=dict(parameters=PARAMS,seeds=SEEDS,instances=INSTANCES,workers=5,
                checkpoint_interval=INTERVAL,policy=POLICY,sources=hashes)
    config_path=OUT/'experiment_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=json.loads(json.dumps(config)):
        raise ValueError('Existing audit configuration/source differs; refusing to mix runs')
    atomic_json(config_path,config)
    started=time.perf_counter()
    previous=[]
    manifest_path=OUT/'run_manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text()).get('sessions',[])
    statuses=[]
    if args.mode!='analyze':
        tasks=[(name,seed) for name in INSTANCES for seed in SEEDS]
        # Schedule large runs first to reduce the tail; each run has its own RNG.
        tasks.sort(key=lambda t:-int(t[0].split('_')[1][1:]))
        with ProcessPoolExecutor(max_workers=5,mp_context=get_context('spawn')) as pool:
            pending={pool.submit(run_task,name,seed,str(OUT),hashes):(name,seed) for name,seed in tasks}
            for future in as_completed(pending):
                name,seed=pending[future]
                try:
                    status=future.result()
                except Exception:
                    status=dict(instance=name,seed=seed,status='failed',error=traceback.format_exc())
                statuses.append(status)
                atomic_json(OUT/'progress.json',dict(expected=60,finished=len(statuses),tasks=statuses))
                print(json.dumps(status),flush=True)
        # A single retry handles transient task failures without accepting partial files.
        failures=[r for r in statuses if r['status']=='failed']
        for failed in failures:
            try:
                replacement=run_task(failed['instance'],failed['seed'],str(OUT),hashes)
                statuses[statuses.index(failed)]=replacement
            except Exception:
                failed['retry_error']=traceback.format_exc()
    valid,missing=[],[]
    for name in INSTANCES:
        for seed in SEEDS:
            try:
                path=OUT/'raw'/name/('seed_%d.json'%seed)
                record=json.loads(path.read_text())
                validate_record(record,identity(name,seed,hashes));valid.append((name,seed))
            except (OSError,ValueError,KeyError,TypeError) as exc:
                missing.append(dict(instance=name,seed=seed,error=str(exc)))
    session=dict(mode=args.mode,elapsed_seconds=time.perf_counter()-started,completed_utc=datetime.now(timezone.utc).isoformat())
    sessions=previous+[session]
    manifest=dict(expected=60,completed=len(valid),missing=missing,complete=not missing,
                  formal_FE=len(valid)*60000,sessions=sessions,tasks=statuses,
                  sources_before=hashes,sources_after=source_hashes())
    atomic_json(manifest_path,manifest)
    if missing:
        raise RuntimeError('Incomplete audit: '+str(missing))
    if source_hashes()!=hashes:
        raise RuntimeError('Sources changed')
    if args.mode!='run':
        decision=analyze(OUT,hashes)
        session['elapsed_seconds']=time.perf_counter()-started
        atomic_json(manifest_path,manifest)
        print('RECOMMENDATION',decision['recommended_FE'],decision['evidence'],flush=True)


if __name__=='__main__':
    main()
