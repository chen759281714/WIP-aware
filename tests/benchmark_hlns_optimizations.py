"""Bounded code-performance and exact-equivalence checks, not a benchmark experiment.

PYTHONHASHSEED=0 python3 tests/benchmark_hlns_optimizations.py \
  --baseline-algorithm /tmp/hlns_perf_algorithm_before.py \
  --baseline-decoder /tmp/hlns_perf_decoder_before.py --stage before
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.hlns_test_support import fixture
from tests.performance_trajectory_regression import signature
from src.solution.encoder import Encoder
import src.algorithms.wip_graph_dual_population as current
import src.solution.decoder as current_decoder


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def family(label, algorithm_path, decoder_path):
    decoder = load(label + '_decoder', decoder_path)
    original_decoder = sys.modules['src.solution.decoder']
    original_algorithm = sys.modules['src.algorithms.wip_graph_dual_population']
    try:
        sys.modules['src.solution.decoder'] = decoder
        algorithm = load(label + '_algorithm', algorithm_path)
        sys.modules['src.algorithms.wip_graph_dual_population'] = algorithm
        classes = [algorithm.WIPGraphDualPopulation]
        for suffix, name in (('mgs', 'MGS'), ('hlns', 'HLNS'), ('ac', 'AC')):
            m = load(label + '_wo_' + suffix,
                     ROOT/f'src/algorithms/wip_graph_dual_population_wo_{suffix}.py')
            classes.append(getattr(m, 'WIPGraphDualPopulationWithout' + name))
    finally:
        sys.modules['src.solution.decoder'] = original_decoder
        sys.modules['src.algorithms.wip_graph_dual_population'] = original_algorithm
    return algorithm, decoder, classes


def engine(cls, data, seed=7, observe=True, n=4, fe=24):
    return cls(data['operations'], data['buffers'], variant=cls.algorithm_variant.fget(None),
               N=n, N_A=2*n, FE_max=fe, T_coop=1, seed=seed, collect_hlns_stats=observe)


def trajectory(cls, data, seed=7, observe=True, n=3, fe=18):
    e = engine(cls, data, seed, observe, n, fe)
    trace = []
    evaluated = e.evaluate
    def evaluate(ind):
        result = evaluated(ind)
        trace.append(signature((ind, e.rng.getstate(), e.n_evaluations)))
        return result
    e.evaluate = evaluate
    def checkpoint(e):
        trace.append(signature((e.P_M, e.P_S, e.A, e.diagnostics, e.rng.getstate(),
            e.n_evaluations, e.round, e._first_source_by_genotype, e.machine_priority_move_spans,
            e.unanchored_shortage_intervals)))
    e.run(diagnostic_callback=checkpoint)
    return signature(trace)


def verify(before, after, data, corpus):
    ops, buffers, _, _ = fixture()
    tiny = dict(operations=ops, buffers=buffers)
    cases = 0
    digests = []
    for b, a in zip(before[2], after[2]):
        for seed in (0, 7):
            for observe in (True, False):
                old = trajectory(b, tiny, seed, observe)
                new = trajectory(a, tiny, seed, observe)
                assert old == new, (a.__name__, seed, observe)
                digests.append([a.__name__, 'fixture', seed, observe, old, new])
                cases += 1
    for b, a in zip(before[2], after[2]):
        old = trajectory(b, data, n=4, fe=24)
        new = trajectory(a, data, n=4, fe=24)
        assert old == new, (a.__name__, 'S8')
        digests.append([a.__name__, 'S8', 7, True, old, new])
        cases += 1
    for seq, ms in corpus:
        b, a = (f[1].StageBufferWIPScheduler(data['operations'], data['buffers']) for f in (before, after))
        for provenance in (False, True):
            rb = b.decode(seq, ms, return_provenance=provenance)
            ra = a.decode(seq, ms, return_provenance=provenance)
            assert signature(rb) == signature(ra)
            assert signature(b.analyze(rb[1], rb[2], rb[0])) == signature(a.analyze(ra[1], ra[2], ra[0]))
    # Unassigned MS uses the original auto-machine path, too.
    b, a = (f[1].StageBufferWIPScheduler(ops, buffers) for f in (before, after))
    seq = fixture()[2]
    assert signature(b.decode(seq, None, True)) == signature(a.decode(seq, None, True))
    # Equal-objective initialization stresses the repeated-distance path.
    selected = []
    for f in (before, after):
        e = engine(f[2][0], data, n=12, fe=24)
        pool = [e.random_individual() for _ in range(24)]
        for ind in pool:
            ind.makespan = 1; ind.shortage = 1
        selected.append(signature(e._diverse_initial_population(pool, 'shortage')))
    assert selected[0] == selected[1]
    return dict(trajectory_cases=cases, formal_test_FE_per_family=16*18+4*24,
                decoder_genotypes=len(corpus), exact=True, trajectory_digests=digests)


def measure_pair(before, after, repeats=3):
    values = [[], []]
    for r in range(repeats):
        for i in ((0, 1) if r % 2 == 0 else (1, 0)):
            start = time.perf_counter()
            (before, after)[i]()
            values[i].append(time.perf_counter()-start)
    return dict(before_seconds=values[0], after_seconds=values[1],
                before_median=statistics.median(values[0]), after_median=statistics.median(values[1]),
                speedup=statistics.median(values[0])/statistics.median(values[1]))


def benchmark(before, after, data, corpus):
    jobs = []
    schedulers = [f[1].StageBufferWIPScheduler(data['operations'], data['buffers']) for f in (before, after)]
    decoded = [schedulers[0].decode(seq, ms, True) for seq, ms in corpus]
    for decoder in schedulers:
        for result in decoded:
            decoder.analyze(result[1], result[2], result[0])
    decode = [lambda s=s: [s.decode(seq, ms, True) for seq, ms in corpus] for s in schedulers]
    analyze = [lambda s=s: [s.analyze(r[1], r[2], r[0]) for _ in range(20) for r in decoded]
               for s in schedulers]
    jobs.append(('decode_6_genotypes', decode))
    jobs.append(('analyze_120_schedules', analyze))
    diverse = []
    for f in (before, after):
        e = engine(f[2][0], data, n=12, fe=24)
        pool = [e.random_individual() for _ in range(24)]
        for ind in pool:
            ind.makespan = 1; ind.shortage = 1
        def cold_selection(e=e, pool=pool):
            e._distance_rank_cache.clear()
            if hasattr(e, '_os_pair_cache'):
                e._os_pair_cache.clear()
            return e._diverse_initial_population(pool, 'shortage')
        diverse.append(cold_selection)
    jobs.append(('tied_initialization_selection_cold', diverse))
    mgs = []
    for f in (before, after):
        e = engine(f[2][0], data)
        individuals = [f[0].Individual(seq, ms, r[0], 0, r[1], r[2], None, r[3],
                        tuple(e.encoder.ms_index_order)) for (seq, ms), r in zip(corpus, decoded)]
        mgs.append(lambda e=e, individuals=individuals: [e.makespan_actions(ind) for ind in individuals])
    jobs.append(('mgs_enumeration_6_schedules', mgs))
    for i, name in enumerate(('full', 'wo_mgs', 'wo_hlns', 'wo_ac')):
        jobs.append((name+'_24FE', [lambda cls=f[2][i]: engine(cls, data).run() for f in (before, after)]))
    return {name: measure_pair(*pair) for name, pair in jobs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-algorithm', required=True, type=Path)
    p.add_argument('--baseline-decoder', required=True, type=Path)
    p.add_argument('--stage', required=True)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--output-dir', type=Path, default=ROOT/'experiments/results/hlns_code_optimization')
    args = p.parse_args()
    data = json.loads((ROOT/'data/final_benchmark/instances/WIPHFSP_S8_P2_B1.json').read_text())
    enc = Encoder(data['operations'], rng=random.Random(71))
    corpus = [(enc.generate_random_os(), enc.build_ms_map(enc.generate_random_ms())) for _ in range(6)]
    before = family('perf_before', args.baseline_algorithm, args.baseline_decoder)
    after = family('perf_after', ROOT/'src/algorithms/wip_graph_dual_population.py', ROOT/'src/solution/decoder.py')
    result = dict(stage=args.stage, instance='WIPHFSP_S8_P2_B1', jobs=120, operations=960,
                  parameters=dict(N=4, N_A=8, FE_max=24, T_coop=1, seed=7),
                  verification=verify(before, after, data, corpus))
    if not args.verify_only:
        result['timing'] = benchmark(before, after, data, corpus)
    result['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        args.baseline_algorithm, args.baseline_decoder, ROOT/'src/algorithms/wip_graph_dual_population.py',
        ROOT/'src/solution/decoder.py')}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/(args.stage+'.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
