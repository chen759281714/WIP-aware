"""问题适配算子：工件序列 POX/IM，以及按工序键继承的 MS。"""
from src.solution.encoder import Encoder


def explicit_os_to_job_sequence(os_seq):
    seen = {}
    for job, index in os_seq:
        if index != seen.get(job, 0):
            raise ValueError('显式 OS 工序编号不连续或重复')
        seen[job] = index + 1
    return [job for job, _ in os_seq]


def job_sequence_to_explicit_os(job_seq, operations):
    counts = {j: 0 for j in operations}
    result = []
    for job in job_seq:
        if job not in counts:
            raise ValueError('未知工件')
        result.append((job, counts[job]))
        counts[job] += 1
    Encoder(operations).validate_os(result)
    return result


def pox(a, b, operations, rng):
    """原文工件集合 POX；单工件时退化为父代副本。MS 使用同一次划分。"""
    from .comparison_common import Solution
    encoder = Encoder(operations)
    encoder.validate_os(a.os_seq)
    encoder.validate_os(b.os_seq)
    jobs = list(operations)
    if len(jobs) < 2:
        return a.copy(), b.copy()
    chosen = set(rng.sample(jobs, rng.randrange(1, len(jobs))))
    def child(first, second):
        x = explicit_os_to_job_sequence(first.os_seq)
        y = iter(j for j in explicit_os_to_job_sequence(second.os_seq) if j not in chosen)
        seq = [j if j in chosen else next(y) for j in x]
        ms = {key: (first if key[0] in chosen else second).ms_map[key]
              for key in encoder.ms_index_order}
        return Solution(job_sequence_to_explicit_os(seq, operations), ms)
    return child(a, b), child(b, a)


def insertion(a, operations, rng):
    from .comparison_common import Solution
    seq = explicit_os_to_job_sequence(a.os_seq)
    if seq:
        start, end = sorted((rng.randrange(len(seq)), rng.randrange(len(seq))))
        seq.insert(end, seq.pop(start))
    return Solution(job_sequence_to_explicit_os(seq, operations), dict(a.ms_map))


def machine_mutation(a, operations, rng):
    from .comparison_common import Solution
    keys = [(j,i) for j,ops in operations.items() for i,op in enumerate(ops) if len(op['machines']) > 1]
    if not keys:
        return insertion(a, operations, rng)
    ms = dict(a.ms_map)
    j,i = rng.choice(keys)
    ms[j,i] = rng.choice([m for m in operations[j][i]['machines'] if m != ms[j,i]])
    return Solution(list(a.os_seq), ms)


def mutate(a, operations, rng):
    # 一次变异事件只执行一种算子；50/50 为本问题的适配约定。
    if rng.random() < .5:
        return insertion(a, operations, rng)
    return machine_mutation(a, operations, rng)
