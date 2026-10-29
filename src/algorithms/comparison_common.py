"""独立于 Full 的统一评价、个体和标准 Pareto 辅助函数。"""
from dataclasses import dataclass
import math
import random
from src.solution.encoder import Encoder
from src.solution.decoder import StageBufferWIPScheduler


@dataclass
class Solution:
    os_seq: list
    ms_map: dict
    makespan: float = None
    shortage: float = None

    def copy(self):
        return Solution(list(self.os_seq), dict(self.ms_map), self.makespan, self.shortage)

    @property
    def objectives(self):
        return self.makespan, self.shortage


def dominates(a,b):
    return all(x <= y for x,y in zip(a.objectives,b.objectives)) and a.objectives != b.objectives


def fronts(pop):
    # 标准快速非支配分层；以索引区分重复目标个体。
    dominated = [[] for _ in pop]
    count = [0]*len(pop)
    for i,a in enumerate(pop):
        for j,b in enumerate(pop):
            if dominates(a,b): dominated[i].append(j)
            elif dominates(b,a): count[i] += 1
    layer = [i for i,c in enumerate(count) if c == 0]
    result = []
    while layer:
        result.append(layer)
        following = []
        for i in layer:
            for j in dominated[i]:
                count[j] -= 1
                if count[j] == 0: following.append(j)
        layer = following
    return result


def crowding(pop, indices):
    # 按完整目标向量分组：每组仅稳定选择第一个个体代表几何位置。
    # 非代表副本距离为零；不删除基因型，不改变种群规模。
    representatives = {}
    for i in indices:
        representatives.setdefault(pop[i].objectives, i)
    unique = list(representatives.values())
    distances = {i: 0. for i in indices}
    for k in range(2):
        order = sorted(unique, key=lambda i: (pop[i].objectives[k], i))
        if not order: continue
        span = pop[order[-1]].objectives[k]-pop[order[0]].objectives[k]
        if span == 0: continue
        distances[order[0]] = distances[order[-1]] = math.inf
        for p in range(1,len(order)-1):
            distances[order[p]] += (pop[order[p+1]].objectives[k]-pop[order[p-1]].objectives[k])/span
    return distances


def environmental(pop, capacity):
    selected = []
    for layer in fronts(pop):
        if len(selected)+len(layer) <= capacity: selected.extend(layer)
        else:
            d = crowding(pop,layer)
            selected.extend(sorted(layer,key=lambda i:(-d[i],i))[:capacity-len(selected)])
            break
    return [pop[i] for i in selected]


def native(pop):
    # 双目标最小化的排序扫描；只提取当前原生状态，不累计历史候选。
    first = {}
    for a in pop:
        first.setdefault(a.objectives, a)
    front = set()
    best = math.inf
    for x,y in sorted(first):
        if y < best:
            front.add((x,y)); best=y
    return [a.copy() for key,a in first.items() if key in front]


def genotype(a):
    return tuple(a.os_seq), tuple(sorted(a.ms_map.items()))


def deduplicate(pop):
    result,seen=[],set()
    for a in pop:
        key=genotype(a)
        if key not in seen: result.append(a); seen.add(key)
    return result


class ComparisonBase:
    """只共享表示和评价，不共享算法环境选择。回调在状态提交后触发。"""
    def __init__(self,operations,buffers,*,population_size=100,FE_max=40000,seed=None,
                 crossover_probability=.9,mutation_probability=.1,snapshot_interval_fe=2000):
        for name,value in [('population_size',population_size),('FE_max',FE_max),('snapshot_interval_fe',snapshot_interval_fe)]:
            if type(value) is not int or value < 1: raise ValueError(name)
        if FE_max < population_size: raise ValueError('预算不得小于初始化规模')
        if not 0 <= crossover_probability <= 1 or not 0 <= mutation_probability <= 1: raise ValueError('非法概率')
        self.operations,self.buffers=operations,buffers
        self.population_size,self.FE_max=population_size,FE_max
        self.crossover_probability,self.mutation_probability=crossover_probability,mutation_probability
        self.snapshot_interval_fe=snapshot_interval_fe
        self.rng=random.Random(seed)
        self.encoder=Encoder(operations,rng=self.rng)
        self.decoder=StageBufferWIPScheduler(operations,buffers)
        self.n_evaluations=0
        self.A=[]; self.snapshots=[]
        self.diagnostics={'evaluations':0,'infeasible':0,'infeasible_reasons':{},'crossovers':0,'mutations':0}
        self._callback=None; self._last_snapshot=-1; self._pending_snapshot=None

    def evaluate(self,a):
        if self.n_evaluations >= self.FE_max: raise RuntimeError('FE 预算耗尽')
        self.encoder.validate_os(a.os_seq)
        if set(a.ms_map) != set(self.encoder.ms_index_order): raise ValueError('MS 工序键错误')
        self.encoder.build_ms_map([a.ms_map[k] for k in self.encoder.ms_index_order])
        self._flush_snapshot()  # 下一次真实评价前，提交上一 FE 的最终状态
        self.n_evaluations += 1
        self.diagnostics['evaluations'] += 1
        # 仅识别解码器明确的无事件可推进死锁；迭代上限等程序异常原样传播。
        try:
            m,s,t=self.decoder.decode(a.os_seq,a.ms_map)
        except RuntimeError as exc:
            if not str(exc).startswith('死锁：'): raise
            self.diagnostics['infeasible'] += 1
            reasons=self.diagnostics['infeasible_reasons']; reasons[str(exc)]=reasons.get(str(exc),0)+1
            return None
        stats=self.decoder.analyze(s,t,m)
        a.makespan,a.shortage=m,stats['shortage']['total_shortage_area']
        if any(not math.isfinite(v) or v < 0 for v in a.objectives): raise ValueError('非法目标值')
        return a

    def initial(self):
        pop=[]
        while len(pop)<self.population_size and self.n_evaluations < self.FE_max:
            a=self.evaluate(Solution(self.encoder.generate_random_os(),self.encoder.build_ms_map(self.encoder.generate_random_ms())))
            if a is not None: pop.append(a)
            self.observe(pop)
        if len(pop) != self.population_size:
            self._pending_snapshot=None
            raise RuntimeError(f'初始化失败：预算 {self.FE_max} 已耗尽，仅取得 {len(pop)}/{self.population_size} 个可行个体')
        return pop

    def children(self,a,b):
        from .comparison_operators import pox,mutate
        if self.rng.random()<self.crossover_probability:
            self.diagnostics['crossovers']+=1
            pair=pox(a,b,self.operations,self.rng)
        else: pair=(a.copy(),b.copy())
        output=[]
        for c in pair:
            if self.rng.random()<self.mutation_probability:
                self.diagnostics['mutations']+=1
                c=mutate(c,self.operations,self.rng)
            output.append(c)
        return output

    def generate_single_offspring(self,a,b):
        """单候选更新：POX 取第一个子代，未交叉复制第一父代，仅一次变异判断。"""
        from .comparison_operators import pox,mutate
        if self.rng.random()<self.crossover_probability:
            self.diagnostics['crossovers']+=1
            child=pox(a,b,self.operations,self.rng)[0]
        else:
            child=a.copy()
        if self.rng.random()<self.mutation_probability:
            self.diagnostics['mutations']+=1
            child=mutate(child,self.operations,self.rng)
        return child

    def _flush_snapshot(self):
        if self._pending_snapshot is None: return
        snap=self._pending_snapshot
        assert snap['fe']==self.n_evaluations
        self._pending_snapshot=None
        self.snapshots.append(snap)
        self._last_snapshot=snap['fe']
        if self._callback is not None: self._callback(self)

    def observe(self,state,final=False):
        # 同一 FE 的环境选择/档案更新可多次提交，但外部通知延迟至下一次
        # 实际评价前；最终 FE 仅由 finish 在合法收尾后通知。没有未来评价回填。
        self.A=native(state)
        fe=self.n_evaluations
        if final or (fe and fe % self.snapshot_interval_fe == 0):
            self._pending_snapshot={'fe':fe,'archive':[list(x.objectives) for x in self.A],
                                    'archive_size':len(self.A)}
        if final: self._flush_snapshot()

    def start(self,callback):
        if self.n_evaluations: raise RuntimeError('同一优化器仅允许运行一次')
        self._callback=callback

    def finish(self,state):
        self.observe(state,final=True)
        return self.A
