"""SPEA2：强度适应度、当前联合集归一化距离与字典序近邻截断。"""
import math
from .comparison_common import ComparisonBase,dominates


def fitness(pop):
    n=len(pop)
    strength=[sum(dominates(a,b) for b in pop) for a in pop]
    raw=[sum(strength[j] for j,b in enumerate(pop) if dominates(b,a)) for a in pop]
    lo=[min(a.objectives[j] for a in pop) for j in range(2)]
    span=[max(a.objectives[j] for a in pop)-lo[j] or 1. for j in range(2)]
    normalized=[tuple((a.objectives[j]-lo[j])/span[j] for j in range(2)) for a in pop]
    distances=[[math.dist(a,b) for b in normalized] for a in normalized]
    k=min(max(1,int(math.sqrt(n))),max(0,n-1))
    density=[1/(sorted(d for j,d in enumerate(distances[i]) if j!=i)[k-1]+2) if n>1 else 0. for i in range(n)]
    return strength,raw,density,[r+d for r,d in zip(raw,density)],distances


def archive_selection(pop,capacity,return_fitness=False):
    _,raw,_,f,d=fitness(pop)
    chosen=[i for i in range(len(pop)) if raw[i]==0]
    if len(chosen)<capacity:
        chosen+=sorted((i for i in range(len(pop)) if i not in chosen),key=lambda i:(f[i],i))[:capacity-len(chosen)]
    while len(chosen)>capacity:
        victim=min(chosen,key=lambda i:(sorted(d[i][j] for j in chosen if j!=i),i))
        chosen.remove(victim)
    selected = [pop[i] for i in chosen]
    return (selected, [f[i] for i in chosen]) if return_fitness else selected


class SPEA2(ComparisonBase):
    algorithm_variant='SPEA2'
    def __init__(self,*args,archive_size=100,**kwargs):
        super().__init__(*args,**kwargs)
        if type(archive_size) is not int or archive_size<1: raise ValueError('非法档案容量')
        self.archive_size=archive_size

    def run(self,diagnostic_callback=None):
        self.start(diagnostic_callback)
        self.population=self.initial()
        self.archive, mating_fitness=archive_selection(self.population,self.archive_size,True)
        self.observe(self.archive)
        while self.n_evaluations<self.FE_max:
            # 交配沿用本轮联合集计算的适应度，不在筛选后的档案上重新定义。
            f=mating_fitness
            def parent():
                a,b=self.rng.choices(range(len(f)),k=2)
                return self.archive[a if f[a]<f[b] else b if f[b]<f[a] else self.rng.choice((a,b))]
            offspring=[]
            while len(offspring)<self.population_size and self.n_evaluations<self.FE_max:
                for c in self.children(parent(),parent()):
                    if self.n_evaluations>=self.FE_max or len(offspring)>=self.population_size: break
                    child=self.evaluate(c)
                    if child is not None: offspring.append(child)
                    if len(offspring)<self.population_size and self.n_evaluations<self.FE_max: self.observe(self.archive)
            self.population=offspring
            self.archive,mating_fitness=archive_selection(self.archive+offspring,self.archive_size,True)
            self.observe(self.archive)
        return self.finish(self.archive)
