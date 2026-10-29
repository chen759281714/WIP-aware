"""双目标离散 MOEA/D。
核心分解框架来自 Zhang & Li (2007)；局部/全局模式和限次替换借鉴
后续 MOEA/D-DE (2009)。本项目使用离散 POX/MS，而非实数 DE。
δ=.9、nr=2 为当前工程约定，不是2007论文统一规定的参数。
"""
import math
from .comparison_common import ComparisonBase


class MOEAD(ComparisonBase):
    algorithm_variant='MOEAD'
    def __init__(self,*args,neighborhood_size=20,neighborhood_mating_probability=.9,max_replacements=2,**kwargs):
        super().__init__(*args,**kwargs)
        if self.population_size<2 or type(neighborhood_size) is not int or neighborhood_size<2 or type(max_replacements) is not int or max_replacements<1: raise ValueError('非法分解参数')
        if not 0<=neighborhood_mating_probability<=1: raise ValueError('非法邻域交配概率')
        self.neighborhood_size=min(neighborhood_size,self.population_size)
        self.neighborhood_mating_probability=neighborhood_mating_probability
        self.max_replacements=max_replacements
        n=self.population_size
        self.weights=[(i/(n-1),1-i/(n-1)) for i in range(n)]
        self.neighborhoods=[sorted(range(n),key=lambda j:(math.dist(w,self.weights[j]),j))[:self.neighborhood_size] for w in self.weights]
        self.ideal=[math.inf]*2; self.upper=[-math.inf]*2
        self.diagnostics['replacements']=0; self.diagnostics['replacement_counts']=[]
        self.diagnostics.update(candidates=0,local_updates=0,global_updates=0)

    def update_bounds(self,a):
        for j,v in enumerate(a.objectives):
            self.ideal[j]=min(self.ideal[j],v); self.upper[j]=max(self.upper[j],v)

    def scalarize(self,a,w):
        # 零权重替换为 1e-6；零跨度采用分母 1。两解比较共用当前尺度。
        return max(max(w[j],1e-6)*abs(a.objectives[j]-self.ideal[j])/(self.upper[j]-self.ideal[j] or 1.) for j in range(2))

    def run(self,diagnostic_callback=None):
        self.start(diagnostic_callback)
        self.population=self.initial()
        for a in self.population: self.update_bounds(a)
        while self.n_evaluations<self.FE_max:
            for i in range(self.population_size):
                local=self.rng.random()<self.neighborhood_mating_probability
                scope=self.neighborhoods[i] if local else list(range(self.population_size))
                self.diagnostics['local_updates' if local else 'global_updates']+=1
                parents=self.rng.sample(scope,2)
                # POX 只取第一个子代进行一次实际评价。
                c=self.evaluate(self.generate_single_offspring(*(self.population[k] for k in parents)))
                self.diagnostics['candidates']+=1
                count=0
                if c is not None:
                    self.update_bounds(c)
                    order=list(scope); self.rng.shuffle(order)
                    for j in order:
                        if self.scalarize(c,self.weights[j])<=self.scalarize(self.population[j],self.weights[j]):
                            self.population[j]=c.copy(); count+=1
                            if count>=self.max_replacements: break
                self.diagnostics['replacements']+=count; self.diagnostics['replacement_counts'].append(count)
                self.observe(self.population)
                if self.n_evaluations>=self.FE_max: break
        return self.finish(self.population)
