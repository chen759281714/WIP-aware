"""MPMOGA 双目标适配。AST/AUS 按任务给出的原文机制实现，非 Full 派生类。"""
from .comparison_common import ComparisonBase,deduplicate,environmental
from .comparison_operators import pox,mutate,insertion


class MPMOGA(ComparisonBase):
    algorithm_variant='MPMOGA'
    def __init__(self,*args,makespan_population_size=105,shortage_population_size=105,archive_size=210,**kwargs):
        if any(type(n) is not int or n<1 for n in (makespan_population_size,shortage_population_size,archive_size)): raise ValueError('非法种群/档案容量')
        super().__init__(*args,population_size=makespan_population_size+shortage_population_size,**kwargs)
        self.makespan_population_size=makespan_population_size
        self.shortage_population_size=shortage_population_size
        self.archive_size=archive_size
        self.diagnostics.update(ast_crossovers=0,ast_no_crossovers=0,aus_attempts=0,mutation_accepted=0)

    def ast(self,pop,objective):
        result=[]
        for index,p1 in enumerate(pop):
            if self.n_evaluations>=self.FE_max: return result+pop[index:]
            p2=self.rng.choice(self.archive)
            candidates=[p1]
            if self.rng.random()<self.crossover_probability:
                self.diagnostics['ast_crossovers']+=1
                for c in pox(p1,p2,self.operations,self.rng):
                    if self.n_evaluations>=self.FE_max: break
                    a=self.evaluate(c)
                    if a is not None: candidates.append(a)
                    # 当次竞争未结束时，真实状态仍为上一份档案。
                    if self.n_evaluations<self.FE_max: self.observe(self.archive)
            else:
                self.diagnostics['ast_no_crossovers']+=1
                candidates.append(p2)
            result.append(min(candidates,key=lambda x:x.objectives[objective]).copy())
        return result

    def mutate_population(self,pop,objective):
        result=list(pop)
        for i,a in enumerate(pop):
            if self.n_evaluations>=self.FE_max: break
            if self.rng.random()<self.mutation_probability:
                self.diagnostics['mutations']+=1
                c=self.evaluate(mutate(a,self.operations,self.rng))
                if c is not None and c.objectives[objective]<a.objectives[objective]:
                    result[i]=c; self.diagnostics['mutation_accepted']+=1
                if self.n_evaluations<self.FE_max: self.observe(self.archive)
        return result

    def aus(self):
        candidates=self.makespan_population+self.shortage_population+self.archive
        for a in list(self.archive):
            if self.n_evaluations>=self.FE_max: break
            self.diagnostics['aus_attempts']+=1
            c=self.evaluate(insertion(a,self.operations,self.rng))
            if c is not None: candidates.append(c)
            if self.n_evaluations<self.FE_max: self.observe(self.archive)
        candidates=deduplicate(candidates)
        self.archive=environmental(candidates,self.archive_size) if len(candidates)>self.archive_size else candidates
        self.observe(self.archive)

    def run(self,diagnostic_callback=None):
        self.start(diagnostic_callback)
        initial=self.initial()
        self.makespan_population=initial[:self.makespan_population_size]
        self.shortage_population=initial[self.makespan_population_size:]
        self.archive=deduplicate(initial)
        # 初始化仅去重；若显式缩小档案容量则按 AUS 规则截断。
        if len(self.archive)>self.archive_size: self.archive=environmental(self.archive,self.archive_size)
        self.observe(self.archive)
        while self.n_evaluations<self.FE_max:
            before=self.n_evaluations
            self.makespan_population=self.ast(self.makespan_population,0)
            self.shortage_population=self.ast(self.shortage_population,1)
            self.makespan_population=self.mutate_population(self.makespan_population,0)
            self.shortage_population=self.mutate_population(self.shortage_population,1)
            self.aus()  # 即使预算在 AST 中用尽，也整理已完成的部分代，不再评价。
            if self.n_evaluations==before: raise RuntimeError('未发生评价，无法推进 FE')
        return self.finish(self.archive)
