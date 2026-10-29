"""标准 NSGA-II；默认参数为本项目工程配置。"""
from .comparison_common import ComparisonBase,fronts,crowding,environmental


class NSGAII(ComparisonBase):
    algorithm_variant='NSGA_II'

    def tournament(self,pop,ranks,distances):
        a,b=self.rng.choices(range(len(pop)),k=2)
        ka,kb=(ranks[a],-distances[a]),(ranks[b],-distances[b])
        return pop[a if ka<kb else b if kb<ka else self.rng.choice((a,b))]

    def run(self,diagnostic_callback=None):
        self.start(diagnostic_callback)
        self.population=self.initial()
        while self.n_evaluations<self.FE_max:
            ranks={}; distances={}
            for rank,layer in enumerate(fronts(self.population)):
                ranks.update({i:rank for i in layer}); distances.update(crowding(self.population,layer))
            offspring=[]
            # 中间检查点报告尚未提交的上一代；预算结束后做部分代环境选择。
            while len(offspring)<self.population_size and self.n_evaluations<self.FE_max:
                pair=self.children(*(self.tournament(self.population,ranks,distances) for _ in range(2)))
                for c in pair:
                    if self.n_evaluations>=self.FE_max or len(offspring)>=self.population_size: break
                    child=self.evaluate(c)
                    if child is not None: offspring.append(child)
                    if len(offspring)<self.population_size and self.n_evaluations<self.FE_max: self.observe(self.population)
            self.population=environmental(self.population+offspring,self.population_size)
            self.observe(self.population)
        return self.finish(self.population)
