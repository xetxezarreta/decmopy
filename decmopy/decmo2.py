from typing import List, TypeVar
from jmetal.core.algorithm import Algorithm
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.replacement import (
    RankingAndDensityEstimatorReplacement,
    RemovalPolicyType,
)

S = TypeVar("S")
R = TypeVar("R")


class DECMO2(Algorithm[S, R]):
    def __init__(
        self,
        problem: Problem,
        dominance_comparator: Comparator = store.default_comparator,
        max_iterations: int = 250,
        individual_population_size: int = 100,
        report_interval: int = 100,
    ):
        super().__init__()
        self.problem = problem
        self.population_size = individual_population_size
        self.max_iterations = max_iterations
        self.report_interval = report_interval
        self.mix_interval = self.population_size / 10
        """Replacement"""
        ranking = FastNonDominatedRanking(dominance_comparator)
        density_estimator = CrowdingDistance()
        self.r = RankingAndDensityEstimatorReplacement(
            ranking, density_estimator, RemovalPolicyType.SEQUENTIAL
        )

    def run(self) -> List[S]:
        pass
