from typing import List, TypeVar
from jmetal.core.algorithm import Algorithm
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.replacement import (
    RankingAndDensityEstimatorReplacement,
    RemovalPolicyType,
)
from jmetal.operator import (
    BinaryTournamentSelection,
    DifferentialEvolutionCrossover,
    PolynomialMutation,
    SBXCrossover,
)
from jmetal.operator.selection import DifferentialEvolutionSelection

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
        self.mix_interval = 1
        """Replacement"""
        ranking = FastNonDominatedRanking(dominance_comparator)
        density_estimator = CrowdingDistance()
        self.r = RankingAndDensityEstimatorReplacement(
            ranking, density_estimator, RemovalPolicyType.SEQUENTIAL
        )

    def run(self) -> List[S]:
        # Selection operator 1
        selection_operator_1 = BinaryTournamentSelection()
        # Selection operator 2
        selection_operator_2 = DifferentialEvolutionSelection()
        # Crossover operator 1
        crossover_operator_1 = SBXCrossover(1.0, 20.0)
        # Crossover operator 2
        crossover_operator_2 = DifferentialEvolutionCrossover(0.2, 0.5, 0.5)
        # Crossover operator 3
        crossover_operator_3 = DifferentialEvolutionCrossover(1.0, 0.5, 0.5)
        # Mutation operator 1
        mutation_operator_1 = PolynomialMutation(
            1.0 / self.problem.number_of_variables, 20.0
        )
        # Array that stores the "generational" HV quality
        generational_hv: List[float] = []

        # Initialize some local and global variables
        pool_1: List[FloatSolution] = []
        pool_2: List[FloatSolution] = []
        current_gen = 0

        # size of elite subset used for fitness sharing between subpopulations
        nrOfDirectionalSolutionsToEvolve = self.population_size / 5
        pool_1_size = self.population_size - (nrOfDirectionalSolutionsToEvolve / 2)
        pool_2_size = self.population_size - (nrOfDirectionalSolutionsToEvolve / 2)
        print(
            str(pool_1_size)
            + " - "
            + str(nrOfDirectionalSolutionsToEvolve)
            + " - "
            + str(self.mix_interval)
        )

        # Create the initial pools
        # pool1
        pool_1: List[FloatSolution] = []
        for i in range(pool_1_size):
            pool_1.append(self.problem.create_solution())
            pool_1[i] = self.problem.evaluate(pool_1[i])
        # pool2
        pool_2: List[FloatSolution] = []
        for i in range(pool_2_size):
            pool_2.append(self.problem.create_solution())
            pool_2[i] = self.problem.evaluate(pool_2[i])

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "DECMO"
