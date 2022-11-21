import random
from typing import List, TypeVar

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import Problem
from jmetal.core.quality_indicator import HyperVolume
from jmetal.core.solution import FloatSolution
from jmetal.operator import (
    BinaryTournamentSelection,
    DifferentialEvolutionCrossover,
    PolynomialMutation,
    SBXCrossover,
)
from jmetal.operator.selection import DifferentialEvolutionSelection
from jmetal.util.comparator import Comparator, DominanceComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.replacement import (
    RankingAndDensityEstimatorReplacement,
    RemovalPolicyType,
)
from jmetal.util.solution import read_solutions

S = TypeVar("S")
R = TypeVar("R")


class DECMO_FLOAT(Algorithm[S, R]):
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
        pool_1_size = self.population_size
        pool_2_size = self.population_size

        selection_operator_1 = BinaryTournamentSelection()
        crossover_operator_1 = SBXCrossover(1.0, 20.0)
        mutation_operator_1 = PolynomialMutation(
            1.0 / self.problem.number_of_variables, 20.0
        )
        selection_operator_2 = DifferentialEvolutionSelection()
        crossover_operator_2 = DifferentialEvolutionCrossover(0.2, 0.5, 0.5)

        dominance = DominanceComparator()

        max_iterations = self.max_iterations
        iterations = 0

        parent_1: List[FloatSolution] = [None, None]

        generational_hv: List[float] = []

        current_gen = 0

        """Create the initial subpopulation pools and evaluate them"""
        pool_1: List[FloatSolution] = []
        for i in range(pool_1_size):
            pool_1.append(self.problem.create_solution())
            pool_1[i] = self.problem.evaluate(pool_1[i])

        pool_2: List[FloatSolution] = []
        for i in range(pool_2_size):
            pool_2.append(self.problem.create_solution())
            pool_2[i] = self.problem.evaluate(pool_2[i])

        evaluations = pool_1_size + pool_2_size

        mix = self.mix_interval

        problem = self.problem
        # problem.reference_front = read_solutions(
        #    filename="./resources/" + problem.get_name() + ".3D.pf"
        # )

        # h = HyperVolume(reference_point=[1, 1, 1])
        h = HyperVolume(reference_point=[1] * self.problem.number_of_objectives)

        initial_population = True

        """The main evolutionary cycle"""
        while iterations < max_iterations:
            combi: List[FloatSolution] = []
            if not initial_population:
                offspring_pop_1: List[FloatSolution] = []
                offspring_pop_2: List[FloatSolution] = []
                """Evolve pool 1"""
                for i in range(pool_1_size):
                    parent_1[0] = selection_operator_1.execute(pool_1)
                    parent_1[1] = selection_operator_1.execute(pool_1)

                    child_1: FloatSolution = crossover_operator_1.execute(parent_1)[0]
                    child_1 = mutation_operator_1.execute(child_1)

                    child_1 = problem.evaluate(child_1)
                    evaluations += 1

                    offspring_pop_1.append(child_1)
                """Evolve pool 2"""
                for i in range(pool_2_size):
                    parent_2: List[FloatSolution] = selection_operator_2.execute(pool_2)

                    crossover_operator_2.current_individual = pool_2[i]
                    child_2 = crossover_operator_2.execute(parent_2)
                    child_2 = problem.evaluate(child_2[0])

                    evaluations += 1

                    result = dominance.compare(pool_2[i], child_2)

                    if result == -1:
                        offspring_pop_2.append(pool_2[i])
                    elif result == 1:
                        offspring_pop_2.append(child_2)
                    else:
                        offspring_pop_2.append(child_2)
                        offspring_pop_2.append(pool_2[i])

                ind_1 = pool_1[random.randint(0, pool_1_size - 1)]
                ind_2 = pool_2[random.randint(0, pool_2_size - 1)]

                offspring_pop_1.append(ind_1)
                offspring_pop_2.append(ind_2)

                offspring_pop_1.extend(pool_1)
                pool_1 = self.r.replace(
                    offspring_pop_1[:pool_1_size], offspring_pop_1[pool_1_size:]
                )

                pool_2 = self.r.replace(
                    offspring_pop_2[:pool_2_size], offspring_pop_2[pool_2_size:]
                )

                mix -= 1
                if mix == 0:
                    """Time to perform fitness sharing"""
                    mix = self.mix_interval
                    combi = combi + pool_1 + pool_2
                    print("Combi size: ", len(combi))
                    """pool1size/10"""

                    combi = self.r.replace(
                        combi[: int(pool_1_size / 10)],
                        combi[int(pool_1_size / 10) : len(combi)],
                    )

                    print(
                        "Sizes: ",
                        len(pool_1) + len(combi),
                        len(pool_2) + len(combi),
                        "\n",
                    )

                    pool_1 = self.r.replace(pool_1, combi)

                    pool_2 = self.r.replace(pool_2, combi)

            if initial_population:
                initial_population = False

            iterations += 1

            hval_1 = h.compute([s.objectives for s in pool_1])
            hval_2 = h.compute([s.objectives for s in pool_2])
            print("Iterations: ", str(iterations))
            print("hval_1: ", str(hval_1))
            print("hval_2: ", str(hval_2), "\n")

            new_gen = int(evaluations / self.report_interval)
            if new_gen > current_gen:
                combi = combi + pool_1 + pool_2

                combi = self.r.replace(
                    combi[: (2 * pool_1_size)], combi[(2 * pool_1_size) :]
                )

                hval = h.compute([s.objectives for s in combi])
                for i in range(current_gen, new_gen, 1):
                    generational_hv.append(hval)

                current_gen = new_gen

        """#Write runtime generational HV to file"""

        """Return the first non dominated front"""
        combi_ini: List[FloatSolution] = []
        combi_ini.extend(pool_1)
        combi_ini.extend(pool_2)
        combi_ini = self.r.replace(
            combi_ini[: pool_1_size + pool_2_size],
            combi_ini[pool_1_size + pool_2_size :],
        )
        return combi_ini

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "DECMO"

    def create_initial_solutions(self) -> List[S]:
        pass

    def evaluate(self, FloatSolutions: List[S]) -> List[S]:
        pass

    def stopping_condition_is_met(self) -> bool:
        pass

    def get_observable_data(self) -> dict:
        pass

    def init_progress(self) -> None:
        pass

    def step(self) -> None:
        pass

    def update_progress(self):
        pass
