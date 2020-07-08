import sys, math, operator
import numpy as np
from typing import List, TypeVar
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.comparator import Comparator
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

from direction_rec import DirectionRec
from distribution_gen import DistribGen
from comp_rec import CompRec


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
        dataDirectory: str = "./weigths",
    ):
        super().__init__()
        self.problem = problem
        self.population_size = individual_population_size
        self.max_iterations = max_iterations
        self.report_interval = report_interval
        self.dataDirectory = dataDirectory
        self.mix_interval = 1
        """Replacement"""
        ranking = FastNonDominatedRanking(dominance_comparator)
        density_estimator = CrowdingDistance()
        self.r = RankingAndDensityEstimatorReplacement(
            ranking, density_estimator, RemovalPolicyType.SEQUENTIAL
        )

        self.MIN_VALUES = 0
        self.MAX_VALUES = 1
        min_values: List[float] = []
        max_values: List[float] = []
        for _ in range(problem.number_of_objectives):
            min_values.append(sys.float_info.max)
            max_values.append(sys.float_info.min)
        self.extreme_values: List[List[float]] = []
        self.extreme_values.append(min_values)
        self.extreme_values.append(max_values)

    def __compute_euclidean_distance(self, vector1: List[float], vector2: List[float]):
        value = 0.0
        for i in range(len(vector1)):
            value += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i])
        return math.sqrt(value)

    def __create_uniform_weights(self, dirArchiveSize: int, nrOfObjectives: int):
        lmdb = np.zeros(shape=(dirArchiveSize, nrOfObjectives))

        if nrOfObjectives == 2 and dirArchiveSize < 500:
            for n in range(dirArchiveSize):
                a = 1.0 * n / (dirArchiveSize - 1)
                lmdb[n][0] = a
                lmdb[n][1] = 1 - a
                print(lmdb[n][0])
                print(lmdb[n][1])
        else:
            dataFileName = "W" + nrOfObjectives + "D_" + dirArchiveSize + ".dat"
            data_path = self.dataDirectory + "/" + dataFileName
            print(dataFileName)
            print(data_path)

            dg = DistribGen()
            dg.create_distribution(
                self.problem.number_of_objectives, dirArchiveSize, data_path
            )

            try:
                i = j = 0
                with open(data_path) as f:
                    # lines = f.readlines()
                    lines = [line.rstrip() for line in f]

                for line in lines:
                    words = line.split()  # "tokenizer"
                    j = 0
                    for word in words:
                        value = float(word.replace(",", "."))
                        lmdb[i][j] = value
                        j += 1
                    i += 1
            except Exception as e:
                print(e)
                print("initUniformWeight: failed when reading for file: " + data_path)
        return lmdb

    def __create_directional_archive(self, lmbd: List[float]):
        directionalArchive: List[DirectionRec] = []
        for i in range(len(lmbd)):
            di = DirectionRec(i, lmbd[i], None, sys.float_info.max, 0)
            directionalArchive.append(di)
        return directionalArchive

    def __create_neighbourhoods(
        self, dirArchive: List[DirectionRec], neighborhood_size: int
    ):
        neighbourhoods: List[List[int]] = []

        for di1 in dirArchive:
            distToNeighbour: List[CompRec] = []
            for di2 in dirArchive:
                if di1.id != di2.id:
                    distToNeighbour.append(
                        CompRec(
                            di2.id,
                            self.__compute_euclidean_distance(
                                di1.weigh_vector, di2.weigh_vector
                            ),
                        )
                    )
            distToNeighbour.sort()
            neighbourhood: List[int] = []
            for i in range(neighborhood_size):
                if i < len(distToNeighbour):
                    neighbourhood.append(distToNeighbour[i].id)
            neighbourhoods.append(neighbourhood)
        return neighbourhoods

    def __update_extreme_values(self, sol: FloatSolution):
        for i in range(sol.number_of_objectives):
            objValue = sol.objectives[i]
            if objValue < self.extreme_values[self.MIN_VALUES][i]:
                self.extreme_values[self.MIN_VALUES][i] = objValue
            if objValue > self.extreme_values[self.MAX_VALUES][i]:
                self.extreme_values[self.MAX_VALUES][i] = objValue

    def run(self) -> List[S]:
        # selection operator 1
        selection_operator_1 = BinaryTournamentSelection()
        # selection operator 2
        selection_operator_2 = DifferentialEvolutionSelection()
        # crossover operator 1
        crossover_operator_1 = SBXCrossover(1.0, 20.0)
        # crossover operator 2
        crossover_operator_2 = DifferentialEvolutionCrossover(0.2, 0.5, 0.5)
        # crossover operator 3
        crossover_operator_3 = DifferentialEvolutionCrossover(1.0, 0.5, 0.5)
        # mutation operator 1
        mutation_operator_1 = PolynomialMutation(
            1.0 / self.problem.number_of_variables, 20.0
        )
        # array that stores the "generational" HV quality
        generational_hv: List[float] = []

        # initialize some local and global variables
        pool_1: List[FloatSolution] = []
        pool_2: List[FloatSolution] = []

        # size of elite subset used for fitness sharing between subpopulations
        nrOfDirectionalSolutionsToEvolve = self.population_size / 5
        # subpopulation 1
        pool_1_size = int(self.population_size - (nrOfDirectionalSolutionsToEvolve / 2))
        # subpopulation 2
        pool_2_size = int(self.population_size - (nrOfDirectionalSolutionsToEvolve / 2))

        print(
            str(pool_1_size)
            + " - "
            + str(nrOfDirectionalSolutionsToEvolve)
            + " - "
            + str(self.mix_interval)
        )

        evaluations = 0
        current_gen = 0
        directionalArchiveSize = 2 * self.population_size
        weights = self.__create_uniform_weights(
            directionalArchiveSize, self.problem.number_of_objectives
        )

        directionalArchive = self.__create_directional_archive(weights)
        neighbourhoods = self.__create_neighbourhoods(
            directionalArchive, self.population_size
        )

        nrOfReplacements = 1
        iniID = 0

        # Create the initial pools
        # pool1
        pool_1: List[FloatSolution] = []
        for _ in range(pool_1_size):
            new_solution = self.problem.create_solution()
            new_solution = self.problem.evaluate(new_solution)
            evaluations += 1
            pool_1.append(new_solution)

            self.__update_extreme_values(new_solution)
            dr = directionalArchive[iniID]
            dr.curr_sol = new_solution
            iniID += 1
        # pool2
        pool_2: List[FloatSolution] = []
        for _ in range(pool_2_size):
            new_solution = self.problem.create_solution()
            new_solution = self.problem.evaluate(new_solution)
            evaluations += 1
            pool_2.append(new_solution)

            self.__update_extreme_values(new_solution)
            dr = directionalArchive[iniID]
            dr.curr_sol = new_solution
            iniID += 1
        # directional archive initialization
        pool_A: List[FloatSolution] = []
        while iniID < directionalArchiveSize:
            new_solution = self.problem.create_solution()
            new_solution = self.problem.evaluate(new_solution)
            evaluations += 1
            pool_A.append(new_solution)

            self.__update_extreme_values(new_solution)
            dr = directionalArchive[iniID]
            dr.curr_sol = new_solution
            iniID += 1

        # continue here
        return 0

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "DECMO"

    def create_initial_solutions(self) -> List[S]:
        pass

    def evaluate(self, solutions: List[S]) -> List[S]:
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
