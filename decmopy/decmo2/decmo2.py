import sys, math, operator, random
import numpy as np
from typing import List, TypeVar
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.comparator import Comparator, DominanceComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.core.quality_indicator import HyperVolume
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

from .drec import DirectionRec
from .dgen import DistribGen
from .crec import CompRec


S = TypeVar("S")
R = TypeVar("R")


class DECMO2(Algorithm[S, R]):
    def __init__(
        self,
        problem: Problem,
        dominance_comparator: Comparator = store.default_comparator,
        max_evaluations: int = 250,
        individual_population_size: int = 100,
        report_interval: int = 100,
        dataDirectory: str = "./decmopy/decmo2/weigths",
    ):
        super().__init__()
        self.problem = problem
        self.population_size = individual_population_size
        self.max_evaluations = max_evaluations
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
        # dominance comparator
        dominance = DominanceComparator()

        # array that stores the "generational" HV quality
        generational_hv: List[float] = []

        parent_1: List[FloatSolution] = [None, None]
        parent_2: List[FloatSolution] = []
        parent_3: List[FloatSolution] = []

        # initialize some local and global variables
        pool_1: List[FloatSolution] = []
        pool_2: List[FloatSolution] = []

        # size of elite subset used for fitness sharing between subpopulations
        nrOfDirectionalSolutionsToEvolve = int(self.population_size / 5)
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

        mix = self.mix_interval
        h = HyperVolume(reference_point=[1] * self.problem.number_of_objectives)

        insertionRate: List[float] = [0, 0, 0]
        bonusEvals: List[int] = [0, 0, nrOfDirectionalSolutionsToEvolve]
        testRun = True

        # record the generational HV of the initial population
        combiAll: List[FloatSolution] = []
        cGen = int(evaluations / self.report_interval)
        if cGen > 0:
            combiAll = pool_1 + pool_2 + pool_A
            combiAll = self.r.replace(
                combiAll[: pool_1_size + pool_2_size],
                combiAll[pool_1_size + pool_2_size :],
            )
            hval = h.compute([s.objectives for s in combiAll])
            for _ in range(cGen):
                generational_hv.append(hval)
            current_gen = cGen

        # the main loop of the algorithm
        while evaluations < self.max_evaluations:
            offspringPop1: List[FloatSolution] = []
            offspringPop2: List[FloatSolution] = []
            offspringPop3: List[FloatSolution] = []

            dirInsertPool1: List[FloatSolution] = []
            dirInsertPool2: List[FloatSolution] = []
            dirInsertPool3: List[FloatSolution] = []

            # evolve pool1 - using SPEA2 evolutionary model
            nfe: int = 0
            while nfe < (pool_1_size + bonusEvals[0]):
                parent_1[0] = selection_operator_1.execute(pool_1)
                parent_1[1] = selection_operator_1.execute(pool_1)

                child1a: FloatSolution = crossover_operator_1.execute(parent_1)[0]
                child1a = mutation_operator_1.execute(child1a)

                child1a = self.problem.evaluate(child1a)
                evaluations += 1
                nfe += 1

                offspringPop1.append(child1a)
                dirInsertPool1.append(child1a)

            # evolve pool2 - using DEMO SP evolutionary model
            i: int = 0
            unselectedIDs: List[int] = []
            for ID in range(len(pool_2)):
                unselectedIDs.append(ID)

            nfe = 0
            while nfe < (pool_2_size + bonusEvals[1]):
                index = random.randint(0, len(unselectedIDs) - 1)
                i = unselectedIDs[index]
                unselectedIDs.pop(index)

                parent_2 = selection_operator_2.execute(pool_2)

                crossover_operator_2.current_individual = pool_2[i]
                child2 = crossover_operator_2.execute(parent_2)
                child2 = self.problem.evaluate(child2[0])

                evaluations += 1
                nfe += 1

                result = dominance.compare(pool_2[i], child2)

                if result == -1:  # solution i dominates child
                    offspringPop2.append(pool_2[i])
                elif result == 1:  # child dominates
                    offspringPop2.append(child2)
                else:  # the two solutions are non-dominated
                    offspringPop2.append(child2)
                    offspringPop2.append(pool_2[i])

                dirInsertPool2.append(child2)

                if len(unselectedIDs) == 0:
                    for ID in range(len(pool_2)):
                        unselectedIDs.append(random.randint(0, len(pool_2) - 1))

            # evolve pool3 - Directional Decomposition DE/rand/1/bin
            IDs = self.__compute_neighbourhood_Nfe_since_last_update(
                neighbourhoods, directionalArchive, nrOfDirectionalSolutionsToEvolve
            )

            nfe = 0
            for j in range(len(IDs)):
                if nfe < bonusEvals[2]:
                    nfe += 1
                else:
                    break

                cID = IDs[j]

                chosenSol: FloatSolution = None
                if directionalArchive[cID].curr_sol != None:
                    chosenSol = directionalArchive[cID].curr_sol
                else:
                    chosenSol = pool_1[0]
                    print("error!")

                parent_3: List[FloatSolution] = [None, None, None]

                r1 = random.randint(0, len(neighbourhoods[cID]) - 1)
                r2 = random.randint(0, len(neighbourhoods[cID]) - 1)
                r3 = random.randint(0, len(neighbourhoods[cID]) - 1)
                while r2 == r1:
                    r2 = random.randint(0, len(neighbourhoods[cID]) - 1)
                while r3 == r1 or r3 == r2:
                    r3 = random.randint(0, len(neighbourhoods[cID]) - 1)

                parent_3[0] = directionalArchive[r1].curr_sol
                parent_3[1] = directionalArchive[r2].curr_sol
                parent_3[2] = directionalArchive[r3].curr_sol

                crossover_operator_3.current_individual = chosenSol
                child3 = crossover_operator_3.execute(parent_3)[0]
                child3 = mutation_operator_1.execute(child3)

                child3 = self.problem.evaluate(child3)
                evaluations += 1

                dirInsertPool3.append(child3)

            # compute directional improvements
            # pool1
            improvements = 0
            for j in range(len(dirInsertPool1)):
                testSol = dirInsertPool1[j]
                self.__update_extreme_values(testSol)
                improvements += self.__update_neighbourhoods(
                    directionalArchive, testSol, nrOfReplacements
                )
            insertionRate[0] += (1.0 * improvements) / len(dirInsertPool1)

            # pool2
            improvements = 0
            for j in range(len(dirInsertPool2)):
                testSol = dirInsertPool2[j]
                self.__update_extreme_values(testSol)
                improvements += self.__update_neighbourhoods(
                    directionalArchive, testSol, nrOfReplacements
                )
            insertionRate[1] += (1.0 * improvements) / len(dirInsertPool2)

            # pool3
            improvements = 0
            for j in range(len(dirInsertPool3)):
                testSol = dirInsertPool3[j]
                self.__update_extreme_values(testSol)
                improvements += self.__update_neighbourhoods(
                    directionalArchive, testSol, nrOfReplacements
                )
            # on java, dividing a floating number by 0, returns NaN
            # on python, dividing a floating number by 0, returns an exception
            if len(dirInsertPool3) == 0:
                insertionRate[2] = None
            else:
                insertionRate[2] += (1.0 * improvements) / len(dirInsertPool3)

            for dr in directionalArchive:
                offspringPop3.append(dr.curr_sol)

            offspringPop1 = offspringPop1 + pool_1
            pool_1 = self.r.replace(
                offspringPop1[:pool_1_size], offspringPop1[pool_1_size:]
            )
            pool_2 = self.r.replace(
                offspringPop2[:pool_2_size], offspringPop2[pool_2_size:]
            )

            combi: List[FloatSolution] = []
            mix -= 1

            if mix == 0:
                mix = self.mix_interval
                combi = combi + pool_1 + pool_2 + offspringPop3
                print("Combi size: " + str(len(combi)))

                combi = self.r.replace(
                    combi[:nrOfDirectionalSolutionsToEvolve],
                    combi[nrOfDirectionalSolutionsToEvolve:],
                )

                insertionRate[0] /= self.mix_interval
                insertionRate[1] /= self.mix_interval
                if insertionRate[2] != None:
                    insertionRate[2] /= self.mix_interval

                """
                print(
                    "Insertion rates: "
                    + str(insertionRate[0])
                    + " - "
                    + str(insertionRate[1])
                    + " - "
                    + str(insertionRate[2])
                    + " - Test run:"
                    + str(testRun)
                )
                """
                if testRun:
                    if (insertionRate[0] > insertionRate[1]) and (
                        insertionRate[0] > insertionRate[2]
                    ):
                        print("SPEA2 win - bonus run!")
                        bonusEvals[0] = nrOfDirectionalSolutionsToEvolve
                        bonusEvals[1] = 0
                        bonusEvals[2] = 0
                    if (insertionRate[1] > insertionRate[0]) and (
                        insertionRate[1] > insertionRate[2]
                    ):
                        print("DE win - bonus run!")
                        bonusEvals[0] = 0
                        bonusEvals[1] = nrOfDirectionalSolutionsToEvolve
                        bonusEvals[2] = 0
                    if (insertionRate[2] > insertionRate[0]) and (
                        insertionRate[2] > insertionRate[1]
                    ):
                        print("Directional win - no bonus!")
                        bonusEvals[0] = 0
                        bonusEvals[1] = 0
                        bonusEvals[2] = nrOfDirectionalSolutionsToEvolve
                else:
                    print("Test run - no bonus!")
                    bonusEvals[0] = 0
                    bonusEvals[1] = 0
                    bonusEvals[2] = nrOfDirectionalSolutionsToEvolve

                testRun = not testRun

                insertionRate[0] = 0.0
                insertionRate[1] = 0.0
                insertionRate[2] = 0.0

                pool_1 = pool_1 + combi
                pool_2 = pool_2 + combi
                print("Sizes: " + str(len(pool_1)) + " " + str(len(pool_2)))

                pool_1 = self.r.replace(pool_1[:pool_1_size], pool_1[pool_1_size:])
                pool_2 = self.r.replace(pool_2[:pool_2_size], pool_2[pool_2_size:])

                self.__clear_Nfe_history(directionalArchive)

            hVal1 = h.compute([s.objectives for s in pool_1])
            hVal2 = h.compute([s.objectives for s in pool_2])
            hVal3 = h.compute([s.objectives for s in offspringPop3])

            newGen = int(evaluations / self.report_interval)

            if newGen > current_gen:
                print(
                    "Hypervolume: "
                    + str(newGen)
                    + " - "
                    + str(hVal1)
                    + " - "
                    + str(hVal2)
                    + " - "
                    + str(hVal3)
                )
                combi = combi + pool_1 + pool_2 + offspringPop3
                combi = self.r.replace(
                    combi[: self.population_size * 2], combi[self.population_size * 2 :]
                )
                hval = h.compute([s.objectives for s in combi])
                for j in range(current_gen, newGen):
                    generational_hv.append(hval)
                current_gen = newGen

        # return the final combined non-dominated set of maximum size = (populationSize * 2)
        combiAll: List[FloatSolution] = []
        combiAll = combiAll + pool_1 + pool_2 + pool_A
        combiAll = self.r.replace(
            combiAll[: self.population_size * 2], combiAll[self.population_size * 2 :]
        )
        return combiAll

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
            dataFileName = (
                "W" + str(nrOfObjectives) + "D_" + str(dirArchiveSize) + ".dat"
            )
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

    def __compute_neighbourhood_Nfe_since_last_update(
        self,
        neighbourhoods: List[List[int]],
        directionalArchive: List[DirectionRec],
        intensificationClusters: int,
    ):
        averageNfe: List[CompRec] = []
        ID = 0

        for neighbourhood in neighbourhoods:
            avg = 0.0
            for nID in neighbourhood:
                avg += directionalArchive[nID].nfeSinceLastUpdate
            avg /= len(neighbourhood)

            averageNfe.append(CompRec(ID, avg))
            ID += 1
        averageNfe.sort()

        result: List[int] = []
        for i in range(intensificationClusters):
            result.append(averageNfe[len(averageNfe) - 1 - i].id)
        return result

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

    def __update_neighbourhoods(
        self,
        directionalArchive: List[DirectionRec],
        newSolution: FloatSolution,
        nrOfReplacements: int,
    ):
        improvedDistances: List[CompRec] = []
        isImprovement = False

        for cdr in directionalArchive:
            newFitnessValue = self.__evaluate_Tchebycheff_Fitness(
                newSolution, cdr.weigh_vector
            )
            if newFitnessValue < cdr.fitness_value:
                improvedDistances.append(CompRec(cdr.id, newFitnessValue))
                isImprovement = True
            else:
                cdr.nfeSinceLastUpdate = cdr.nfeSinceLastUpdate + 1

        improvedDistances.sort()
        improvedDistances.reverse()

        if isImprovement:
            for _ in range(nrOfReplacements):
                j = 0
                cdr = directionalArchive[improvedDistances[j].id]
                cdr.curr_sol = newSolution
                cdr.fitness_value = improvedDistances[j].value
                cdr.nfeSinceLastUpdate = 0
            return 1
        return 0

    def __evaluate_Tchebycheff_Fitness(
        self, individual: FloatSolution, lmbd: List[float]
    ):
        max = sys.float_info.min

        for i in range(self.problem.number_of_objectives):
            diff = abs(
                individual.objectives[i] - self.extreme_values[self.MIN_VALUES][i]
            )
            tcheFuncVal: float = None

            if lmbd[i] == 0:
                tcheFuncVal = 0.000001 * diff
            else:
                tcheFuncVal = diff * lmbd[i]

            if tcheFuncVal > max:
                max = tcheFuncVal

        return max

    def __update_extreme_values(self, sol: FloatSolution):
        for i in range(self.problem.number_of_objectives):
            objValue = sol.objectives[i]
            if objValue < self.extreme_values[self.MIN_VALUES][i]:
                self.extreme_values[self.MIN_VALUES][i] = objValue
            if objValue > self.extreme_values[self.MAX_VALUES][i]:
                self.extreme_values[self.MAX_VALUES][i] = objValue

    def __clear_Nfe_history(self, directionalArchive: List[DirectionRec]):
        for dr in directionalArchive:
            dr.nfeSinceLastUpdate = 0

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
