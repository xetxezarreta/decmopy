from jmetal.problem import ZDT1
from jmetal.problem.multiobjective import dtlz, zdt
from jmetal.util.solution import read_solutions

from decmopy.decmo2 import DECMO2


def main():
    problems = [
        # dtlz.DTLZ1()
        # dtlz.DTLZ3()
        zdt.ZDT1()
    ]

    for problem in problems:
        algorithm = DECMO2(problem, max_evaluations=25000)
        result = algorithm.run()
        print(f"Algorithm: ${algorithm.get_name()}")
        print(f"Problem: ${problem.get_name()}")
        # print(f"Computing time: ${algorithm.total_computing_time}")
        print(f"Final non-dominted solution set size: ${len(result)}")


if __name__ == "__main__":
    main()
