from jmetal.problem import ZDT1
from jmetal.problem.multiobjective import dtlz, zdt
from jmetal.util.solution import read_solutions

from decmopy import DECMO_FLOAT

"""
def create_problem(problem):
    p = problem()
    p.reference_front = read_solutions(
        filename="./decmopy/resources/" + p.__class__.__name__ + ".pf"
    )
    return p
"""


def main():
    problems = [
        # ZTD
        # create_problem(zdt.ZDT1)
        zdt.ZDT1()
    ]

    for problem in problems:
        algorithm = DECMO_FLOAT(problem, max_iterations=250)
        result = algorithm.run()
        print(f"Algorithm: ${algorithm.get_name()}")
        print(f"Problem: ${problem.get_name()}")
        # print(f"Computing time: ${algorithm.total_computing_time}")
        print(f"Final non-dominted solution set size: ${len(result)}")


if __name__ == "__main__":
    main()
