from jmetal.problem import ZDT1
from jmetal.problem.multiobjective import dtlz, zdt
from jmetal.util.solution import read_solutions

from decmo2 import DECMO2


def create_problem(problem):
    p = problem()
    p.reference_front = read_solutions(
        filename="./resources/" + p.__class__.__name__ + ".pf"
    )
    return p


def main():
    problems = [
        # ZTD
        create_problem(zdt.ZDT1)
    ]

    for problem in problems:
        algorithm = DECMO2(problem)
        algorithm.run()
        front = algorithm.get_result()
        print(f"Algorithm: ${algorithm.get_name()}")
        print(f"Problem: ${problem.get_name()}")
        print(f"Computing time: ${algorithm.total_computing_time}")


if __name__ == "__main__":
    main()
