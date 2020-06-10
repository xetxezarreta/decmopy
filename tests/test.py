import sys

sys.path.append("../decmopy")

from decmopy.decmo import DECMO

from jmetal.problem import ZDT1
from jmetal.util.solution import read_solutions


def main():
    problem = ZDT1()
    problem.reference_front = read_solutions(filename="./resources/ZDT1.pf")

    algorithm = DECMO(problem)
    algorithm.run()
    front = algorithm.get_result()

    print(f"Algorithm: ${algorithm.get_name()}")
    print(f"Problem: ${problem.get_name()}")
    print(f"Computing time: ${algorithm.total_computing_time}")
    print(front)


if __name__ == "__main__":
    main()
