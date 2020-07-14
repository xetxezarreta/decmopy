from jmetal.problem import ZDT1
from jmetal.problem.multiobjective import dtlz, zdt
from jmetal.util.solution import read_solutions

from decmo2 import DECMO2



def main():
    problems = [
        #dtlz.DTLZ1()
        dtlz.DTLZ7()
    ]

    for problem in problems:
        algorithm = DECMO2(problem, max_iterations=50000)
        algorithm.run()
        front = algorithm.get_result()
        print(f"Algorithm: ${algorithm.get_name()}")
        print(f"Problem: ${problem.get_name()}")
        print(f"Computing time: ${algorithm.total_computing_time}")
        print(f"Final non-dominted solution set size: ${len(front)}")


if __name__ == "__main__":
    main()
