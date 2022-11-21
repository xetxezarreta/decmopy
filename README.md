# DECMOPY

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

Python implementation of DECMO algorithms inside the JMetalPy 1.5.5 framework.

## Installation
```bash
pip install decmopy
```

## DECMO
```python
from jmetal.problem import ZDT1
from decmopy import DECMO_FLOAT

def main():
    problem = ZDT1()

    algorithm = DECMO_FLOAT(problem, max_iterations=250)
    result = algorithm.run()
    print(f"Algorithm: ${algorithm.get_name()}")
    print(f"Problem: ${problem.get_name()}")
    print(f"Final non-dominted solution set size: ${len(result)}")

if __name__ == "__main__":
    main()
```

## DECMO2
```python
from jmetal.problem import ZDT1
from decmopy import DECMO2

def main():
    problem = ZDT1()

    algorithm = DECMO2(problem, max_iterations=250)
    result = algorithm.run()
    print(f"Algorithm: ${algorithm.get_name()}")
    print(f"Problem: ${problem.get_name()}")
    print(f"Final non-dominted solution set size: ${len(result)}")

if __name__ == "__main__":
    main()
```
