from typing import List
from jmetal.core.solution import FloatSolution


class DirectionRec:
    def __init__(
        self,
        id: int,
        weigh_vector: List[float],
        curr_sol: FloatSolution,
        fitness_value: float,
        nfeSinceLastUpdate: int,
    ):
        super().__init__()
        self.id = id
        self.weigh_vector = weigh_vector
        self.curr_sol = curr_sol
        self.fitness_value = fitness_value
        self.nfeSinceLastUpdate = nfeSinceLastUpdate
