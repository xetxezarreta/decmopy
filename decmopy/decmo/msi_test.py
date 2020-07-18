import datetime, time, random
from typing import List

from jmetal.core.problem import FloatProblem, IntegerProblem
from jmetal.core.solution import FloatSolution, BinarySolution
from decmo import DECMO


class Compressor(object):
   """Compresor.

   Attributes:
      id                Número del Compresor (identificador).
      variable_speed    Variable binaria que dice si el compresor permite velocidad variable o no.
      speed             Velocidad en Hz a la que funciona el compresor (regulable, discretizado) (4bit).
      h_func            Horas de Funcionamiento desde último mantenimiento.
      h_func_obj        Horas de Funcionamiento Objetivo antes del Mantenimiento.
      f_mtmto           Fecha de Mantenimiento Programado (timestamp).
      caudal            Caudal en m3 de aire que da el Compresor a una velocidad determinada.
      consumption       Consumo del compresor.
   """

   def __init__(
      self,
      id: int,
      variable_speed: int,
      speed: int,
      h_func: float,
      h_func_obj: float,
      f_mtmto: float,
      caudal: float,
      consumption: float,
   ):
      self.id = id
      self.variable_speed = variable_speed
      self.speed = speed #
      self.h_func = h_func
      self.h_func_obj = h_func_obj
      self.f_mtmto = 0
      self.caudal = caudal
      self.consumption = consumption
      self.avg_useful_life = (h_func_obj - h_func) / (time.time() - f_mtmto)


SPEED_CONSUMPTION = {
   0: 0,
   1: 25,
   2: 50,
   3: 75   
}  

SPEED_CAUDAL = {
   0: 0,
   1: 50,
   2: 75,
   3: 100,
}

class MSI(IntegerProblem):
   def __init__(self, compressors: List[Compressor]):
      super(MSI, self).__init__()
      self.compressors = compressors
      self.number_of_variables = len(compressors)
      self.number_of_objectives = 2
      self.number_of_constraints = 0

      self.lower_bound = self.number_of_variables * [0]
      self.upper_bound = self.number_of_variables * [3]

   def evaluate(self, solution: FloatSolution) -> FloatSolution:
      # obj1: Minimizar la suma de los consumos de todos los compresores
      total_consumption = 0
      for speed in solution.variables:
         total_consumption += SPEED_CONSUMPTION[int(round(speed))]
      solution.objectives[0] = total_consumption

      # obj2: Minimizar Cambios desde la solucion anterior
      changes = 0
      for i, speed in enumerate(solution.variables):
         if speed != self.compressors[i].speed:
            changes += 1
      solution.objectives[1] = changes

      return solution 

   def create_solution(self) -> FloatSolution:
      new_solution = FloatSolution(
         self.lower_bound,
         self.upper_bound,
         self.number_of_objectives,
         self.number_of_constraints,
      )      
      new_solution.number_of_variables = self.number_of_variables

      new_solution.variables = \
         [random.randint(0, 3) for _ in range(self.number_of_variables)]    

      return new_solution

   def get_name(self) -> str:
      return "MSI"

def main():
   variable_speed = [0, 1]   
   speed = [0, 1, 2, 3]
   h_func = [50, 100, 150, 200]
   h_func_obj = 250
   f_mtmto = time.mktime(datetime.datetime.strptime("01/07/2020", "%d/%m/%Y").timetuple())
   caudal = [0, 50, 75, 100]
   consumption = [0, 25, 50, 75]

   compressors = []

   for i in range(4):
      comp = Compressor(i+1, 1, speed[i], h_func[i], h_func_obj, f_mtmto, caudal[i], consumption[i])
      compressors.append(comp)
   
   problem = MSI(compressors)
   algorithm = DECMO(problem, max_iterations=250)
   result = algorithm.run()
   print(f"Algorithm: ${algorithm.get_name()}")
   print(f"Problem: ${problem.get_name()}")
   print(f"Final non-dominted solution set size: ${len(result)}")


if __name__ == "__main__":
    main()
