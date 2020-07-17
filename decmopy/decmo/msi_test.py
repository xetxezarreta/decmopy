import datetime, time, random
from typing import List

from jmetal.core.problem import FloatProblem
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
      self.speed = speed
      self.h_func = h_func
      self.h_func_obj = h_func_obj
      self.f_mtmto = 0
      self.caudal = caudal
      self.consumption = consumption
      self.avg_useful_life = (h_func_obj - h_func) / (time.time() - f_mtmto)


class MSI(FloatProblem):
   def __init__(self, compressors: List[Compressor], number_of_variables: int):
      super(MSI, self).__init__()
      self.compressors = compressors
      self.number_of_variables = number_of_variables
      self.number_of_objectives = 4
      self.number_of_constraints = 0

      self.lower_bound = self.number_of_variables * [0.0]
      self.upper_bound = self.number_of_variables * [1.0]

   def evaluate(self, solution: FloatSolution) -> FloatSolution:
      # obj1: Minimizar la suma de los consumos de todos los compresores
      total_consumption = 0.0
      for comp in solution.variables:
         total_consumption += comp[3]
      solution.objectives[0] = total_consumption
      return solution

   def create_solution(self) -> FloatSolution:
      new_solution = FloatSolution(
         self.lower_bound,
         self.upper_bound,
         self.number_of_objectives,
         self.number_of_constraints,
      )
      new_solution.variables = [[] for _ in range(len(self.compressors))]

      for i in range(len(self.compressors)):
         vars = [
            self.compressors[i].speed,
            self.compressors[i].h_func,
            self.compressors[i].caudal,
            self.compressors[i].consumption,
            self.compressors[i].avg_useful_life
         ]
         new_solution.variables[i] = vars
      
      return new_solution

   def get_name(self) -> str:
      return "MSI"

def main():
   variable_speed = [0, 1]   
   speed = [0, 1, 2, 3]
   h_func = [50, 100, 150, 200]
   h_func_obj = 250
   f_mtmto = time.mktime(datetime.datetime.strptime("01/07/2020", "%d/%m/%Y").timetuple())
   caudal = [25, 50, 75, 100]
   consumption = [50, 75, 100, 125]

   compressors = []

   for i in range(4):
      comp = Compressor(i+1, 1, speed[i], h_func[i], h_func_obj, f_mtmto, caudal[i], consumption[i])
      compressors.append(comp)
   
   problem = MSI(compressors, number_of_variables=4)
   algorithm = DECMO(problem, max_iterations=250)
   result = algorithm.run()
   print(f"Algorithm: ${algorithm.get_name()}")
   print(f"Problem: ${problem.get_name()}")
   print(f"Final non-dominted solution set size: ${len(result)}")


if __name__ == "__main__":
    main()
