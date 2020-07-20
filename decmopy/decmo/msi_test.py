import datetime, time, random
from typing import List

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from decmo import DECMO

def speed_to_caudal(speed):
   return speed * 20

def speeds_to_caudal(speeds: List[int]):
   caudal = 0
   for s in speeds:
      caudal += speed_to_caudal(s)
   return caudal

def speed_to_consumption(speed):
   return speed * 10

def speeds_to_consumption(speeds: List[int]):
   consumption = 0
   for s in speeds:
      consumption += speed_to_consumption(s)
   return consumption

class Compressor(object):
   """Compresor.

   Attributes:
      id                Número del Compresor (identificador).
      variable_speed    Variable binaria que dice si el compresor permite velocidad variable o no.
      speed             Velocidad a la que funciona el compresor (regulable, discretizado).
      max_speed         Velocidad máxima a la que funciona el compresor (regulable, discretizado).
      min_speed         Velocidad máxima a la que funciona el compresor (regulable, discretizado).
      h_func            Horas de Funcionamiento desde último mantenimiento.
      h_func_obj        Horas de Funcionamiento Objetivo antes del Mantenimiento.
      f_mtmto           Fecha de Mantenimiento Programado (timestamp).
      caudal            Caudal en m3 de aire que da el Compresor a una velocidad determinada.
      consumption       Consumo del compresor.
   """

   def __init__(
      self,
      id: int,
      variable_speed: bool,
      speed: int,
      max_speed: int,
      min_speed: int,
      h_func: float,
      h_func_obj: float,
      f_mtmto: float,
   ):
      self.id = id
      self.variable_speed = variable_speed
      self.speed = speed
      self.max_speed = max_speed
      self.min_speed = min_speed
      self.h_func = h_func
      self.h_func_obj = h_func_obj
      self.f_mtmto = f_mtmto
      self.caudal = speed_to_caudal(speed)
      self.consumption = speed_to_consumption(speed)
      self.avg_useful_life = (h_func_obj - h_func) / (time.time() - f_mtmto)


class MSI(FloatProblem):
   def __init__(self):
      super(MSI, self).__init__()
      self.compressors = []
      self.number_of_variables = 0
      self.number_of_objectives = 4
      self.number_of_constraints = 0
      self.lower_bound = []
      self.upper_bound = []

   def evaluate(self, solution: FloatSolution) -> FloatSolution:      
      variables = [int(round(i)) for i in solution.variables]

      # obj1: Minimizar la suma de los consumos de todos los compresores
      consumption = 0
      for speed in variables:
         if speed != 0:
            consumption += speed_to_consumption(speed)
      solution.objectives[0] = consumption

      # obj2: Maximizar la distribución de horas de funcionamiento 
      avg_useful_life = 0
      j = 0
      for i, speed in enumerate(variables):
         if speed != 0:
            avg_useful_life += self.compressors[i].avg_useful_life
            j += 1
      if j is 0:
         solution.objectives[1] = -1.0 * (avg_useful_life)
      else:
         solution.objectives[1] = -1.0 * (avg_useful_life / j)

      # obj3: Minimizar Cambios desde la solucion anterior
      changes = 0
      for i, speed in enumerate(variables):
         if speed != self.compressors[i].speed:
            changes += 1
      solution.objectives[2] = changes  

      # obj4: Maximizar la diferencia de caudal respecto a la solución anterior
      caudal_diff = sum(i.caudal for i in self.compressors)
      for i, speed in enumerate(variables):
         caudal_diff -= speed_to_caudal(speed)
      solution.objectives[3] = caudal_diff     

      return solution 
   
   def add_compressor(self, compressor: Compressor):
      self.compressors.append(compressor)
      self.number_of_variables += 1
      self.upper_bound.append(compressor.max_speed)
      self.lower_bound.append(compressor.min_speed)   

   def get_name(self) -> str:
      return "MSI"

def main():
   h_func_obj = 250
   f_mtmto = time.mktime(datetime.datetime.strptime("01/07/2020", "%d/%m/%Y").timetuple())
   
   problem = MSI()
   problem.add_compressor(Compressor(1, False, 1, 1, 0, 20, h_func_obj, f_mtmto))
   problem.add_compressor(Compressor(2, False, 1, 1, 0, 250, h_func_obj, f_mtmto))
   problem.add_compressor(Compressor(3, False, 1, 1, 0, 250, h_func_obj, f_mtmto))
   problem.add_compressor(Compressor(4, True, 0, 5, 0, 20, h_func_obj, f_mtmto))

   algorithm = DECMO(problem, individual_population_size=50, max_iterations=100)
   results = algorithm.run()
   print(f"Algorithm: ${algorithm.get_name()}")
   print(f"Problem: ${problem.get_name()}")
   print(f"Final non-dominted solution set size: ${len(results)}")

   global_caudal = 0
   global_consumption = 0
   for c in problem.compressors:
      if c.speed != 0:
         global_caudal += speed_to_caudal(c.speed)
         global_consumption += speed_to_consumption(c.speed)

   final_solutions = []

   for result in results:
      s = [int(round(i)) for i in result.variables]
      caudal = speeds_to_caudal(s)
      consumption = speeds_to_consumption(s)

      if global_caudal <= caudal and s not in final_solutions:
         final_solutions.append(s)
         print(s, "Caudal Instalación:", str(global_caudal), "Caudal Solución:", str(caudal), "Consumo Instalación:", str(global_consumption), "Consumo Solución:", str(consumption))  

if __name__ == "__main__":
    main()
