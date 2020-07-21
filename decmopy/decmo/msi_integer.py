import datetime, time, statistics, random
from typing import List

from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution

from decmo_integer import DECMO

def speed_to_caudal(speed: int):
   return speed * 20

def speeds_to_caudal(speeds: List[int]):
   caudal = 0
   for s in speeds:
      caudal += speed_to_caudal(s)
   return caudal

def speed_to_consumption(speed: int):
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
      f_mtmto           Fecha de Mantenimiento Programado (dd/mm/yyyy).
      caudal            Caudal en m3 de aire que da el Compresor a una velocidad determinada.
      consumption       Consumo del compresor (Kw/m3 aire).
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
      f_mtmto: str,
   ):
      self.id = id
      self.variable_speed = variable_speed
      self.speed = speed
      self.max_speed = max_speed
      self.min_speed = min_speed
      self.h_func = h_func
      self.h_func_obj = h_func_obj
      self.f_mtmto = f_mtmto = time.mktime(datetime.datetime.strptime(f_mtmto, "%d/%m/%Y").timetuple())
      self.caudal = speed_to_caudal(speed)
      self.consumption = speed_to_consumption(speed)
      self.avg_useful_life = (h_func_obj - h_func) / (self.f_mtmto - time.time())


class MSI(IntegerProblem):
   def __init__(self, caudal_obj: float):
      super(MSI, self).__init__()

      self.caudal_obj = caudal_obj

      self.compressors = []
      self.number_of_variables = 0
      self.number_of_objectives = 3
      self.number_of_constraints = 2
      self.lower_bound = []
      self.upper_bound = []

   def evaluate(self, solution: IntegerSolution) -> IntegerSolution:      
      variables = solution.variables

      # obj1: Minimizar la suma de los consumos de todos los compresores
      consumption = 0
      for speed in variables:
         if speed != 0:
            consumption += speed_to_consumption(speed)
      solution.objectives[0] = consumption

      # obj2: Maximizar la distribución de horas de funcionamiento 
      avg_useful_life = []
      for i, speed in enumerate(variables):
         if speed != 0:
            avg_useful_life.append(self.compressors[i].avg_useful_life) 
      if len(avg_useful_life) == 0:
         solution.objectives[1] = 0
      elif len(avg_useful_life) == 1:
         solution.objectives[1] = -1 * avg_useful_life[0]
      else:
         solution.objectives[1] = -1 * statistics.mean(avg_useful_life)

      # obj3: Minimizar Cambios desde la solucion anterior (encendido/apagado)
      changes = 0
      for i, speed in enumerate(variables):
         if (speed != self.compressors[i].speed) and (speed == 0 or self.compressors[i].speed == 0):
            changes += 1
      solution.objectives[2] = changes  

      self.__evaluate_constrains(solution)

      return solution 

   def __evaluate_constrains(self, solution: IntegerSolution) -> None:
      '''
      Every constraint must be expressed as an unequality of type expression >=0.0. 
      When expression < 0.0 then it is considered as a constraint violation
      '''
      variables = solution.variables
      # constrain 1: caudal igual o superior al indicado
      sol_caudal = sum([speed_to_caudal(i) for i in variables])
      diff_caudal = sol_caudal - self.caudal_obj
      solution.constraints[0] = diff_caudal

      # constrain 2: al menos un compresor variable en marcha
      var_running = -1
      for comp in self.compressors:
         if comp.variable_speed:
            var_running = 1
            break
      solution.constraints[1] = var_running
      
   def create_solution(self) -> IntegerSolution:
      new_solution = IntegerSolution(
         self.lower_bound,
         self.upper_bound,
         self.number_of_objectives,
         self.number_of_constraints)
      new_solution.variables = \
         [random.randint(self.lower_bound[i], self.upper_bound[i]) for i in range(self.number_of_variables)]
      return new_solution
   
   def add_compressor(self, compressor: Compressor):
      self.compressors.append(compressor)
      self.number_of_variables += 1
      self.upper_bound.append(compressor.max_speed)
      self.lower_bound.append(compressor.min_speed)   

   def get_name(self) -> str:
      return "MSI"

def main():
   h_func = [100, 100, 100, 100]
   h_func_obj = [250, 250, 250, 250]
   f_mtmto = ["31/08/2020", "31/08/2020", "31/08/2020", "31/08/2020"]  
   caudal_obj = 100

   problem = MSI(caudal_obj)
   problem.add_compressor(Compressor(1, False, 1, 1, 0, h_func[0], h_func_obj[0], f_mtmto[0]))
   problem.add_compressor(Compressor(2, False, 1, 1, 0, h_func[1], h_func_obj[1], f_mtmto[1]))
   problem.add_compressor(Compressor(3, True, 0, 5, 0, h_func[2], h_func_obj[2], f_mtmto[2]))
   problem.add_compressor(Compressor(4, True, 0, 5, 0, h_func[3], h_func_obj[3], f_mtmto[3]))

   algorithm = DECMO(problem, individual_population_size=50, max_iterations=250)
   results = algorithm.run()
   print(f"Algorithm: ${algorithm.get_name()}")
   print(f"Problem: ${problem.get_name()}")
   print(f"Final non-dominted solution set size: ${len(results)}")

   final_solutions = []

   for result in results:
      s = [int(round(i)) for i in result.variables]
      caudal = speeds_to_caudal(s)
      consumption = speeds_to_consumption(s)

      if s not in final_solutions:
         final_solutions.append(s)
         print(s, "Caudal Solución:", str(caudal), "Consumo Solución:", str(consumption))  

if __name__ == "__main__":
    main()
