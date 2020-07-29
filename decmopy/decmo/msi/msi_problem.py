import random

from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution

from msi_compressor import Compressor
from msi_utils import speeds_to_flow, speeds_to_consumption, solution_changes

class MSI(IntegerProblem):
   def __init__(self, flow_obj: float):
      super(MSI, self).__init__()

      self.flow_obj = flow_obj

      self.compressors = []
      self.number_of_variables = 0
      self.number_of_objectives = 3
      self.number_of_constraints = 3
      self.lower_bound = []
      self.upper_bound = []

      self.obj_directions = [self.MINIMIZE, self.MAXIMIZE, self.MINIMIZE]
      self.obj_labels = ['Consumo', 'Distribución Horas', 'Cambios']
      

   def evaluate(self, solution: IntegerSolution) -> IntegerSolution:      
      variables = [int(round(i)) for i in solution.variables]

      # obj1: Minimizar la suma de los consumos de todos los compresores
      consumption = speeds_to_consumption(variables, self.compressors)
      solution.objectives[0] = consumption

      # obj2: Maximizar la distribución de horas de funcionamiento 
      avg_useful_life = []
      for i, speed in enumerate(variables):
         if speed != 0:
            avg_useful_life.append(self.compressors[i].avg_useful_life)     
      if len(avg_useful_life) == 0:
         solution.objectives[1] = 0
      else:
         solution.objectives[1] = -1 * sum(avg_useful_life) / len(avg_useful_life)

      # obj3: Minimizar Cambios desde la solucion anterior (encendido/apagado)
      changes = solution_changes(variables, self.compressors)
      solution.objectives[2] = changes  

      self.__evaluate_constrains(solution)

      return solution 

   def __evaluate_constrains(self, solution: IntegerSolution) -> None:
      '''
      Every constraint must be expressed as an unequality of type expression >=0.0. 
      When expression < 0.0 then it is considered as a constraint violation.
      '''
      variables = [int(round(i)) for i in solution.variables]
      # constrain 1: caudal igual o superior al indicado
      solution_flow = speeds_to_flow(variables, self.compressors)
      flow_diff = solution_flow - self.flow_obj
      solution.constraints[0] = flow_diff

      # constrain 2: al menos un compresor variable en marcha
      var_running = -1
      for i, speed in enumerate(variables):
         if speed != 0 and self.compressors[i].variable_speed:
            var_running = 1
            break     
      solution.constraints[1] = var_running

      # constrain 3: no puede utilizarse un compresor con f_mtmto-hoy=<0
      mantainance = 1
      for i, speed in enumerate(variables):
         if speed != 0 and self.compressors[i].avg_useful_life <= 0:
            mantainance = -1
            break
      solution.constraints[2] = mantainance

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
