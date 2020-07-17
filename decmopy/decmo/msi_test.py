import time
from typing import List

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from decmo import DECMO

"""
OBJETIVOS:
- obj1: minimizar el consumo (var: consumo)
- obj2: 

- obj: maximizar el caudal (caudal_nuevo >= caudal_instalación) (var: caudal)
"""


class Compressor(object):
   """Compresor.

   Attributes:
      id                Número del Compresor (identificador).
      variable_speed    Variable binaria que dice si el compresor permite velocidad variable o no.
      h_func            Horas de Funcionamiento desde último mantenimiento.
      h_func_obj        Horas de Funcionamiento Objetivo antes del Mantenimiento.
      f_mtmto           Fecha de Mantenimiento Programado (timestamp)
      caudal            Caudal en m3 de aire que da el Compresor a una velocidad determinada.
      consumption       Consumo del compresor.
   """

   def __init__(
      self,
      id: int,
      variable_speed: int,
      h_func: float,
      h_func_obj: float,
      f_mtmto: float,
      caudal: float,
      consumption: float
   ):
      self.id = id
      self.variable_speed = variable_speed
      self.h_func = h_func
      self.h_func_obj = h_func_obj
      self.f_mtmto = 0
      self.caudal = caudal
      self.consumption = consumption
      self.avg_useful_life = (h_func_obj - h_func) / (time.time() - f_mtmto)


class MSI(FloatProblem):
   def __init__(self, compressors: List[Compressor], number_of_variables: int = 8):
      super(MSI, self).__init__()
      self.number_of_variables = number_of_variables
      self.number_of_objectives = 4
      self.number_of_constraints = 0

      self.lower_bound = self.number_of_variables * [0.0]
      self.upper_bound = self.number_of_variables * [1.0]

   def evaluate(self, solution: FloatSolution) -> FloatSolution:
      pass

   def create_solution(self) -> FloatSolution:
      pass

   def get_name(self) -> str:
      return "MSI"


def main():
   print("Hello, world!")

if __name__ == "__main__":
   main()
