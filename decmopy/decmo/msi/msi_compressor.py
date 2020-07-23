import time, datetime
from typing import List

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

def solution_changes(compressors, solution):
    changes = 0
    for i, speed in enumerate(solution.variables):
        if (speed != compressors[i].speed) and (speed == 0 or compressors[i].speed == 0):
            changes += 1
    return changes


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
        self.f_mtmto = f_mtmto = time.mktime(datetime.datetime.strptime(f_mtmto, '%d/%m/%Y').timetuple())
        self.caudal = speed_to_caudal(speed) # Caudal en m3 de aire que da el Compresor a una velocidad determinada.
        self.consumption = speed_to_consumption(speed) # Consumo del compresor (Kw/m3 aire).
        self.avg_useful_life = (h_func_obj - h_func) / (self.f_mtmto - time.time())
        