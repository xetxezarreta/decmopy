from datetime import datetime
from typing import List

from msi_utils import speed_to_flow, speed_to_consumption

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
        min_speed: int,
        max_speed: int,
        h_func: float,
        h_func_obj: float,
        f_mtmto: str,
        rel_velocidad_caudal: List[List[float]],
        rel_velocidad_consumo: List[List[float]],
    ):
        self.id = id
        self.variable_speed = variable_speed
        self.speed = speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.h_func = h_func
        self.h_func_obj = h_func_obj
        self.f_mtmto = datetime.strptime(f_mtmto, "%d/%m/%Y")
        self.rel_velocidad_caudal = rel_velocidad_caudal
        self.rel_velocidad_consumo = rel_velocidad_consumo

        self.flow = speed_to_flow(speed, rel_velocidad_caudal) # Caudal en m3 de aire que da el Compresor a una velocidad determinada.
        self.consumption = speed_to_consumption(speed, rel_velocidad_consumo) # Consumo del compresor (Kw/m3 aire).
        self.avg_useful_life = (h_func_obj - h_func) / ((self.f_mtmto - datetime.now()).total_seconds() / 3600)
