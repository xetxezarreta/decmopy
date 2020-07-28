from typing import List

def speed_to_flow(speed: int, rels: List[List[float]]):
    flow = 0
    for r in rels:
        if r[0] == speed:
            flow += r[1]
            break
    return flow 

def speeds_to_flow(speeds: List[int], compressors):
    flow = 0
    for i, s in enumerate(speeds):
        flow += speed_to_flow(s, compressors[i].rel_velocidad_caudal)
    return flow

def speed_to_consumption(speed: int, rels: List[List[float]]):
    consumption = 0
    for r in rels:
        if r[0] == speed:
            consumption += r[1]
            break
    return consumption 

def speeds_to_consumption(speeds: List[int], compressors):
    consumption = 0
    for i, s in enumerate(speeds):
        consumption += speed_to_consumption(s, compressors[i].rel_velocidad_consumo)
    return consumption

def solution_changes(speeds: List[int], compressors):
    changes = 0
    for i, speed in enumerate(speeds):
        if (speed != compressors[i].speed) and (speed == 0 or compressors[i].speed == 0):
            changes += 1
    return changes