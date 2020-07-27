import sys, json

from msi_compressor import (
   Compressor, 
   speeds_to_flow, 
   speeds_to_consumption, 
   solution_changes
)
from msi_problem import MSI
from decmo_integer import DECMO
from msi_plot import plot_front


# python msi_integer.py <caudal_consigna> <data_path>
# python msi_integer.py 100 ./data.csv
def main(argv):
   try:
      flow_obj = int(argv[0])
      with open(argv[1], 'r') as f:
         data = f.read()
      data = json.loads(data)

      problem = MSI(flow_obj)
      for i in data:
         comp = Compressor(
            i["id"], 
            i["variable_speed"], 
            i["speed"], 
            i["min_speed"], 
            i["max_speed"], 
            i["h_func"], 
            i["h_func_obj"], 
            i["f_mtmto"]
         )
         problem.add_compressor(comp)
      
      algorithm = DECMO(problem, individual_population_size=50, max_iterations=250)
      results = algorithm.run()
      plot_front(results)

      print(f"Algorithm: ${algorithm.get_name()}")
      print(f"Problem: ${problem.get_name()}")
      print(f"Final non-dominted solution set size: ${len(results)}")    

      print("-----------------------------------------------")
      print("SOLUCIONES")
      print("-----------------------------------------------")

      final_solutions = []

      for r in results:
         vars = [int(round(i)) for i in r.variables]
         sol_flow = speeds_to_flow(vars)
         if (flow_obj <= sol_flow) and (vars not in final_solutions):
            final_solutions.append(vars)
            sol_consumption = speeds_to_consumption(vars)      
            sol_changes = solution_changes(problem.compressors, vars)
            sol_distribution = ["%.2f" % i.avg_useful_life for i in problem.compressors]
            print(vars, \
               "| Caudal:", str(sol_flow), \
               "| Consumo:", str(sol_consumption), \
               "| Cambios:", str(sol_changes), \
               "| Vida Útil Promedio:", sol_distribution \
            )             
            
      if len(final_solutions) == 0:
         print("No hay soluciones para el caudal objetivo " + str(flow_obj))      
      print("-----------------------------------------------")
   except Exception as e:
      print(e)

if __name__ == "__main__":
   main(sys.argv[1:])
