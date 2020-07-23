import sys, pandas

from msi_compressor import (
   Compressor, 
   speeds_to_flow, 
   speeds_to_consumption, 
   solution_changes
)
from msi_problem import MSI
from decmo_integer import DECMO

# python msi_integer.py <caudal_consigna> <data_path>
# python msi_integer.py 100 ./data.csv
def main(argv):
   try:
      flow_obj = int(argv[0])
      data = pandas.read_csv(argv[1])

      problem = MSI(flow_obj)
      for _, row in data.iterrows():
         comp = Compressor(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
         problem.add_compressor(comp)
      
      algorithm = DECMO(problem, individual_population_size=50, max_iterations=250)
      results = algorithm.run()

      print(f"Algorithm: ${algorithm.get_name()}")
      print(f"Problem: ${problem.get_name()}")
      print(f"Final non-dominted solution set size: ${len(results)}")    

      print("-----------------------------------------------")
      print("SOLUCIONES")
      print("-----------------------------------------------")

      final_solutions = []

      for r in results:
         vars = [int(round(i)) for i in r.variables]
         flow = speeds_to_flow(vars)
         consumption = speeds_to_consumption(vars)      
         changes = solution_changes(problem.compressors, r)
         if (flow_obj <= flow) and (vars not in final_solutions):
            final_solutions.append(vars)
            print(vars, "Caudal:", str(flow), "Consumo:", str(consumption), "Cambios:", str(changes))  
      
      if len(final_solutions) == 0:
         print("No hay soluciones para el caudal objetivo " + str(flow_obj))      
      print("-----------------------------------------------")
   except Exception as e:
      print(e)

if __name__ == "__main__":
    main(sys.argv[1:])
