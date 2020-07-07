from typing import List

class DistribGen():
    def create_distribution(self, dimensions: int, num_samples: int, steps: int, mock_run: bool, filename: str):
        distrib : List[int] = []
        final_distrib : List[float] = []

        for i in range(steps):
            sample : List[int] = []
            sample.append(i)
            distrib.append(sample)

        for sample in distrib:
            for i in range(dimensions):
                sample.append(0)
        
        addedNewSamples = True
        modifColumn = 1

        while addedNewSamples == True:
            addedNewSamples = False
            newDistrib : List[List[int]] = []
            for sample in distrib:
                sum = 0
                for i in range(len(sample)):
                    sum += sample[i]
            for i in range(steps):
                if (modifColumn + 1) < dimensions:
                    if (sum + i) <= steps:
                        newSample : List[int] = []
                        







    