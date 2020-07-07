import random
from typing import List

class DistribGen():
    def create_distribution(self, dimensions: int, num_samples: int, filename: str):
        steps = 2
        remaining = 0.0
        stop = False

        while not stop:
            remaining = self.__create_distrib(dimensions, num_samples, steps, filename)
            if remaining > 0.0:
                stop = True
            else:
                steps += 1


    def __create_distrib(self, dimensions: int, num_samples: int, steps: int, filename: str):
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
                            newSample.extend(sample)
                            newSample[modifColumn] = i
                            newDistrib.append(newSample)
                            addedNewSamples = True
                    else:
                        if (sum + i) == steps:
                            newSample : List[int] = []
                            newSample.extend(sample)
                            newSample[modifColumn] = i
                            newDistrib.append(newSample)
                            addedNewSamples = True
            distrib = newDistrib
            modifColumn += 1
            if (modifColumn + 1) > dimensions:
                addedNewSamples = False
        
        originalSetSize = len(distrib)
        while len(distrib) > num_samples:
            ind = random.randint(0, len(distrib))
            distrib.remove(ind)

        percent_rem = (1.0 - (len(distrib) * 1.0 / originalSetSize)) * 100.0
        return percent_rem
    