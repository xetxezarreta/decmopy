import random
from pathlib import Path
from typing import List


class DistribGen:
    def create_distribution(self, dimensions: int, num_samples: int, filename: str):
        steps = 2
        remaining = 0.0
        stop = False

        while not stop:
            remaining = self.__create_distrib(
                dimensions, num_samples, steps, True, filename
            )
            if remaining > 0.0:
                stop = True
            else:
                steps += 1
        print(
            "Required steps: "
            + str(steps)
            + "\n"
            + "Rmoved samples: "
            + str("%.2f" % remaining)
            + "%"
        )
        self.__create_distrib(dimensions, num_samples, steps, False, filename)

    def __create_distrib(
        self,
        dimensions: int,
        num_samples: int,
        steps: int,
        mockRun: bool,
        filename: str,
    ):
        distrib: List[List[int]] = []
        final_distrib: List[List[float]] = []

        for i in range(steps + 1):
            sample: List[int] = []
            sample.append(i)
            distrib.append(sample)

        for sample in distrib:
            for i in range(1, dimensions):
                sample.append(0)

        addedNewSamples = True
        modifColumn = 1

        while addedNewSamples == True:
            addedNewSamples = False
            newDistrib: List[List[int]] = []
            for sample in distrib:
                sum = 0
                for i in range(len(sample)):
                    sum += sample[i]
                for i in range(steps + 1):
                    if (modifColumn + 1) < dimensions:
                        if (sum + i) <= steps:
                            newSample: List[int] = []
                            newSample.extend(sample)
                            newSample[modifColumn] = i
                            newDistrib.append(newSample)
                            addedNewSamples = True
                    else:
                        if (sum + i) == steps:
                            newSample: List[int] = []
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
            ind = random.randint(0, len(distrib) - 1)
            distrib.pop(ind)

        if not mockRun:
            try:
                path = Path(filename)
                if not path.parent.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.parent.exists():
                        print("Could not create directory path: ")
                if not path.exists():
                    path.touch()

                for sample in distrib:
                    s = ""
                    finalSample: List[int] = []
                    for i in range(len(sample)):
                        finalSample.append((sample[i] * 1.0) / steps)
                        s += str("%.6f" % finalSample[i])
                        s += " "
                    final_distrib.append(finalSample)
                    with open(path, "a") as file:
                        file.write(s + "\n")
            except Exception as e:
                print(e)

        percent_rem = (1.0 - (len(distrib) * 1.0 / originalSetSize)) * 100.0
        return percent_rem
