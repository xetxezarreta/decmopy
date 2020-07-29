from jmetal.lab.visualization import Plot
import numpy as np
import matplotlib.pyplot as plt

def plot_front(fronts):
    title = "Pareto front approximation"
    axis_labels = ["obj1: consumo", "obj2: distribución", "obj3: cambios"]
    filename = "./img/MSI-PnP"
    format = "png"

    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    for sol in fronts:                
        ax.scatter(sol.objectives[0], sol.objectives[1], sol.objectives[2])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])    

    plt.savefig(filename + '.' + format, format=format, dpi=1000)

def plot_front1(results):
    plot = Plot( \
        title="Pareto front approximation", \
        axis_labels=["obj1: consumo", "obj2: distribución", "obj3: cambios"] \
    )
    plot.plot(results, label="MSI PnP", filename="./img/MSI-PnP", format="png")