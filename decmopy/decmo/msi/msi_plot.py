from jmetal.lab.visualization import Plot

def plot_front(results):
    plot = Plot( \
        title='Pareto front approximation', \
        axis_labels=['obj1: consumo', 'obj2: distribución', 'obj3: cambios'] \
    )
    plot.plot(results, label='MSI PnP', filename='./img/MSI-PnP', format='png')
