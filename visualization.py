import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import surface_func, real, clean_signal, x


def plot_current(neuron):
    fig = plt.figure()
    current = neuron.I_tot()
    plt.plot(x, current)

    plt.show()


def plot_signal(simulated, plot_real=True, plot_clean=True, title=None):
    """
    TODO: Description
    :param simulated:
    :param plot_real:
    :param plot_clean:
    :param title:
    :return:
    """
    fig = plt.figure()
    plt.plot(x, simulated)
    if plot_real:
        plt.plot(x, real)
    if plot_clean:
        plt.plot(x, clean_signal)
    if title is not None:
        plt.title(title)

    plt.savefig(f"{title.split(' ')[0]}.png")


def visualize(neurons, interval=(-5, 5), points=100,
              visualize_orientation=True,
              visualize_neurons=True,
              visualize_connections=True,
              title=None):
    """
    Visualize the neurons and the surface
    :param neurons: List of Neuron objects. If empty only the surface is plotted
    :param interval: Tuple that gives the start and end points for x and y (default: (-5, 5))
    :param points: Number of points on the interval (default: 100)
    :param visualize_orientation: Boolean to decide whether unit normal vectors depicting neuron orientation are plotted
                                  (default: True)
    :param visualize_neurons: Boolean to decide whether neuron positions are plotted
                              (default: True)
    :param visualize_connections: Boolean to decide whether connections between neurons is plotted
                                  (default: True)
    :param title: String to be passed as the title argument to plot (Default: None)
    :return: None
    """
    x = np.linspace(interval[0], interval[1], points)
    y = np.linspace(interval[0], interval[1], points)

    X, Y = np.meshgrid(x, y)
    Z = surface_func(X, Y)

    vectors = np.array([[*neuron.coordinates, *neuron.orientation()] for neuron in neurons])
    coords = np.array([[*neuron.coordinates] for neuron in neurons])
    coords = coords.transpose()
    X_2, Y_2, Z_2, U, V, W = zip(*vectors)

    fig = plt.figure()
    if title is not None:
        ax = plt.axes(projection='3d', title=title)
    else:
        ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    if visualize_orientation and len(neurons) != 0:
        ax.quiver(X_2, Y_2, Z_2, U, V, W)
    if visualize_neurons and len(neurons) != 0:
        ax.scatter(*coords)
    if visualize_connections and len(neurons) > 1:
        for neuron in neurons:
            if neuron.prev_layer is not None:
                for prev_neuron in neuron.prev_layer:
                    ax.plot(*[[neuron.coordinates[i], prev_neuron.coordinates[i]] for i in range(3)])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
