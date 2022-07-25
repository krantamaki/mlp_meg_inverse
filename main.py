import numpy as np
import scipy.optimize as opt
from scipy.constants import mu_0
import itertools
import random

from config import surface_func, margin_of_error, accuracy, num_points, real, z, r, locations, network_struct, x
from visualization import visualize, plot_signal, plot_current
from neuron import Neuron

random.seed(42)


def initialize_network():
    """
    Create a list of Neuron object
    :return: List of neurons
    """
    neurons = []
    for i, layer in enumerate(network_struct):
        for j in range(layer):
            if i == 0:
                if j == 0:
                    neurons.append([Neuron(None, locations[i][j], random.random(), random.random(), random.random())])
                else:
                    neurons[i].append(Neuron(None, locations[i][j], random.random(), random.random(), random.random()))
            else:
                if j == 0:
                    neurons.append([Neuron(neurons[i - 1], locations[i][j], random.random(), random.random(), None)])
                else:
                    neurons[i].append(Neuron(neurons[i - 1], locations[i][j], random.random(), random.random(), None))

    return list(itertools.chain(*neurons))


# Add list of Neurons into the global scope
neurons = initialize_network()


def B_t(coordinates, t):
    """
    Calculate the magnetic field at given coordinates at time point t
    Implementation is naive one using Biot-Savart law and assuming that the current
    is found at the point of the neuron
    :param coordinates: Coordinates of wanted point
    :param t: Time point (milliseconds)
    :return: Value corresponding to the magnetic field strength at given point (and time)
    """
    # Find all the neurons that are active at given time
    active_neurons = [neuron for neuron in neurons if neuron.t_start <= t <= neuron.calc_t_end()]
    # Find the neurons which magnetic fields correspond with the given point (with some margin of error)
    B_i = [0]
    for neuron in active_neurons:
        if neuron.prev_layer is not None:
            point = neuron.coordinates
            r = np.sqrt(sum([(point[j] - coordinates[j]) ** 2 for j in range(3)]))
            normal_vect = np.array(neuron.orientation())
            # Calculate the angle theta between the neuron orientation and vector from neuron point to the given point
            # and if |pi/2 - |theta|| < margin of error magnetic field is included
            direct_vect = np.array([coordinates[i] - point[i] for i in range(3)])
            theta = np.abs(np.arccos(np.dot(normal_vect, direct_vect) / r))
            if np.abs(np.pi / 2 - theta) < margin_of_error or np.abs(3 * np.pi / 2 - theta) < margin_of_error:
                current = neuron.I_tot()
                B_i.append((mu_0 * current[int(accuracy * ((t - neuron.t_start) / 1000))]) / (2 * np.pi * r))

    return sum(B_i)


def squid_measurement(t):
    """
    Simulate what the SQUID-sensor would measure. In this naive implementation it's just a sum of the magnetic fields
    at generated points. The plane on which the sensor is located is parallel to xy-plane and located at height z.
    Thus the center point of the sensor is (0, 0, z).
    :param t: Wanted time point
    :return: Measured magnetic field strength
    """
    def circle_points(R, N):
        # From https://stackoverflow.com/questions/33510979/generator-of-evenly-spaced-points-in-a-circle-in-python
        points = []
        for r_i, n_i in zip(R, N):
            theta = np.linspace(0, 2 * np.pi, n_i, endpoint=False)
            x = r_i * np.cos(theta)
            y = r_i * np.sin(theta)
            for x_i, y_i in zip(x, y):
                points.append((x_i, y_i, z))
        return points

    point_nums = [1, 10]
    while sum(point_nums) < num_points:
        point_nums.append(point_nums[-1] + 10)

    radiuses = np.linspace(0, r, len(point_nums))

    if len(radiuses) < len(point_nums):
        radiuses.append(radiuses[-1] + r / len(point_nums))

    all_points = circle_points(radiuses, point_nums)

    B_i = []
    for point in all_points:
        B_i.append(B_t(point, t))

    print("Time point ", t)

    return sum(B_i)


def corr(x0):
    """
    Calculate the correlation between "real" signal and the one caused by neurons in neural net
    "Real" signal and neurons can be accessed from config file. Coordinates of neuron i are found at x0[3 * i:3 * (i + 1)]
    :param x0: ndarray of shape (n,) of coordinates
    :return: Correlation value
    """
    n = len(real)
    for i, neuron in enumerate(neurons):
        # Update neuron coordinates
        neuron.coordinates = x0[i * 3:(i + 1) * 3]
    simulated = np.array([squid_measurement(t) for t in range(int((n / accuracy) * 1000))])
    numerator = n * np.dot(real, simulated) - np.sum(real) * np.sum(simulated)
    denominator = np.sqrt(n * np.sum(np.square(real)) - np.sum(real) ** 2) * np.sqrt(n * np.sum(np.square(simulated)) - np.sum(simulated) ** 2)

    rho = numerator / denominator

    # print("Current correlation is ", rho)

    # Return the negation of the correlation so that minimizing it becomes maximizing the non-negated correlation
    return -rho


def total_distance_to_surf(x0):
    """
    TODO: DESCRIPTION
    :param x0:
    :return:
    """
    points = [x0[i * 3:(i + 1) * 3] for i in range(len(x0) // 3)]
    total_dist = 0
    for point in points:
        surface_point = [point[0], point[1], surface_func(point[0], point[1])]
        total_dist += np.sqrt(sum([(point[i] - surface_point[i]) ** 2 for i in range(3)]))

    return total_dist


def optimize():
    """
    Optimize the neuron positions to maximize the average correlation between real signal and one simulated from given
    input
    :return: Whatever scipy minimize returns
    """
    # Create constraints in the form scipy wants them
    cons = {'type': 'eq', 'fun': total_distance_to_surf}
    x0 = np.array(list(itertools.chain(*[neuron.coordinates for neuron in neurons])))
    x0 = x0.reshape(len(x0),)
    sol = opt.minimize(corr, x0, method='SLSQP', constraints=cons)

    return sol


def main():
    plot_current(neurons[7])
    """
    visualize(neurons, title="Initial")
    # x0 = list(itertools.chain(*[neuron.coordinates for neuron in neurons]))
    # rho = -corr(np.array(x0).reshape(len(x0),))
    simulated = np.array([squid_measurement(t) for t in range(int((len(x) / accuracy) * 1000))])
    plot_signal(simulated, title="Test", plot_real=True, plot_clean=True)
    """
    """
    print("Initial correlation ", rho)
    sol = optimize()
    print(sol)
    simulated = np.array([squid_measurement(t) for t in range(int((len(x) / accuracy) * 1000))])
    plot_signal(simulated, title=f"Optimized corr {-sol.fun}", plot_real=False, plot_clean=False)
    visualize(neurons, title="Optimized")
    """
    # plot_current(neurons[4])


main()
