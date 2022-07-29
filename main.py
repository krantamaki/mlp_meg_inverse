import numpy as np
import scipy.optimize as opt
from scipy.constants import mu_0
import itertools
import random
import timeit

from config import surface_func, margin_of_error, num_points, reals, z, r, locations, network_struct, center_points, layers
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


def B_t(coordinates):
    """
    Calculate the magnetic field at given coordinates over the time interval
    Implementation is naive one using Biot-Savart law and assuming that the current
    is found at the point of the neuron
    :param coordinates: Coordinates of wanted point
    :return: ndarray corresponding the magnetic field strength at given point over time
    """
    magnetic_fields = []
    for neuron in neurons:
        if neuron.prev_layer is not None:
            point = neuron.coordinates
            r_0 = np.sqrt(sum([(point[j] - coordinates[j]) ** 2 for j in range(3)]))
            normal_vect = np.array(neuron.orientation())
            # Calculate the angle theta between the neuron orientation and vector from neuron point to the given point
            # and if |pi/2 - |theta|| < margin of error magnetic field is included
            direct_vect = np.array([coordinates[i] - point[i] for i in range(3)])
            theta = np.abs(np.arccos(np.dot(normal_vect, direct_vect) / r_0))
            if np.abs(np.pi / 2 - theta) < margin_of_error or np.abs(3 * np.pi / 2 - theta) < margin_of_error:
                magnetic_fields.append(mu_0 * neuron.I_tot() / (2 * np.pi * r))

    return np.sum(magnetic_fields, axis=0)


def squid_measurement(i):
    """
    Simulate what the SQUID-sensor would measure. In this naive implementation it's just a sum of the magnetic fields
    at generated points. The plane on which the sensor is located is parallel to xy-plane and located at height z.
    :param i: The index of the center point
    :return: Measured magnetic field strength as a function of time
    """
    def circle_points(R, N):
        # From https://stackoverflow.com/questions/33510979/generator-of-evenly-spaced-points-in-a-circle-in-python
        points = []
        for r_i, n_i in zip(R, N):
            theta = np.linspace(0, 2 * np.pi, n_i, endpoint=False)
            x = r_i * np.cos(theta) + center_points[i][0]
            y = r_i * np.sin(theta) + center_points[i][1]
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
        B_i.append(B_t(point))

    return sum(B_i)


def corr(x0):
    """
    Calculate the correlation between "real" signals and the one caused by neurons in neural net
    Coordinates of neuron i are found at x0[3 * i:3 * (i + 1)]
    :param x0: ndarray of shape (n,) of coordinates
    :return: Correlation value
    """
    n = len(reals[0])
    for i, neuron in enumerate(neurons):
        # Update neuron coordinates
        neuron.coordinates = x0[i * 3:(i + 1) * 3]
    rhos = []
    for i in range(len(center_points)):
        simulated = squid_measurement(i)
        numerator = n * np.dot(reals[i], simulated) - np.sum(reals[i]) * np.sum(simulated)
        denominator = np.sqrt(n * np.sum(np.square(reals[i])) - np.sum(reals[i]) ** 2) * np.sqrt(n * np.sum(np.square(simulated)) - np.sum(simulated) ** 2)
        rho_i = numerator / denominator
        if isinstance(rho_i, np.ndarray):
            rho_i = 0
        rhos.append(rho_i)

    rho = sum(rhos) / len(rhos)

    print("Average correlation ", rho)
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

    print("Total distance from surface ", total_dist)

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
    sol = opt.minimize(corr, x0, method='SLSQP', constraints=cons, options={'ftol': 0.00001, 'maxiter': 10})

    return sol


def main():
    """
    for i in range(len(center_points)):
        simulated = squid_measurement(i)
        plot_signal(simulated, reals[i], title=f"{i}_squashed_{layers}x{network_struct[0]}",
                    plot_real=False, plot_clean=False)
    """
    start_time = timeit.default_timer()
    print("Started: ", start_time)
    x0 = list(itertools.chain(*[neuron.coordinates for neuron in neurons]))
    rho = -corr(np.array(x0).reshape(len(x0),))
    visualize(neurons, title=f"Initial_connections_{layers}x{network_struct[0]} corr {rho}", interval=(min(x0), max(x0)))
    print("Initial correlation ", rho)
    for i in range(len(center_points)):
        simulated = squid_measurement(i)
        plot_signal(simulated, reals[i], title=f"{i}_Initial_{layers}x{network_struct[0]} corr {rho}",
                    plot_real=True, plot_clean=True)
    sol = optimize()
    print(sol)
    for i in range(len(center_points)):
        simulated = squid_measurement(i)
        plot_signal(simulated, reals[i], title=f"{i}_Optimized_{layers}x{network_struct[0]} corr {-sol.fun}",
                    plot_real=True, plot_clean=True)
    visualize(neurons, title=f"Optimized_connections_{layers}x{network_struct[0]} corr {-sol.fun}", interval=(min(sol.x), max(sol.x)))
    print("Total runtime: ", timeit.default_timer() - start_time)


main()
