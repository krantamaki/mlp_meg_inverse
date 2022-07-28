import numpy as np
import random

random.seed(42)


def surface_func(x, y):
    """
    Function to calculate the z value for given x and y
    :param x: Either float, int or numpy array
    :param y: Either float, int or numpy array
    :return: The z coordinates of given x and y
    """
    return -(3 / 2) * (np.cos(x / 2) ** 2 + np.cos(y / 2) ** 2) ** 2


def gradient_func(x, y):
    """
    Function that calculates the gradient of the surface_func at any given point
    :param x: x-coordinate
    :param y: y-coordinate
    :return: List representing the gradient at given point(s)
    """
    return [3 * (np.cos(x / 2) ** 2 + np.cos(y / 2) ** 2) * np.cos(x / 2) * np.sin(x / 2),
            3 * (np.cos(x / 2) ** 2 + np.cos(y / 2) ** 2) * np.cos(y / 2) * np.sin(y / 2)]


#################### PARAMETERS #######################
I_max = 1  # The maximum current possible in neurons (A)
accuracy = 500  # Number of points calculate to a time period of one second
v = 0.38  # The speed at which signal can be considered propagating  (m/s)
length_coef = int(1 / (v / 500))  # Coefficient to calculate length constant lambda (l)
margin_of_error = 0.25  # Hard to explain but unit is radians
num_points = 100  # Number of points that are evaluated in the squid sensors area
z = 2  # Height at which the sensors are located
r = 3  # Radius of the SQUID-sensors (cm)
center_points = [(0, 0), (0, 6), (6, 0), (0, -6), (-6, 0)]  # Center points of the sensors


################### SIGNALS ############################
signal_t_start = 0  # Starting time point of the signal (milliseconds)
signal_t_end = 1000  # Ending time point of the signal (milliseconds)

x = np.linspace(signal_t_start, signal_t_end, int(accuracy * (signal_t_end - signal_t_start) / 1000))

amplitude = 0.00002
period = 1 / 100
phase_shift = np.pi / (period * 2)
clean_signal = amplitude * np.sin(period * (x + phase_shift)) + amplitude

reals = [clean_signal + np.random.normal(0, 0.000005, int(accuracy * (signal_t_end - signal_t_start) / 1000))
         for i in range(len(center_points))]

################### NEURONS ############################
network_struct = [6, 6, 6, 6, 6, 6, 6]  # Specify network shape
layers = len(network_struct)
locations = []
for _, layer_size in enumerate(network_struct):
    locations.append([])
    i = -layers + 2 * _
    for j in range(-layer_size, layer_size + (layers % 2), 2):
        locations[_].append((i, j, surface_func(i, j)))

