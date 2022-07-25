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
v = 0.25  # The speed at which signal can be considered propagating  (m/s)
length_coef = int(1 / (v / 100))  # Coefficient to calculate length constant lambda (l)
margin_of_error = 0.15  # Hard to explain but unit is radians
num_points = 200  # Number of points that are evaluated in the squid sensors area
z = 2  # Height at which the sensor is located
r = 5  # Radius of the SQUID-sensor (cm)

################### SIGNALS ############################
signal_t_start = 0  # Starting time point of the signal (milliseconds)
signal_t_end = 1000  # Ending time point of the signal (milliseconds)

x = np.linspace(signal_t_start, signal_t_end, int(accuracy * (signal_t_end - signal_t_start) / 1000))
clean_signal = (1 / 1000000) * np.sin(3 * x)

noise = np.random.normal(0, 0.0000001, 500)

real = clean_signal + noise

################### NEURONS ############################
network_struct = [3, 3, 3]  # Specify network shape
locations = [[(-2, 2, surface_func(-2, 2)), (-2, 0, surface_func(-2, 0)), (-2, -2, surface_func(-2, -2))],
             [(0, 2, surface_func(0, 2)), (0, 0, surface_func(0, 0)), (0, -2, surface_func(0, -2))],
             [(2, 2, surface_func(2, 2)), (2, 0, surface_func(2, 0)), (2, -2, surface_func(2, -2))]]








