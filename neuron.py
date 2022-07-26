import numpy as np

from config import surface_func, gradient_func, length_coef, accuracy, I_max, v, signal_t_start, signal_t_end, x


class Neuron:

    def __init__(self, prev_layer, coordinates, weight, bias, activation):
        """
        TODO: Description
        :param prev_layer: Array of Neuron objects from preceding layer
        :param coordinates: Array of the neurons coordinates [x, y, z]
        :param weight: The trained weight of the neuron
        :param bias: Bias of the neuron
        :param activation: The activation of the neuron if the neuron is located on the initial layer
        """
        self.prev_layer = prev_layer
        self.coordinates = coordinates
        self.weight = weight
        self.bias = bias
        self.activation = activation
        if prev_layer is not None:
            self.t_start = min([neuron.calc_t_end() for neuron in prev_layer])
        else:
            self.t_start = 0

    def calc_activation(self):
        """
        Calculate the activation of the neuron. Scaled to range [0, 1] by the sigmoid function
        :return: Float describing the activation
        """
        if self.prev_layer is not None:
            activations = np.array([neuron.calc_activation() if neuron.prev_layer is not None else neuron.activation
                                for neuron in self.prev_layer])
            weights = np.array([neuron.weight for neuron in self.prev_layer])
            activation = 1 / (1 + np.exp(-(np.dot(weights, activations) + self.bias)))
        else:
            if self.activation is not None:
                activation = self.activation
            else:
                raise Exception("No valid activation can be calculated")

        return activation

    def distance_to_surf(self):
        """
        Calculate neurons distance from wanted surface. This should be zero.
        :return: Float describing the distance
        """
        surface_point = [self.coordinates[0], self.coordinates[1], surface_func(self.coordinates[0], self.coordinates[1])]

        return np.sqrt(sum([(self.coordinates[i] - surface_point[i]) ** 2 for i in range(3)]))

    def distance_to_i(self, i):
        """
        Distance on given surface between current neuron and neuron i of previous layer
        :param i: Index of the neuron on previous layer
        :return: Float describing the distance (m)
        """
        # TODO: Replace naive implementation with proper one
        # Possibly: https://stackoverflow.com/questions/42634704/how-can-i-get-a-fast-estimate-for-the-distance-between-a-point-and-a-bicubic-spl
        return np.sqrt(sum([(self.coordinates[j] - self.prev_layer[i].coordinates[j]) ** 2 for j in range(3)])) / 100

    def calc_t_end(self):
        """
        Calculate the end time point of signal
        :return: Float describing the end time (in milliseconds)
        """
        if self.prev_layer is not None:
            max_dist = max([self.distance_to_i(i) for i in range(len(self.prev_layer))])
        else:
            max_dist = 0
        t_end = int(self.t_start + (max_dist / v) * 1000)

        return t_end

    def orientation(self):
        """
        Calculate the orientation of the neuron ie. the normal vector at the neurons coordinates at the surface
        :return: ndarray describing the orientation
        """
        grads = gradient_func(*self.coordinates[0:2])
        grads.append(-1)
        normal_vect = np.array(grads)
        length = np.sqrt(sum([x_i ** 2 for x_i in normal_vect]))

        return normal_vect / length

    def I_i(self, i):
        """
        Calculate the current caused by neuron at index i on previous layer on given neuron
        :param i: Index of the neuron on previous layer
        :return: ndarray with the current as a function of distance
        """
        if self.prev_layer is not None:
            t_0 = self.prev_layer[i].calc_t_end()
            dist = self.distance_to_i(i)
            duration = (dist / v) * 1000
            # print("Duration ", duration)
            start_filler = [0 for i in range(int(accuracy * ((t_0 - signal_t_start) / 1000)))]
            end_filler = [0 for i in range(int(accuracy * ((signal_t_end - (t_0 + duration)) / 1000)))]

            if self.prev_layer[i].activation is not None:
                I_0 = self.prev_layer[i].activation * I_max
            else:
                I_0 = self.prev_layer[i].calc_activation() * I_max

            # print("I_0 ", I_0)

            l = dist * length_coef

            x_0 = np.linspace(0, int(duration), int(accuracy * (duration / 1000)))
            I_x = I_0 * np.exp(-(x_0 / l))

            I_final = start_filler + I_x.tolist() + end_filler

            # Check that the length is correct. This might vary by a few do to type conversions
            required_length = len(x)
            current_length = len(I_final)
            # print(required_length, current_length)
            if current_length > required_length:
                # Remove values from the end to make lengths match
                I_final = I_final[:required_length]
            elif current_length < required_length:
                # Add zeroes to the end to make lengths match
                filler = [0 for i in range(required_length - current_length)]
                I_final = I_final + filler

            # print("Final length", len(I_final))
        else:
            raise Exception("Current cannot be calculated without previous layer")

        return np.array(I_final)

    def I_tot(self):
        """
        Combine all incoming currents into a single signal
        :return: ndarray with the combined current as a function of distance
        """
        if self.prev_layer is not None:
            currents = [self.I_i(i) for i in range(len(self.prev_layer))]
            combined = np.sum(currents, axis=0)
        else:
            raise Exception("Current cannot be calculated without previous layer")

        return combined

