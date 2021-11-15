import numpy as np


class Generators:

    def __init__(self):
        pass

    @staticmethod
    def generator_divide_array_in_n_bactches(array, n=10):
        nb_elements = array.shape[0]
        list_inputs = np.linspace(0, nb_elements, n)
        for i, input_ in enumerate(list_inputs):
            if input_ != list_inputs[-1]:
                begin = np.intp(list_inputs[i])
                end = np.intp(list_inputs[i+1])
                yield array[begin:end, :]

    @staticmethod
    def split_array(array):
        nb_elements = array.shape[0]
        return array[:nb_elements//2, :], array[nb_elements//2:, :]

    @staticmethod
    def generator_single_element(array):
        for i, _ in enumerate(array):
            yield array[i, :]

    @staticmethod
    def create_callable_generator(array):
        def _generator_single_element():
            for i, _ in enumerate(array):
                yield array[i, :]
        return _generator_single_element
