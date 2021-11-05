import numpy as np


class Generators:

    def __init__(self):
        pass

    @staticmethod
    def create_generator(array):
        nb_elements = array.shape[0]
        list_inputs = np.linspace(0, nb_elements, 10)
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
    def create_callable_generator(array):
        def create_generator_2():
            for i, _ in enumerate(array):
                yield array[i, :]
        return create_generator_2
