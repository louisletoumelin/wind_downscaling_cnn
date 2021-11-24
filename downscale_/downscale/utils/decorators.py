import numpy as np
from time import time as t
from functools import wraps


def print_func_executed_decorator(argument, level_begin="", level_end="", end=""):
    def decorator(function):
        def wrapper(*args, **kwargs):
            print(f"{level_begin}Begin {argument}")
            result = function(*args, **kwargs)
            print(f"{level_end}End {argument}{end}")
            return result
        return wrapper
    return decorator


def timer_decorator(argument, unit='minute', level="__"):
    def decorator(function):
        def wrapper(*args, **kwargs):
            t0 = t()
            result = function(*args, **kwargs)
            t1 = t()
            if unit == "hour":
                time_execution = np.round((t1 - t0) / (3600), 2)
            elif unit == "minute":
                time_execution = np.round((t1-t0) / 60, 2)
            elif unit == "second":
                time_execution = np.round((t1 - t0), 2)
            print(f"{level}Time to calculate {argument}: {time_execution} {unit}s")
            return result
        return wrapper
    return decorator


def change_dtype_if_required_decorator(dtype):
    def decorator(function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)

            if isinstance(result, list):
                result = np.array(result)

            if result.dtype != dtype:
                result = result.astype(dtype, copy=False)

            return result
        return wrapper
    return decorator


def check_type_kwargs_inputs(dict_argument_type):
    def decorator(func):
        @wraps(func)
        def type_decorator(*args, **kwargs):
            for key, value in kwargs.items():
                if key in dict_argument_type:
                    if type(value) not in dict_argument_type[key]:
                        error_message = f"Input type {type(value)} for {key}, expected {dict_argument_type[key]}"
                        raise NotImplementedError(error_message)
            return func(*args, **kwargs)
        return type_decorator
    return decorator
