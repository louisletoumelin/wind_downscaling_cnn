import numpy as np
from time import time as t


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
            if result.dtype != dtype:
                result = result.astype(dtype, copy=False)
            return result
        return wrapper
    return decorator