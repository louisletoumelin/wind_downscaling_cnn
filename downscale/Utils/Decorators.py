import numpy as np

def print_func_executed_decorator(argument):
    def decorator(function):
        def wrapper(*args, **kwargs):
            print(f"Begin {argument}")
            result = function(*args, **kwargs)
            print(f"End {argument}\n")
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