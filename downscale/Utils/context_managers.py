import numpy as np
from contextlib import contextmanager
from time import time as t


@contextmanager
def timer_context(argument, level=None, unit=None, verbose=True):
    if verbose:
        t0 = t()
        yield
        t1 = t()
        if unit == "hour":
            time_execution = np.round((t1 - t0) / (3600), 2)
        elif unit == "minute":
            time_execution = np.round((t1 - t0) / 60, 2)
        elif unit == "second":
            time_execution = np.round((t1 - t0), 2)
        print(f"{level}Time to calculate {argument}: {time_execution} {unit}s")
    else:
        yield
