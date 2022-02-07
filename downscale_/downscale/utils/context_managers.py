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


@contextmanager
def creation_context(argument, level=None, verbose=True):
    if verbose:
        print(f"\n{level}Begin calculating {argument}")
        yield
        print(f"{level}End calculating {argument}\n")
    else:
        yield


@contextmanager
def print_all_context(argument, level=0, unit=None, verbose=True):
    if verbose:
        t0 = t()
        level_creation = '_' * level
        level_time = '. ' * level
        print(f"\n{level_creation}Begin calculating {argument}")
        yield
        t1 = t()
        if unit == "hour":
            time_execution = np.round((t1 - t0) / (3600), 2)
        elif unit == "minute":
            time_execution = np.round((t1 - t0) / 60, 2)
        elif unit == "second":
            time_execution = np.round((t1 - t0), 2)
        print(f"{level_time}Time to calculate {argument}: {time_execution} {unit}s")
        print(f"{level_creation}End calculating {argument}\n")
    else:
        yield
