import numpy as np


def select_range(month_begin, month_end, year_begin, year_end, date_begin, date_end):
    import pandas as pd
    if (month_end != month_begin) or (year_begin != year_end):
        dates = pd.date_range(date_begin, date_end, freq='M')
        iterator = zip(dates.day, dates.month, dates.year)
    else:
        dates = pd.to_datetime(date_end)
        iterator = zip([dates.day], [dates.month], [dates.year])
    return (iterator)


def check_save_and_load(load_z0, save_z0):
    if load_z0 and save_z0:
        save_z0 = False
    return (load_z0, save_z0)


def print_current_line(time_step, nb_sim, division):
    nb_sim_divided = nb_sim // division
    for k in range(1, division + 1):
        print(f" {k}/{division}") if (time_step == k * nb_sim_divided) else True


def change_dtype_if_required(variable, dtype):
    if variable.dtype != dtype:
        variable = variable.astype(dtype, copy=False)
    return (variable)


def change_several_dtype_if_required(list_variable, dtypes):
    result = []
    for variable, dtype in zip(list_variable, dtypes):
        result.append(change_dtype_if_required(variable, dtype))
    return(result)


def change_dtype_decorator(dtype):
    """Timer decorator"""
    def decorator(function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            result = change_dtype_if_required(result, dtype)
            return result
        return wrapper
    return decorator


def assert_equal_shapes(arrays, shape):
    for k in range(len(arrays) - 1):
        assert arrays[k].shape == shape


def round(t1, t2):
    return (np.round(t2 - t1, 2))


def reshape_list_array(list_array=None, shape=None):
    """
    Utility function that takes as input a list of arrays to reshape to the same shape

    Parameters
    ----------
    list_array : list
        List of arrays
    shape : tuple
        typle of shape

    Returns
    -------
    result : list
        List of reshaped arrays
    """
    result = []
    for array in list_array:
        result.append(np.reshape(array, shape))
    return (result)


def several_empty_like(array_like, nb_empty_arrays=None):
    result = []
    for array in range(nb_empty_arrays):
        result.append(np.empty_like(array_like))
    return result


def _list_to_array_if_required(list_or_array):
    if isinstance(list_or_array, list):
        return np.array(list_or_array)
    else:
        return list_or_array


def lists_to_arrays_if_required(lists_or_arrays):
    if np.ndim(lists_or_arrays) > 1:
        return (_list_to_array_if_required(list_or_array) for list_or_array in lists_or_arrays)
    else:
        return _list_to_array_if_required(lists_or_arrays)
