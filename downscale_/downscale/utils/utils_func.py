import numpy as np
import pandas as pd
import datetime

from downscale.utils.decorators import timer_decorator


def select_range(month_begin, month_end, year_begin, year_end, date_begin, date_end):
    import pandas as pd
    if (month_end != month_begin) or (year_begin != year_end):
        dates = pd.date_range(date_begin, date_end, freq='M')
        iterator = zip(dates.day, dates.month, dates.year)
    else:
        dates = pd.to_datetime(date_end)
        iterator = zip([dates.day], [dates.month], [dates.year])
    return iterator


def select_range_7days_for_long_periods_prediction(begin="2017-8-2", end="2020-6-30", prm=None):
    """
    This function takes as input a date range (begin and end) and split it in 7-days range around excluded dates

    Works if we have only one splitting in a week
    """

    begin = np.datetime64(pd.to_datetime(begin))
    end = np.datetime64(pd.to_datetime(end))

    # Define 7 days periods within date range
    dates = pd.date_range(start=begin, end=end, freq="7D")
    dates_shift = pd.date_range(start=begin, end=end, freq="7D").shift()
    dates_shift = dates_shift.where(dates_shift <= end, [end])

    # Split range around selected dates
    if prm["GPU"]:
        d1 = datetime.datetime(2017, 8, 1, 6)
        d2 = datetime.datetime(2018, 8, 1, 6)
        d3 = datetime.datetime(2019, 5, 1, 6)
        d4 = datetime.datetime(2019, 6, 1, 6)
        d5 = datetime.datetime(2020, 6, 2, 6)
        splitting_dates = [np.datetime64(date) for date in [d1, d2, d3, d4, d5]]
    else:
        d1 = datetime.datetime(2017, 8, 1, 6)
        d2 = datetime.datetime(2018, 8, 1, 6)
        d3 = datetime.datetime(2019, 6, 1, 6)
        d6 = datetime.datetime(2020, 7, 1, 6)
        splitting_dates = [np.datetime64(date) for date in [d1, d2, d3, d6]]

    begins = []
    ends = []
    for index, (begin, end) in enumerate(zip(dates.values, dates_shift.values)):

        # Add one day to begin after first element
        begin = begin if index == 0 else begin + np.timedelta64(1, "D")
        end = end + np.timedelta64(23, "h")

        if begin > end:
            continue

        split = False
        for splt_date in splitting_dates:

            # If date range needs to be splitted
            if begin <= splt_date < end:
                begins.append(begin)
                ends.append(splt_date - np.timedelta64(1, "h"))
                begins.append(splt_date)
                ends.append(end)
                split = True

        # If we didn't split date range
        if not split:
            begins.append(begin)
            ends.append(end)

    begins = [pd.to_datetime(begin) for begin in begins]
    ends = [pd.to_datetime(end) for end in ends]

    return begins, ends


def select_range_30_days_for_long_periods_prediction(begin="2017-8-2", end="2020-6-30", GPU=False):

    begin = np.datetime64(pd.to_datetime(begin))
    end = np.datetime64(pd.to_datetime(end))

    # Define 30 days periods within date range
    dates = pd.date_range(start=begin, end=end, freq="MS")
    dates_shift = pd.date_range(start=begin, end=end, freq="M", closed='right').shift()
    dates_shift = dates_shift.where(dates_shift <= end, [end])

    # Split range around selected dates
    if not GPU:
        d1 = datetime.datetime(2017, 8, 1, 6)
        d2 = datetime.datetime(2018, 8, 1, 6)
        d3 = datetime.datetime(2019, 6, 1, 6)
        d6 = datetime.datetime(2020, 7, 1, 6)
        splitting_dates = [np.datetime64(date) for date in [d1, d2, d3, d6]]
    else:
        d1 = datetime.datetime(2017, 8, 1, 6)
        d2 = datetime.datetime(2018, 8, 1, 6)
        d3 = datetime.datetime(2019, 5, 1, 6)
        d4 = datetime.datetime(2019, 6, 1, 6)
        d5 = datetime.datetime(2020, 6, 2, 6)
        splitting_dates = [np.datetime64(date) for date in [d1, d2, d3, d4, d5]]


    begins = []
    ends = []
    for index, (begin, end) in enumerate(zip(dates.values, dates_shift.values)):

        # Add one day to begin after first element
        end = end + np.timedelta64(23, "h")
        split = False
        for splt_date in splitting_dates:

            # If date range needs to be splitted
            if begin <= splt_date < end:
                begins.append(begin)
                ends.append(splt_date - np.timedelta64(1, "h"))
                begins.append(splt_date)
                ends.append(end)
                split = True

        # If we didn't split date range
        if not split:
            begins.append(begin)
            ends.append(end)

    # begins = [pd.to_datetime(begin) for begin in begins]
    for index, begin in enumerate(begins):
        if not isinstance(begin, str):
            begins[index] = pd.to_datetime(begin)
    # ends = [pd.to_datetime(end) for end in ends]

    for index, end in enumerate(ends):
        if not isinstance(end, str):
            ends[index] = pd.to_datetime(end)

    return begins, ends


def print_current_line(time_step, nb_sim, division):
    nb_sim_divided = nb_sim // division
    for k in range(1, division + 1):
        print(f" {k}/{division}") if (time_step == k * nb_sim_divided) else True


def change_dtype_if_required(variable, dtype):
    if variable.dtype != dtype:
        variable = variable.astype(dtype, copy=False)
    return variable


def change_several_dtype_if_required(list_variable, dtypes):
    result = []
    for variable, dtype in zip(list_variable, dtypes):
        if isinstance(variable, (list, int, float)):
            variable = np.array(variable)
        result.append(change_dtype_if_required(variable, dtype))
    return result


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
    return np.round(t2 - t1, 2)


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
    return result


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


@timer_decorator("statistical description array", unit="minute", level="")
def print_statistical_description_array(array, name="Acceleration CNN", level="________"):

    print(f"{level}{name} min", np.nanmin(array))
    print(f"\n{level}{name} q0.10", np.nanquantile(array, 0.1))
    print(f"\n{level}{name} q0.25", np.nanquantile(array, 0.25))
    print(f"\n{level}{name} median", np.nanmedian(array))
    print(f"\n{level}{name} q0.75", np.nanquantile(array, 0.75))
    print(f"\n{level}{name} q0.90", np.nanquantile(array, 0.9))
    print(f"\n{level}{name} q0.95", np.nanquantile(array, 0.95))
    print(f"\n{level}{name} q0.99", np.nanquantile(array, 0.99))
    print(f"\n{level}{name} maximum", np.nanmax(array))

    return None


def print_with_frame(text):
    print('\n\n__________________________')
    print('__________________________\n')
    print(f'_______{text}_______\n')
    print('__________________________')
    print('__________________________\n\n')


def print_begin_end(begin, end):
    print('\n\n__________________________')
    print('__________________________\n')
    print(f'_______{begin}___________\n')
    print(f'_______{end}___________\n')
    print('__________________________')
    print('__________________________\n\n')


def print_second_begin_end(begin, end):
    print('\n__________________________')
    print(f'____{begin}___')
    print(f'____{end}___')
    print('__________________________')


def print_intro():

    intro = """
                                            ''' '
                                          '   ' '
                                     ''' ''' '''
                              + hs    ' '''''  '.' '   
                            'shh  ho            '   '   
                           .yhhh  hh+           ' ''  
                          /hhhs    +hhh/         
                          hhhh'     hhhh         '''
                         ohhho      +hhh:       '.  '.' 
                       'yhhh:        ohhh: ''''' ''' .  
               .+.    -hhhy.          ohhh:  '  ''''' ''
              -hhho' /hhhs'            ohhh:  ''''''''' 
             :hhhhhhyhhh+               ohhh/      .' ''
            /hhho+hhhhh:                 +hhh+    '. '.'
           +hhh+  '+hy                    /hhho     ''  
          ohhh/     '                       :hhhs'       
        'shhh:                               :yhhy-      
       gyhhhg           Wind speed            'shhh/     
      hyhhyf                                   +hhhs'   
     :hhhs'             Downscaling              -hhhh:  
    +hhho                                       'ohhhsg
    hhh/                 using CNN                  :yhhh
    hy-                                              '+hh
    o'              by Louis Le Toumelin               .s

                     CEN - Meteo-France
    """
    print(intro)
