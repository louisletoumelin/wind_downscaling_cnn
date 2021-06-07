def connect_GPU_to_horovod():
    import horovod.tensorflow.keras as hvd
    import tensorflow as tf
    tf.keras.backend.clear_session()
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def environment_GPU(GPU=True):
    if GPU:
        import os
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # tf.get_logger().setLevel('WARNING')
        # tf.autograph.set_verbosity(0)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # tf.debugging.set_log_device_placement(True)


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
        load_z0 = True
        save_z0 = False
    return (load_z0, save_z0)


def print_current_line(time_step, nb_sim, division):
    nb_sim_divided = nb_sim // division
    for k in range(1, division + 1):
        if time_step == k * nb_sim_divided: print(f" {k}/{division}")


def change_dtype_if_required(variable, dtype):
    if variable.dtype != dtype:
        variable = variable.astype(dtype, copy=False)
    return (variable)


def change_several_dtype_if_required(list_variable, dtypes):
    result = []
    for variable, dtype in zip(list_variable, dtypes):
        result.append(change_dtype_if_required(variable, dtype))
    return(result)


def assert_equal_shapes(arrays, shape):
    assert arrays[0].shape == shape
    for k in range(len(arrays) - 1):
        assert arrays[k].shape == arrays[k + 1].shape