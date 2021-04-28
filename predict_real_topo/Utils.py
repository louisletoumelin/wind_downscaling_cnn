def connect_GPU_to_horovod():
    import horovod.tensorflow.keras as hvd
    import tensorflow as tf
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    print(gpus)


def select_range(month_begin, month_end, year_begin, year_end, date_begin, date_end):
    import pandas as pd
    if (month_end != month_begin) or (year_begin != year_end):
        dates = pd.date_range(date_begin, date_end, freq='M')
        iterator = zip(dates.day, dates.month, dates.year)
    else:
        dates = pd.to_datetime(date_end)
        iterator = zip([dates.day], [dates.month], [dates.year])
    return(iterator)


def check_save_and_load(load_z0, save_z0):
    if load_z0 and save_z0:
        load_z0 = True
        save_z0 = False
    return(load_z0, save_z0)


def print_current_line(time_step, nb_sim, division):
    nb_sim_divided = nb_sim // division
    for k in range(1, division+1):
        if time_step == k * nb_sim_divided: print(f" {k}/{division}")