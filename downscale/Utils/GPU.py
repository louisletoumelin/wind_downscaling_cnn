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