import tensorflow as tf
import os


def connect_GPU_to_horovod(verbose=True):
    import horovod.tensorflow.keras as hvd
    import tensorflow as tf
    tf.keras.backend.clear_session()
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    if verbose:
        print("____'TF_FORCE_GPU_ALLOW_GROWTH' = True")
        print("____tf.config.experimental.set_memory_growth = True")
        print("____tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')")


def environment_GPU(GPU=True):
    if GPU:
        import os
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # tf.get_logger().setLevel('WARNING')
        # tf.autograph.set_verbosity(0)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        # tf.debugging.set_log_device_placement(True)


def check_connection_GPU(prm):
    """Check tensorflow is connected to GPU"""
    if prm['GPU']:
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))


def connect_on_GPU(prm):
    if prm["GPU"]:
        print("\nBegin connection on GPU") if prm["verbose"] else None
        connect_GPU_to_horovod()
        check_connection_GPU(prm)
        print("End connection on GPU\n") if prm["verbose"] else None
