import tensorflow as tf
print(tf.__version__)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

except:
    try:
        import tensorflow as tf
        tf.config.gpu.set_per_process_memory_growth(True)
        print(2)
    except:
        print(3)
