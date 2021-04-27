import tensorflow as tf


def check_connection_GPU(prm):
    """Check tensorflow is connected to GPU"""
    if prm['GPU']:
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('\n\nFound GPU at: {}\n\n'.format(device_name))