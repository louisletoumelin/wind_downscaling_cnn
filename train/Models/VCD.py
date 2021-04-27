from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Flatten, BatchNormalization, LeakyReLU, \
    LSTM, UpSampling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D, UpSampling1D


def build_VCD(prm):
    model = Sequential()

    model.add(ZeroPadding2D(padding=((0, 1), (0, 1)), input_shape=prm['input_shape']))

    # CONVOLUTION
    model.add(Conv2D(prm['nb_filters'], (5, 5), activation=prm['activation'], padding=prm['padding'],
                     kernel_initializer=prm['initializer_func']))
    model.add(Conv2D(prm['nb_filters'], (5, 5), activation=prm['activation'], padding=prm['padding'],
                     kernel_initializer=prm['initializer_func']))
    model.add(MaxPool2D(pool_size=prm['pool_size']))
    model.add(BatchNormalization())
    model.add(Dropout(prm['dropout']))

    model.add(Conv2D(2 * prm['nb_filters'], (5, 5), activation=prm['activation'], padding=prm['padding'],
                     kernel_initializer=prm['initializer_func']))
    model.add(Conv2D(2 * prm['nb_filters'], (5, 5), activation=prm['activation'], padding=prm['padding'],
                     kernel_initializer=prm['initializer_func']))
    model.add(MaxPool2D(pool_size=prm['pool_size']))
    model.add(BatchNormalization())
    model.add(Dropout(prm['dropout']))

    model.add(Conv2D(4 * prm['nb_filters'], (3, 3), activation=prm['activation'], padding=prm['padding'],
                     kernel_initializer=prm['initializer_func']))
    model.add(Conv2D(4 * prm['nb_filters'], (3, 3), activation=prm['activation'], padding=prm['padding'],
                     kernel_initializer=prm['initializer_func']))
    model.add(MaxPool2D(pool_size=prm['pool_size']))
    model.add(BatchNormalization())
    model.add(Dropout(prm['dropout']))

    model.add(Conv2D(8 * prm['nb_filters'], (3, 3), activation=prm['activation'], padding=prm['padding'],
                     kernel_initializer=prm['initializer_func']))
    model.add(Conv2D(8 * prm['nb_filters'], (3, 3), activation=prm['activation'], padding=prm['padding'],
                     kernel_initializer=prm['initializer_func']))
    model.add(BatchNormalization())
    model.add(Dropout(prm['dropout']))

    # DECONVOLUTION
    model.add(Conv2DTranspose(8 * prm['nb_filters'], (3, 3), activation=prm['activation'], padding=prm['padding'],
                              kernel_initializer=prm['initializer_func']))
    model.add(Conv2DTranspose(8 * prm['nb_filters'], (3, 3), activation=prm['activation'], padding=prm['padding'],
                              kernel_initializer=prm['initializer_func']))
    model.add(BatchNormalization())
    model.add(Dropout(prm['dropout']))

    model.add(UpSampling2D(size=prm['up_conv']))
    model.add(ZeroPadding2D(padding=((0, 0), (0, 1))))

    model.add(Conv2DTranspose(4 * prm['nb_filters'], (3, 3), activation=prm['activation'], padding=prm['padding'],
                              kernel_initializer=prm['initializer_func']))
    model.add(Conv2DTranspose(4 * prm['nb_filters'], (3, 3), activation=prm['activation'], padding=prm['padding'],
                              kernel_initializer=prm['initializer_func']))
    model.add(UpSampling2D(size=prm['up_conv']))
    model.add(ZeroPadding2D(padding=((0, 0), (0, 1))))
    model.add(BatchNormalization())
    model.add(Dropout(prm['dropout']))

    model.add(Conv2DTranspose(2 * prm['nb_filters'], (5, 5), activation=prm['activation'], padding=prm['padding'],
                              kernel_initializer=prm['initializer_func']))
    model.add(Conv2DTranspose(2 * prm['nb_filters'], (5, 5), activation=prm['activation'], padding=prm['padding'],
                              kernel_initializer=prm['initializer_func']))
    model.add(UpSampling2D(size=prm['up_conv']))
    model.add(BatchNormalization())
    model.add(Dropout(prm['dropout']))

    model.add(Conv2DTranspose(prm['nb_filters'], (5, 5), activation=prm['activation'], padding=prm['padding'],
                              kernel_initializer=prm['initializer_func']))
    model.add(Conv2DTranspose(prm['nb_filters'], (5, 5), activation=prm['activation'], padding=prm['padding'],
                              kernel_initializer=prm['initializer_func']))
    model.add(Dropout(prm['dropout']))

    model.add(Conv2DTranspose(prm['nb_channels_output'], (5, 5), activation=prm['activation_regression'],
                              padding=prm['padding'], kernel_initializer=prm['initializer_func']))
    model.add(Cropping2D(cropping=((0, 1), (0, 1))))

    return (model)


def VCD(prm):
    model = build_VCD(prm)
    print('\n Model selected: VCD\n')
    print(model.summary())
    return (model)
