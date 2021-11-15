from tensorflow.keras import backend as K


def root_mse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def nrmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) / (K.max(y_pred) - K.min(y_pred))
