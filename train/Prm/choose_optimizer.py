'''
Modify Prm
'''
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam


def create_prm_optimizer(prm):
    if prm['optimizer'] == 'RMSprop':
        prm['optimizer_func'] = RMSprop(lr=prm['learning_rate'], decay=prm['decay'])

    if prm['optimizer'] == 'Adam':
        prm['optimizer_func'] = Adam(lr=prm['learning_rate'], decay=prm['decay'])

    if prm['optimizer'] == 'AMSgrad':
        prm['optimizer_func'] = Adam(lr=prm['learning_rate'], decay=prm['decay'], amsgrad=True)

    if prm['optimizer'] == 'Adamax':
        prm['optimizer_func'] = Adamax(lr=prm['learning_rate'], decay=prm['decay'])

    if prm['optimizer'] == 'Nadam':
        prm['optimizer_func'] = Nadam(lr=prm['learning_rate'], decay=prm['decay'])

    return (prm)
