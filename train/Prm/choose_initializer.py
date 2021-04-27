'''
Modify Prm
'''
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.initializers import lecun_uniform
from tensorflow.keras.initializers import lecun_normal


def create_prm_initializer(prm):

    if prm['initializer'] is None:
        prm['initializer_func'] = None

    if prm['initializer'] == 'glorot_normal':
        prm['initializer_func'] = glorot_normal()

    if prm['initializer'] == 'lecun_uniform':
        prm['initializer_func'] = lecun_uniform()

    if prm['initializer'] == 'lecun_normal':
        prm['initializer_func'] = lecun_normal()


    return (prm)
