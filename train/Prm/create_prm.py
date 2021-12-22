import itertools

'''
prm
'''


def create_prm_dict():
    """
    Create several dictionaries containing information about the training parameters.

    Input: Parameters (defined inside the function)
    Outputs: [prm1, prm2, ...] with element being a dictionary with info about training
    """
    prms = {
        # Necessary
        'date': ['21_12_2021'],
        # 'VCD' or 'UNet'
        'model': ['UNet'],
        # Specify only one name even if multiple prm are contained in prms
        'name_simu': ['classic_all_low_epochs'],
        # 'fold', 'class', 'degree', 'xi', 'all'
        'type_of_training': ['all'],

        # General
        'loss': ["mse"],
        'learning_rate': [0.001],
        'decay': [0.0001],
        # 'RMSprop' 'Adam' 'AMSgrad' 'Adamax' 'Nadam'
        'optimizer': ['RMSprop'],
        'list_metrics': [['mae', 'root_mse']],
        'epochs': [48],  # 150, 48 for classic epochs after early stopping, 40 for no_dropout

        'batch_size': [32],
        'additional_flat_topo': [False],

        # Reduce on plateau
        'ROP_factor': [0.1],
        'ROP_patience': [5],
        'ROP_min_lr': [1e-10],

        # Convolution
        'kernel_size': [(3, 3)],  # (3,3)
        'padding': ['same'],
        'nb_filters': [32],
        # Initializer
        # Default = glorot_uniform_initializer, 'glorot_normal', 'lecun_uniform', 'lecun_normal'
        'initializer': [None],
        # Up conv
        'up_conv': [(2, 2)],
        # Activation
        # 'relu', 'elu', 'selu'
        'activation': ['relu'],
        'activation_regression': ['linear'],
        # Pooling, batch norm and dropout
        'pool_size': [(2, 2)], # (2, 2)
        'minimal_dropout_layers': [True], # True
        'full_dropout': [False],  # False
        'dropout': [0.25],
        'full_batch_norm': [False],  # False
        'early_stopping_patience': [15],
        'early_stopping_min_delta': [0.0001],

        # Other
        'n_rows': [79],
        'n_col': [69],
        'input_shape': [(79, 69, 1)],
        'output_shape': [(79, 69, 3)],
        'nb_channels_output': [3],
        'input_dir': ["//home/mrmn/letoumelinl/train"],
        'output_dir': ["//scratch/mrmn/letoumelinl/ARPS/"],
        'GPU': [True]

    }

    keys, values = zip(*prms.items())
    list_prm = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return (list_prm)
