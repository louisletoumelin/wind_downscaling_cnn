from Type_of_training import training_on_folds
from Type_of_training import training_on_xi
from Type_of_training import training_on_class
from Type_of_training import training_on_degree
from Type_of_training import final_training


def select_type_of_training(prm):
    """Select the type of training"""
    if prm['type_of_training'] == 'fold':
        train_model = training_on_folds.train_model
    if prm['type_of_training'] == 'class':
        train_model = training_on_class.train_model
    if prm['type_of_training'] == 'degree':
        train_model = training_on_degree.train_model
    if prm['type_of_training'] == 'xi':
        train_model = training_on_xi.train_model
    if prm['type_of_training'] == 'all':
        train_model = final_training.train_model
    return (train_model)
