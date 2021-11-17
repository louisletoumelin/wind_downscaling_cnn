import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from Metrics import metrics
from Prm import choose_optimizer, save_prm, create_prm_2, choose_initializer
from Models import choose_model
from Type_of_training import choose_training
from Utils import check_GPU, new_folders, print_prm_info
from Test import test_pipeline

'''
Script
'''


print_prm_info.print_intro()

# prm contains all information about training experience
list_prm = create_prm_2.create_prm_dict()

start_all = time.perf_counter()
for index, prm in enumerate(list_prm):

    # Timer
    start_prm = time.perf_counter()

    # Create name_simu and info key
    prm = save_prm.create_name_simu_and_info(index, prm)

    # Information for user
    print_prm_info.print_prm_information(prm)

    # Create a folder specific to the experience
    new_folders.create_specific_folder(prm)

    # Save prm dictionary in a file
    save_prm.save_on_disk(prm)

    # Update the file containing all training experiences
    save_prm.update_training_record(prm)

    # Check connection with GPU
    check_GPU.check_connection_GPU(prm)

    # Metrics
    prm = metrics.create_prm_metrics(prm)

    # Dependencies
    prm = metrics.create_dependencies(prm)

    # Optimizer
    prm = choose_optimizer.create_prm_optimizer(prm)

    # Initializer
    prm = choose_initializer.create_prm_initializer(prm)

    # Model
    prm = choose_model.create_prm_model(prm)

    # Type of training
    train_model = choose_training.select_type_of_training(prm)

    # Training
    trained_model, history = train_model(prm)

    # Test
    if prm['type_of_training'] != 'all':
        test_pipeline.predict_test(prm)

    finish_prm = time.perf_counter()
    print(f'\nPRM finished in {round((start_prm - finish_prm) / 60, 2)} minute(s)')

    # Finished
    print_prm_info.print_finish()

finish_all = time.perf_counter()
print(f'\nFinished in {round((start_all - finish_all) / 60, 2)} minute(s)')
print_prm_info.print_end_of_job()
print_prm_info.print_end_of_training()
