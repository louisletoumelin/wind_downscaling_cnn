import os


def create_specific_folder(prm):
    newpath = prm['output_dir'] + 'training_results/' + prm['info']
    if not os.path.exists(newpath):
        os.makedirs(newpath)
