import pandas as pd
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from utils_preprocess import filenames_to_array
from utils_preprocess import train_valid

def define_best_cluster(nb_clusters_max, all_topo):
    """Determines the best number of clusters between 2 and nb_clusters_max to perform MiniBatchKMeans"""
    best_clusters = 0
    previous_silh_avg = 0.0
    for n_clusters in range(2, nb_clusters_max + 1):

        clusterer = MiniBatchKMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(all_topo)
        silhouette_avg = silhouette_score(all_topo, cluster_labels)

        if silhouette_avg > previous_silh_avg:
            previous_silh_avg = silhouette_avg
            best_clusters = n_clusters
    return (best_clusters)


def KMeans_training_prediction(all_topo, df_all, nb_clusters_max = 10):
    """Categorisation of topographies"""
    best_clusters = define_best_cluster(nb_clusters_max, all_topo)
    kmeans = MiniBatchKMeans(n_clusters=best_clusters).fit(all_topo)

    labels = kmeans.predict(all_topo)
    df_all['class_nb'] = labels
    return(df_all, best_clusters)


def create_specific_folder_by_class_nb(class_nb, output_dir):
    """Create subfolder for each class_nb"""
    newpath = output_dir + '/class_nb_' + f"/class_nb{class_nb}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def train_test_class_nb(class_to_exclude, df_all):
    """Split train and test"""
    df_all['group'] = 0
    df_all['group'][df_all['class_nb'] == class_to_exclude] = 'test'
    df_all['group'][df_all['class_nb'] != class_to_exclude] = 'train'
    return(df_all)


def train_validation_test_split_class_nb(df_all, best_clusters, input_dir):
    for class_to_exclude in range(best_clusters):
        df_all = train_test_class_nb(class_to_exclude, df_all)
        df_all = train_valid(df_all)

        # Create subfolder
        create_specific_folder_by_class_nb(class_to_exclude, input_dir)

        # Save file
        output = input_dir + '/class_nb_' + f"/class_nb{class_to_exclude}"
        df_all.to_csv(output + f'/df_all_class_excluded_{class_to_exclude}.csv')

def preprocess_class_nb(input_dir, nb_clusters_max):
    """Preprocessing funciton for class_nb"""
    # Read file
    df_all = pd.read_csv(input_dir + 'fold/' + 'df_all.csv')

    # Read data
    all_topo = filenames_to_array(df_all, input_dir, 'topo_name')
    all_topo = all_topo.reshape(len(all_topo), 79 * 69)

    # Predict class_nb
    df_all, best_clusters = KMeans_training_prediction(all_topo, df_all, nb_clusters_max=nb_clusters_max)

    # Check no group already exists
    try: df_all = df_all.drop('group', axis=1)
    except: pass

    # Split data and save file
    train_validation_test_split_class_nb(df_all, best_clusters, input_dir)
