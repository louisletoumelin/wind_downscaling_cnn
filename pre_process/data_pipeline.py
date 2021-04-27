import create_df_without_train_test
import preprocess_folds
import preprocess_class_nb
import preprocess_degree_xi
import utils_preprocess

"""
Parameters
"""

input_dir = "//scratch/mrmn/letoumelinl/ARPS/"
nb_clusters_max = 10

"""
Pipeline
"""


# Create df_all
df_all = create_df_without_train_test.create_df_with_simulation_name(input_dir)
utils_preprocess.print_execution('df_all')

# Preprocess folds
preprocess_folds.preprocess_folds(df_all, input_dir)
utils_preprocess.print_execution('folds')

# Preprocess class_nb
preprocess_class_nb.preprocess_class_nb(input_dir, nb_clusters_max)
utils_preprocess.print_execution('class_nb')

# Preprocess degree_xi
preprocess_degree_xi.preprocess_degree_xi(input_dir)
utils_preprocess.print_execution('degree_xi')
