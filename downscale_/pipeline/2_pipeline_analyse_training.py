import uuid

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context('paper', font_scale=1.4)
except ImportError:
    pass

from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.utils.GPU import connect_GPU_to_horovod
from downscale.eval.synthetic_topographies import GaussianTopo
from downscale.eval.metrics import Metrics
from downscale.operators.devine import Devine
from downscale.utils.dependencies import root_mse
from downscale.visu.visualization import Visualization
from downscale.eval.eval_training import figure_training_result_by_metrics, figure_epochs_bias, figure_heatmap_training


#
#
# Figure 1: Bias = f(metric)
#
#

config_figure_training = dict(
    tpi_500=False,
    sx_300=False,
    save_dataframe=False,
    load_dataframe=True,
    curvature=False,
    laplacian=False,
    mu=False,
    name_dataframe_on_gaussian_topo_with_metrics="dataframe_on_gaussian_topo_with_metrics_d182.pkl",
    metric_already_loaded=True,
    fontsize=45,
    return_df_results=True
)
results = figure_training_result_by_metrics(config_figure_training, prm)



#
#
# Figure 2: Bias = f(epochs)
#
#

#figure_epochs_bias(prm)


#
#
# Figure 3: Heatmap errors
#
#
#figure_heatmap_training(prm)









"""
# General boxplot
ax = sns.boxplot(data=test, y="bias", showfliers=False)
plt.axis("square")

# Boxplot fct class_sx_300, saved = True
#order = ['sx_300 <= q25', 'q25 < sx_300 <= q50', 'q50 < sx_300 <= q75', 'q75 < sx_300']
#test = gaussian_topo.classify(test, variable="sx_300", quantile=True)
#ax = sns.boxplot(data=test, x="class_sx_300", y="bias", showfliers=False, order=order)

# Boxplot fct class_tpi_500, saved = True
order = ['tpi_500 <= q25', 'q25 < tpi_500 <= q50', 'q50 < tpi_500 <= q75', 'q75 < tpi_500']
test = gaussian_topo.classify(test, variable="tpi_500", quantile=True)
ax = sns.boxplot(data=test, x="class_tpi_500", y="bias", showfliers=False, order=order)

# Boxplot fct mu, saved = True
order = ['mu <= q25', 'q25 < mu <= q50', 'q50 < mu <= q75', 'q75 < mu']
test = gaussian_topo.classify(test, variable="mu", quantile=True)
ax = sns.boxplot(data=test, x="class_mu", y="bias", showfliers=False, order=order)

# Boxplot fct curvature, saved = True
order = ['curvature <= q25', 'q25 < curvature <= q50', 'q50 < curvature <= q75', 'q75 < curvature']
test = gaussian_topo.classify(test, variable="curvature", quantile=True)
ax = sns.boxplot(data=test, x="class_curvature", y="bias", showfliers=False, order=order)

# Boxplot fct degree, saved = True
ax = sns.boxplot(data=test, x="degree", y="bias", showfliers=False)

# Boxplot fct xi, saved = True
ax = sns.boxplot(data=test, x="xi", y="bias", showfliers=False)

# Distribution bias, save = True
sns.displot(test, x="bias", kind="kde"), saved = True
plt.title("Bias distribution on the test dataset (Gaussian topographies)")

# Distribution bias by xi
sns.displot(test, x="bias", kind="kde", hue="xi")
plt.title("Test dataset: Bias distribution by xi")

sns.displot(test, x="bias", kind="kde", hue="degree")
plt.title("Test dataset: Bias distribution by degree")

"""




