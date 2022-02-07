import uuid

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from downscale.eval.synthetic_topographies import GaussianTopo
from downscale.eval.metrics import Metrics
from downscale.operators.devine import Devine
from downscale.utils.dependencies import root_mse
from downscale.visu.visualization import Visualization
from downscale.visu.visu_gaussian import VisualizationGaussian
from downscale.eval.synthetic_topographies import GaussianTopo
from downscale.utils.utils_func import save_figure

dependencies = {'root_mse': root_mse}


def load_model_fold(p, fold_nb, prm):
    path_fold = prm["model_path_fold"] + f"fold{fold_nb}/"
    p.load_cnn(model_path=path_fold, dependencies=True)
    return p


def load_std_training_fold(fold_nb, prm):
    dict_norm = pd.read_csv(prm["model_path_fold"] + "dict_norm.csv")
    std = dict_norm[str(fold_nb)].iloc[1]
    return std


def load_test_group_in_gaussian_topo_df_all(fold_nb, prm):
    path_fold = prm["model_path_fold"] + f"fold{fold_nb}/"
    train_test_group = pd.read_csv(path_fold+f"df_all_{fold_nb}.csv")
    list_variables = ['degree', 'xi', 'degree_xi', 'topo_name', 'wind_name', 'group']
    test_group = train_test_group[list_variables][train_test_group["group"] == "test"]
    return test_group


def load_test_data(gaussian_topo, test_group, prm):
    topos = gaussian_topo.filenames_to_array(test_group, prm["gaussian_topo_path"], 'topo_name')
    winds = gaussian_topo.filenames_to_array(test_group, prm["gaussian_topo_path"], 'wind_name').reshape(len(topos), 79, 69, 3)
    return topos, winds


def normalize_and_predict_topos(p, topos, std):
    mean_topos = np.mean(topos, axis=(1, 2)).reshape(len(topos), 1, 1)
    std = std.reshape(1, 1, 1)
    topo_norm = p.normalize_topo(topos, mean_topos, std)
    predictions = p.model.predict(topo_norm)
    return predictions


def store_test_data_in_dataframe(topos, winds, df_all):
    df_all["U_test"] = ""
    df_all["V_test"] = ""
    df_all["W_test"] = ""
    for index in range(len(df_all)):
        df_all["topo_name"].iloc[index] = topos[index, :, :]
        df_all["U_test"].iloc[index] = winds[index, :, :, 0]
        df_all["V_test"].iloc[index] = winds[index, :, :, 1]
        df_all["W_test"].iloc[index] = winds[index, :, :, 2]
    return df_all


def store_predict_data_in_dataframe(predictions, df_all):
    df_all["U_pred"] = ""
    df_all["V_pred"] = ""
    df_all["W_pred"] = ""
    for index in range(len(df_all)):
        df_all["U_pred"].iloc[index] = predictions[index, :, :, 0]
        df_all["V_pred"].iloc[index] = predictions[index, :, :, 1]
        df_all["W_pred"].iloc[index] = predictions[index, :, :, 2]
    return df_all


def compute_speed_and_direction_from_components(gaussian_topo, df_all):
    U_test = df_all["U_test"].values
    V_test = df_all["V_test"].values
    W_test = df_all["W_test"].values
    U_pred = df_all["U_pred"].values
    V_pred = df_all["V_pred"].values
    W_pred = df_all["W_pred"].values

    df_all["UV_test"] = [gaussian_topo.compute_wind_speed(U_test[index], V_test[index], verbose=False) for index in
                         range(len(U_test))]
    df_all["UVW_test"] = [gaussian_topo.compute_wind_speed(U_test[index], V_test[index], W_test[index], verbose=False)
                          for index in range(len(U_test))]
    df_all["alpha_test"] = [gaussian_topo.angular_deviation(U_test[index], V_test[index], verbose=False) for index in
                            range(len(U_test))]
    df_all["Wind_DIR_test"] =[gaussian_topo.direction_from_u_and_v(U_test[index], V_test[index], verbose=False) for
                              index in range(len(U_test))]
    df_all["UV_pred"] = [gaussian_topo.compute_wind_speed(U_pred[index], V_pred[index], verbose=False) for index in
                         range(len(U_pred))]
    df_all["UVW_pred"] = [gaussian_topo.compute_wind_speed(U_pred[index], V_pred[index], W_pred[index], verbose=False)
                          for index in range(len(U_pred))]
    df_all["alpha_pred"] = [gaussian_topo.angular_deviation(U_pred[index], V_pred[index], verbose=False) for index in
                            range(len(U_pred))]
    df_all["Wind_DIR_pred"] =[gaussian_topo.direction_from_u_and_v(U_pred[index], V_pred[index], verbose=False) for
                              index in range(len(U_test))]

    return df_all


def add_curvature_to_df_all(gaussian_topo, topos, df_all):
    df_all["curvature"] = [gaussian_topo.curvature_map(topos[index, :, :], verbose=False) for index in range(len(df_all))]
    df_all["curvature"] = df_all["curvature"].apply(border_array_to_nan(1))
    return df_all


def add_tpi_to_df_all(gaussian_topo, topos, df_all):
    df_all["tpi_500"] = [gaussian_topo.tpi_map(topos[index, :, :], 500, 30) for index in range(len(df_all))]
    df_all["tpi_500"] = df_all["tpi_500"].apply(border_array_to_nan(17))
    return df_all


def add_sx_to_df_all(gaussian_topo, topos, df_all):
    df_all["sx_300"] = [gaussian_topo.sx_map(topos[index, :, :], 30, 300, 270, 5, 30) for index in range(len(df_all))]
    df_all["sx_300"] = df_all["sx_300"].apply(border_array_to_nan(10))
    return df_all


def add_laplacian_to_df_all(gaussian_topo, topos, df_all):
    df_all["laplacian"] = [gaussian_topo.laplacian_map(topos[index, :, :], 30, verbose=False, helbig=False) for index in range(len(df_all))]
    df_all["laplacian"] = df_all["laplacian"].apply(border_array_to_nan(1))
    return df_all


def add_mu_to_df_all(gaussian_topo, topos, df_all):
    df_all["mu"] = [gaussian_topo.mu_helbig_map(topos[index, :, :], 30, verbose=False) for index in range(len(df_all))]
    df_all["mu"] = df_all["mu"].apply(border_array_to_nan(1))
    return df_all


def border_array_to_nan(border):
    def _border_array_to_nan(array):
        array[:border, :] = np.nan
        array[-border:, :] = np.nan
        array[:, :border] = np.nan
        array[:, -border:] = np.nan
        return array
    return _border_array_to_nan


def unnesting(df, explode):
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx
    return df1.join(df.drop(explode, 1), how='left')


def flatten_arrays(list_variables, df_all):
    for variable in list_variables:
        df_all[variable] = df_all[variable].apply(lambda x: np.array(x).flatten())
    test_i = unnesting(df_all, list_variables)
    return test_i


def compute_metrics(test_i, metric="bias"):
    metrics = Metrics()

    if metric=="bias":

        # Speed
        test_i["bias"] = metrics.bias(test_i['UVW_pred'].values, test_i['UVW_test'].values)
        #test_i["bias_rel"] = metrics.bias_rel(test_i['UVW_pred'].values, test_i['UVW_test'].values)
        #test_i["absolute_error_rel"] = metrics.absolute_error_relative(test_i['UVW_pred'].values, test_i['UVW_test'].values)

        # Direction
        test_i["bias_DIR"] = metrics.bias_direction(test_i['Wind_DIR_pred'].values, test_i['Wind_DIR_test'].values)

    elif metric == "abs_error":
        try:
            test_i["absolute_error"] = metrics.absolute_error(test_i['UVW_pred'].values, test_i['UVW_test'].values)
            test_i["absolute_error_DIR"] = metrics.abs_bias_direction(test_i['UVW_pred'].values, test_i['UVW_test'].values)
        except AttributeError:
            test_i["absolute_error"] = metrics.absolute_error(np.array(test_i['UVW_pred']), np.array(test_i['UVW_test']))
            test_i["absolute_error_DIR"] = metrics.abs_bias_direction(np.array(test_i['Wind_DIR_pred']), np.array(test_i['Wind_DIR_test']))

    return test_i


def convert_slope_from_Nora(df):
    old_slopes = [5, 10, 13, 16, 20]
    new_slopes = [10, 19, 25, 30, 36]
    df["degree_old"] = df["degree"]
    for old_slope, new_slope in zip(old_slopes, new_slopes):
        df["degree"][df["degree_old"] == old_slope] = new_slope
    return df


def figure_training_result_by_metrics(config, prm):

    p = Devine(prm=prm)
    gaussian_topo = GaussianTopo()

    test = []
    for fold_nb in range(10):

        print(fold_nb)

        print("Loading model and configuration")
        p = load_model_fold(p, fold_nb, prm)
        std = load_std_training_fold(fold_nb, prm)

        print("Selecting test data")
        test_group = load_test_group_in_gaussian_topo_df_all(fold_nb, prm)

        print("Loading test data")
        topos, winds = load_test_data(gaussian_topo, test_group, prm)

        print("Predict test data")
        predictions = normalize_and_predict_topos(p, topos, std)

        print("Store test data in a dataframe")
        df_all = test_group.copy(deep=True)
        df_all = store_test_data_in_dataframe(topos, winds, df_all)

        print("Store predict data in a dataframe")
        df_all = store_predict_data_in_dataframe(predictions, df_all)

        print("Compute speed and direction from components")
        df_all = compute_speed_and_direction_from_components(gaussian_topo, df_all)

        list_variables = ['topo_name', 'U_test', 'V_test',
                          'W_test', 'U_pred', 'V_pred', 'W_pred', 'Wind_DIR_pred',
                          'UV_test', 'UVW_test',
                          'alpha_test', 'UV_pred', 'UVW_pred', 'alpha_pred', 'Wind_DIR_test']
        list_variables_init = [variable for variable in list_variables]

        if config["tpi_500"] or not config["metric_already_loaded"]:
            print("Compute tpi_500")
            df_all = add_tpi_to_df_all(gaussian_topo, topos, df_all)
            list_variables.append('tpi_500')

        if config["sx_300"] or not config["metric_already_loaded"]:
            print("Compute sx_300")
            df_all = add_sx_to_df_all(gaussian_topo, topos, df_all)
            list_variables.append('sx_300')

        if config["curvature"] or not config["metric_already_loaded"]:
            print("Compute curvature")
            df_all = add_curvature_to_df_all(gaussian_topo, topos, df_all)
            list_variables.append('curvature')

        if config["laplacian"] or not config["metric_already_loaded"]:
            print("Compute laplacian")
            df_all = add_laplacian_to_df_all(gaussian_topo, topos, df_all)
            list_variables.append('laplacian')

        if config["mu"] or not config["metric_already_loaded"]:
            print("Compute mu")
            df_all = add_mu_to_df_all(gaussian_topo, topos, df_all)
            list_variables.append('mu')

        print("Flatten arrays")
        test_i = flatten_arrays(list_variables, df_all)

        print("Compute metrics")
        test_i = compute_metrics(test_i, metric="abs_error")
        test.append(test_i)

    test = pd.concat(test)
    print("debug0")
    print(test.describe())
    if config["return_df_results"]:
        return test

    if config["save_dataframe"]:
        save_path = prm["preprocessed_data"] + "Analysis/Gaussian_topo/"
        test.to_pickle(save_path + f"dataframe_on_gaussian_topo_with_metrics_{str(uuid.uuid4())[:4]}.pkl")
    if config["load_dataframe"]:
        save_path = prm["preprocessed_data"] + "Analysis/Gaussian_topo/"
        loaded = pd.read_pickle(save_path + config["name_dataframe_on_gaussian_topo_with_metrics"])
        try:
            list_variables_init.extend(["bias", "bias_DIR"])
        except KeyError:
            list_variables_init.extend(["absolute_error", "absolute_error_DIR"])
        variable_not_loaded = list_variables_init
        loaded[variable_not_loaded] = test[variable_not_loaded]
        del test
        test = loaded

    """
    # bias=f(degree)
    plt.figure(figsize=(10, 13))
    test = convert_slope_from_Nora(test)
    sns_palette = sns.light_palette("SteelBlue", n_colors=len(test["degree"].unique()))
    ax = sns.boxplot(data=test, x="degree", y="bias", palette=sns_palette, showfliers=False)
    plt.xlabel("Mean slope [degree]", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [m/s]", fontsize=config.get("fontsize"))
    plt.xticks(fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    save_figure("speed_degree", prm)

    # bias=f(xi)
    plt.figure(figsize=(10,13))
    sns_palette = sns.light_palette("IndianRed", n_colors=len(test["xi"].unique()))
    ax = sns.boxplot(data=test, x="xi", y="bias", palette=sns_palette, showfliers=False)
    plt.xticks(fontsize=config.get("fontsize")/2)
    plt.yticks(fontsize=config.get("fontsize"))
    plt.xlabel("Xi [m]", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [m/s]", fontsize=config.get("fontsize"))
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    save_figure("speed_xi", prm)

    # bias=f(tpi_500)
    plt.figure(figsize=(10,13))
    test = gaussian_topo.classify(test, variable="tpi_500", quantile=True)
    sns_palette = sns.light_palette("SeaGreen", n_colors=len(test["class_tpi_500"].unique()))
    ax = sns.boxplot(data=test, x="class_tpi_500", y="bias", palette=sns_palette, showfliers=False,
                     order=["$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$", "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"])
    plt.xlabel("TPI [m]", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [m/s]", fontsize=config.get("fontsize"))
    plt.xticks(rotation=45, ha="right", fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    save_figure("speed_tpi_500", prm)


    # bias=f(sx_300)
    plt.figure(figsize=(10,13))
    test = gaussian_topo.classify(test, variable="sx_300", quantile=True)
    sns_palette = sns.light_palette("DarkKhaki", n_colors=len(test["class_sx_300"].unique()))
    ax = sns.boxplot(data=test, x="class_sx_300", y="bias", palette=sns_palette, showfliers=False,
                     order=["$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$", "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"])
    plt.xlabel("Sx [m]", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [m/s]", fontsize=config.get("fontsize"))
    plt.xticks(rotation=45, ha="right", fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    save_figure("speed_sx_300", prm)


    # bias=f(curvature)
    plt.figure(figsize=(10,13))
    test = gaussian_topo.classify(test, variable="curvature", quantile=True)
    sns_palette = sns.light_palette("PaleVioletRed", n_colors=len(test["class_curvature"].unique()))
    ax = sns.boxplot(data=test, x="class_curvature", y="bias", palette=sns_palette, showfliers=False,
                     order=["$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$", "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"])
    plt.xlabel("curvature [m]", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [m/s]", fontsize=config.get("fontsize"))
    plt.xticks(rotation=45, ha="right", fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    save_figure("speed_curvature", prm)

    # bias=f(laplacian)
    plt.figure(figsize=(10,13))
    test = gaussian_topo.classify(test, variable="laplacian", quantile=True)
    sns_palette = sns.light_palette("DarkOrchid", n_colors=len(test["class_laplacian"].unique()))
    ax = sns.boxplot(data=test, x="class_laplacian", y="bias", palette=sns_palette, showfliers=False,
                     order=["$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$", "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"])
    plt.xlabel("Laplacian []", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [m/s]", fontsize=config.get("fontsize"))
    plt.xticks(rotation=45, ha="right", fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    save_figure("speed_laplacian", prm)


    # bias=f(mu)
    plt.figure(figsize=(10,13))
    test = gaussian_topo.classify(test, variable="mu", quantile=True)
    sns_palette = sns.light_palette("LightSlateGray", n_colors=len(test["class_mu"].unique()))
    ax = sns.boxplot(data=test, x="class_mu", y="bias", palette=sns_palette, showfliers=False,
                     order=["$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$", "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"])
    plt.xlabel("Mu []", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [m/s]", fontsize=config.get("fontsize"))
    plt.xticks(rotation=45, ha="right", fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    save_figure("speed_mu", prm)









    # bias=f(degree)
    plt.figure(figsize=(10, 13))
    sns_palette = sns.light_palette("SteelBlue", n_colors=len(test["degree"].unique()))
    ax = sns.boxplot(data=test, x="degree", y="bias_DIR", palette=sns_palette, showfliers=False)
    plt.xlabel("Mean slope [degree]", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [°]", fontsize=config.get("fontsize"))
    plt.xticks(fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-15, 15)
    plt.grid(True)
    plt.tight_layout()
    save_figure("direction_degree", prm)


    # bias=f(xi)
    plt.figure(figsize=(10,13))
    sns_palette = sns.light_palette("IndianRed", n_colors=len(test["xi"].unique()))
    ax = sns.boxplot(data=test, x="xi", y="bias_DIR", palette=sns_palette, showfliers=False)
    plt.xlabel("Xi [m]", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [°]", fontsize=config.get("fontsize"))
    plt.xticks(fontsize=config.get("fontsize")/2)
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-15, 15)
    plt.grid(True)
    plt.tight_layout()
    save_figure("direction_xi", prm)


    # bias=f(tpi_500)
    plt.figure(figsize=(10,13))
    sns_palette = sns.light_palette("SeaGreen", n_colors=len(test["class_tpi_500"].unique()))
    ax = sns.boxplot(data=test, x="class_tpi_500", y="bias_DIR", palette=sns_palette, showfliers=False,
                     order=["$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$", "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"])
    plt.xlabel("TPI [m]", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [°]", fontsize=config.get("fontsize"))
    plt.xticks(rotation=45, ha="right", fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-15, 15)
    plt.grid(True)
    plt.tight_layout()
    save_figure("direction_tpi500", prm)


    # bias=f(sx_300)
    plt.figure(figsize=(10,13))
    sns_palette = sns.light_palette("DarkKhaki", n_colors=len(test["class_sx_300"].unique()))
    ax = sns.boxplot(data=test, x="class_sx_300", y="bias_DIR", palette=sns_palette, showfliers=False,
                     order=["$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$", "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"])
    plt.xlabel("Sx [m]", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [°]", fontsize=config.get("fontsize"))
    plt.xticks(rotation=45, ha="right", fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-15, 15)
    plt.grid(True)
    plt.tight_layout()
    save_figure("direction_sx300", prm)


    # bias=f(curvature)
    plt.figure(figsize=(10,13))
    sns_palette = sns.light_palette("PaleVioletRed", n_colors=len(test["class_curvature"].unique()))
    ax = sns.boxplot(data=test, x="class_curvature", y="bias_DIR", palette=sns_palette, showfliers=False,
                     order=["$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$", "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"])
    plt.xlabel("curvature [m]", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [°]", fontsize=config.get("fontsize"))
    plt.xticks(rotation=45, ha="right", fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-15, 15)
    plt.grid(True)
    plt.tight_layout()
    save_figure("direction_curvature", prm)


    # bias=f(laplacian)
    plt.figure(figsize=(10,13))
    sns_palette = sns.light_palette("DarkOrchid", n_colors=len(test["class_laplacian"].unique()))
    ax = sns.boxplot(data=test, x="class_laplacian", y="bias_DIR", palette=sns_palette, showfliers=False,
                     order=["$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$", "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"])
    plt.xlabel("Laplacian []", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [°]", fontsize=config.get("fontsize"))
    plt.xticks(rotation=45, ha="right", fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-15, 15)
    plt.grid(True)
    plt.tight_layout()
    save_figure("direction_laplacian", prm)


    # bias=f(mu)
    plt.figure(figsize=(10,13))
    sns_palette = sns.light_palette("LightSlateGray", n_colors=len(test["class_mu"].unique()))
    ax = sns.boxplot(data=test, x="class_mu", y="bias_DIR", palette=sns_palette, showfliers=False,
                     order=["$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$", "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"])
    plt.xlabel("Mu []", fontsize=config.get("fontsize"))
    plt.ylabel("Bias [°]", fontsize=config.get("fontsize"))
    plt.xticks(rotation=45, ha="right", fontsize=config.get("fontsize"))
    plt.yticks(fontsize=config.get("fontsize"))
    plt.ylim(-15, 15)
    plt.grid(True)
    plt.tight_layout()
    save_figure("direction_mu", prm)
    """


def figure_epochs_bias(prm):
    import matplotlib
    # matplotlib.use('Agg')
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    all_test = []
    x_epoch = []
    y_epoch = []
    for fold in range(10):
        print(fold)
        test = np.load(prm["model_path_fold"] + f"fold{fold}/fold_{fold}_history.npy", allow_pickle=True)
        test = test.reshape(1)[0]
        test = pd.DataFrame(test)
        print(test)
        ax = plt.gca()
        test['mae'].plot(ax=ax, color="cadetblue", label='_nolegend_', alpha=0.25)
        test['val_mae'].plot(ax=ax, color="lightcoral", label='_nolegend_', alpha=0.25)
        all_test.append(test)
        x_epoch.append(test.index.max())
        y_epoch.append(0)
        # plt.plot(test.index.max(), 0, color='red', marker='x')

    all_test = pd.concat(all_test)
    all_max = all_test.groupby(all_test.index).max()
    all_min = all_test.groupby(all_test.index).min()
    all_mean = all_test.groupby(all_test.index).mean()

    ax = plt.gca()
    all_min['mae'].plot(ax=ax, color="cadetblue", linestyle='--', label='_nolegend_', linewidth=1)
    all_min['val_mae'].plot(ax=ax, color="lightcoral", linestyle='--', label='_nolegend_', linewidth=1)
    all_max['mae'].plot(ax=ax, color="cadetblue", linestyle='--', label='training min/max', linewidth=1)
    all_max['val_mae'].plot(ax=ax, color="lightcoral", linestyle='--', label='validation min/max', linewidth=1)
    all_mean['mae'].plot(ax=ax, color="cadetblue", linestyle='-', linewidth=2, label='training mean')
    all_mean['val_mae'].plot(ax=ax, color="lightcoral", linestyle='-', linewidth=2, label='validation mean')
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    # ax2 = ax.twinx()
    # ax2.hist(x_epoch, color='red', zorder=1000000)
    # ax2.set_ylim(0,20)

    # handles, _ = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), labels)

    # ax = plt.gca()
    # legends
    ax.legend()
    plt.yscale('log')
    plt.xlabel("Number of epochs")
    plt.ylabel("Mean absolute error [m/s]")
    plt.grid()

    plt.show()


def figure_distribution_gaussian(prm):
    p = Devine(prm=prm)
    visu = VisualizationGaussian(p)
    gaussian_topo = GaussianTopo()

    # Distributions by degree or xi
    type_plots = ["wind speed", "wind speed", "angular deviation"]
    type_of_winds = ["uv", "uvw", None]
    for topo_carac in ["degree", "xi"]:
        for type_plot, type_of_wind in zip(type_plots, type_of_winds):
            dict_gaussian_deg_or_xi = gaussian_topo.load_data_by_degree_or_xi(prm, degree_or_xi=topo_carac)
            print("Mean values:")
            print(dict_gaussian_deg_or_xi.groupby(topo_carac).mean())
            visu.plot_gaussian_distrib_by_degree_or_xi(dict_gaussian_deg_or_xi,
                                                       degree_or_xi=topo_carac,
                                                       type_plot=type_plot,
                                                       type_of_wind=type_of_wind,
                                                       fill=False, fontsize=20)
            plt.savefig(
                prm["save_figure_path"] + f"distribution_{topo_carac}_{type_plot}_{type_of_wind}_{str(uuid.uuid4())[:4]}.png")


def figure_heatmap_training(prm):

    p = Devine(prm=prm)
    gaussian_topo = GaussianTopo()

    test = []
    for fold_nb in range(10):

        print(fold_nb)

        print("Loading model and configuration")
        p = load_model_fold(p, fold_nb, prm)
        std = load_std_training_fold(fold_nb, prm)

        print("Selecting test data")
        test_group = load_test_group_in_gaussian_topo_df_all(fold_nb, prm)

        print("Loading test data")
        topos, winds = load_test_data(gaussian_topo, test_group, prm)

        print("Predict test data")
        predictions = normalize_and_predict_topos(p, topos, std)

        print("Store test data in a dataframe")
        df_all = test_group.copy(deep=True)
        df_all = store_test_data_in_dataframe(topos, winds, df_all)

        print("Store predict data in a dataframe")
        df_all = store_predict_data_in_dataframe(predictions, df_all)

        print("Compute speed and direction from components")
        df_all = compute_speed_and_direction_from_components(gaussian_topo, df_all)

        list_variables = ['topo_name', 'U_test', 'V_test',
                          'W_test', 'U_pred', 'V_pred', 'W_pred', 'Wind_DIR_pred',
                          'UV_test', 'UVW_test',
                          'alpha_test', 'UV_pred', 'UVW_pred', 'alpha_pred', 'Wind_DIR_test']
        list_variables_init = [variable for variable in list_variables]

        print("Compute metrics")
        test_i = pd.concat([compute_metrics(row, metric="abs_error") for index, row in df_all.iterrows()])
        test.append(test_i)

    test = pd.concat(test)

    # Speed
    plt.figure()
    sns.heatmap(test["absolute_error"].mean(), cmap=sns.color_palette("mako", as_cmap=True))
    ax = plt.gca()
    fig = ax.get_figure()
    save_path = prm["save_figure_path"]
    fig.savefig(save_path + f"heatmap_speed_training_{str(uuid.uuid4())[:4]}.png")

    # Direction
    plt.figure()
    sns.heatmap(test["absolute_error_DIR"].mean(), cmap=sns.color_palette("mako", as_cmap=True))
    ax = plt.gca()
    fig = ax.get_figure()
    save_path = prm["save_figure_path"]
    fig.savefig(save_path + f"heatmap_direction_training_{str(uuid.uuid4())[:4]}.png")



