import numpy as np
import datetime

from Utils import check_save_and_load


def create_prm(GPU=None, Z0=None, end=None, month_prediction=True):

    prm = {}

    """
    Simulation parameters
    """

    prm["GPU"] = True
    prm["horovod"] = True
    prm["Z0"] = True
    prm["load_z0"] = True
    prm["save_z0"] = False
    prm["peak_valley"] = True
    prm["launch_predictions"] = True
    prm["select_date_time_serie"] = True
    prm["verbose"] = True
    prm["line_profile"] = False
    prm["memory_profile"] = False
    prm["add_additionnal_stations"] = False

    # For predictions at stations
    prm["stations_to_predict"] = ['Col du Lac Blanc']

    # For predictions long periods
    prm["variable"] = "UV"

    # For map prediction
    prm["type_rotation"] = 'scipy'  # 'indexes' or 'scipy'
    prm["interp"] = 2
    prm["nb_pixels"] = 15
    prm["interpolate_final_map"] = True
    prm["dx"] = 20_000
    prm["dy"] = 25_000

    prm["hour_begin"] = 18
    prm["day_begin"] = 1
    prm["month_begin"] = 10
    prm["year_begin"] = 2018

    prm["hour_end"] = 18
    prm["day_end"] = 11
    prm["month_end"] = 10
    prm["year_end"] = 2018

    prm["begin"] = str(prm["year_begin"]) + "-" + str(prm["month_begin"]) + "-" + str(prm["day_begin"])
    prm["begin_after"] = str(prm["year_begin"]) + "-" + str(prm["month_begin"]) + "-" + str(prm["day_begin"] + 1)
    prm["end"] = str(prm["year_end"]) + "-" + str(prm["month_end"]) + "-" + str(prm["day_end"])


    # Please modify te paths
    if not(prm["GPU"]):
        # Parent directory
        working_directory = "C:/Users/louis/git/wind_downscaling_CNN/"
        # Data
        prm["data_path"] = working_directory + "Data/1_Raw/"
        # CNN
        prm["experience_path"] = working_directory + "Models/ARPS/"
        # Topography
        prm["topo_path"] = prm["data_path"] + "MNT/IGN_25m/ign_L93_25m_alpesIG.tif"
        # Observations
        prm["BDclim_stations_path"] = prm["data_path"] + "BDclim/precise_localisation/liste_postes_alps_l93.csv"
        #prm["BDclim_stations_path"] = prm["data_path"] + "BDclim/04_Mai_2021/liste_postes_Alpes_LLT.csv"

        # 2017-2019
        prm["BDclim_data_path"] = prm["data_path"] + "BDclim/extract_BDClim_et_sta_alp_20171101_20190501.csv"
        # 2015-2021
        #prm["BDclim_data_path"] = prm["data_path"] + "BDclim/extract_FF_T_RR1_alp_2015010100_2021013100.csv"
        # 2009-2021
        #prm["BDclim_data_path"] = prm["data_path"] + "BDclim/04_Mai_2021/extract_FF_T_RR1_alp_2009010100_2021013100.csv"

        # NWP
        prm["AROME_path_1"] = prm["data_path"] + "AROME/FORCING_alp_2017080106_2018080106_32bits.nc"
        prm["AROME_path_2"] = prm["data_path"] + "AROME/FORCING_alp_2018080106_2019060106_32bits.nc"
        prm["AROME_path_3"] = prm["data_path"] + "AROME/FORCING_alp_2019060107_2019070106_32bits.nc"
        prm["AROME_path_4"] = prm["data_path"] + "AROME/FORCING_alp_2019060106_2020060206_32bits.nc"
        prm["AROME_path"] = [prm["AROME_path_1"], prm["AROME_path_2"], prm["AROME_path_3"], prm["AROME_path_4"]]


    if prm["GPU"]:
        # Data
        prm["data_path"] = "//scratch/mrmn/letoumelinl/predict_real/"
        # CNN
        prm["experience_path"] = "//scratch/mrmn/letoumelinl/predict_real/Model/"
        # Topography
        prm["topo_path"] = prm["data_path"] + "MNT/IGN_25m/preprocessed_MNT.nc"
        # Observations
        prm["BDclim_stations_path"] = prm["data_path"] + "BDclim/17_05_2021/extract_FF_T_RR1_alp_2009010100_2021013100_stations.csv"
        # 2009-2021
        prm["BDclim_data_path"] = prm["data_path"] + "BDclim/17_05_2021/extract_FF_T_RR1_alp_2009010100_2021013100.csv"

        # NWP
        prm["AROME_path_1"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2017080106_2018080106_32bits.nc"
        prm["AROME_path_2"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2018080106_2019050106_32bits.nc"
        prm["AROME_path_3"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2019060107_2019070106_32bits.nc"
        prm["AROME_path_4"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2019060106_2020060206_32bits.nc"
        prm["AROME_path"] = [prm["AROME_path_1"], prm["AROME_path_2"], prm["AROME_path_3"], prm["AROME_path_4"]]

    # Z0
    prm["path_Z0_2018"] = prm["data_path"] + "AROME/Z0/Z0_alp_2018010100_2018120700.nc" if prm["Z0"] else None
    prm["path_Z0_2019"] = prm["data_path"] + "AROME/Z0/Z0_alp_20190103_20191227.nc" if prm["Z0"] else None
    prm["save_path"] = prm["data_path"] + "AROME/Z0/" if prm["Z0"] else None

    # Observation
    prm["path_vallot"] = prm["data_path"] + "BDclim/Vallot/"
    prm["path_saint_sorlin"] = prm["data_path"] + "BDclim/Saint-Sorlin/"
    prm["path_argentiere"] = prm["data_path"] + "BDclim/Argentiere/"
    prm["path_Dome_Lac_Blanc"] = prm["data_path"] + "BDclim/Col du Lac Blanc/Dome/treated/dome.csv"
    prm["path_Col_du_Lac_Blanc"] = prm["data_path"] + "BDclim/Col du Lac Blanc/Col/treated/col.csv"
    prm["path_Muzelle_Lac_Blanc"] = prm["data_path"] + "BDclim/Col du Lac Blanc/Muzelle/treated/muzelle.csv"
    prm["path_Col_de_Porte"] = prm["data_path"] + "BDclim/Col de Porte/treated/cdp.csv"
    prm["path_Col_du_Lautaret"] = prm["data_path"] + "BDclim/Col du Lautaret/lautaret.csv"

    # CNN model
    prm['model_experience'] = "date_16_02_name_simu_FINAL_1_0_model_UNet/"
    prm["model_path"] = prm["experience_path"] + prm['model_experience']

    # Safety
    prm["load_z0"], prm["save_z0"] = check_save_and_load(prm["load_z0"], prm["save_z0"])

    # Do not modify
    update_selected_path(prm, month_prediction)

    # Do not modify: L93_X and L93_Y
    prm["path_to_file_npy"] = select_path_to_file_npy(prm, GPU=prm["GPU"])

    # Do not modify: add_additionnal_stations
    prm = add_additionnal_stations(prm)

    return(prm)


def update_selected_path(prm, month_prediction):
    if month_prediction:
        prm = _update_selected_path(prm["year_end"], prm["month_end"], prm["day_end"], prm)
    else:
        prm["selected_path"] = prm["AROME_path"]
    return(prm)


def _update_selected_path(year, month, day, prm):

    current_date = datetime.datetime(year, month, day)
    d1 = datetime.datetime(2017, 8, 1, 6)
    d2 = datetime.datetime(2018, 8, 1, 6)
    d3 = datetime.datetime(2019, 6, 1, 6)
    d4 = datetime.datetime(2019, 6, 1, 7)
    d5 = datetime.datetime(2019, 7, 1, 6)
    d6 = datetime.datetime(2020, 7, 1, 6)

    if d1 < current_date <= d2:
        prm["selected_path"] = prm["AROME_path_1"]
    elif d2 < current_date <= d3:
        prm["selected_path"] = prm["AROME_path_2"]
    elif d4 <= current_date <= d5:
        prm["selected_path"] = prm["AROME_path_3"]
    elif d5 < current_date <= d6:
        prm["selected_path"] = prm["AROME_path_4"]
    else:
        prm["selected_path"] = prm["AROME_path_4"]

    return(prm)


def select_path_to_file_npy(prm, GPU=False):
    if GPU:
        prm_path = prm["selected_path"]
        path = "/".join(prm_path.split('/')[:-1]) + "/L93_npy/" + prm_path.split('/')[-1].split('.csv')[0].split('.nc')[0]
        return(path)
    else:
        return(None)


def add_additionnal_stations(prm):

    if not(prm["add_additionnal_stations"]):

        prm["path_vallot"] = None
        prm["path_saint_sorlin"] = None
        prm["path_argentiere"] = None
        prm["path_Dome_Lac_Blanc"] = None
        prm["path_Col_du_Lac_Blanc"] = None
        prm["path_Muzelle_Lac_Blanc"] = None
        prm["path_Col_de_Porte"] = None
        prm["path_Col_du_Lautaret"] = None

    return(prm)