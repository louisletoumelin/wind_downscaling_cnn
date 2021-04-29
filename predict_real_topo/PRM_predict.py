import numpy as np
import datetime

def create_prm(GPU, end=None, month_prediction=True):

    prm = {}

    if not(GPU):
        # Parent directory
        working_directory = "C:/Users/louis/git/wind_downscaling_CNN/"
        # Data
        prm["data_path"] = working_directory + "Data/1_Raw/"
        # CNN
        prm["experience_path"] = working_directory + "Models/ARPS/"
        # Topography
        prm["topo_path"] = prm["data_path"] + "MNT/IGN_25m/ign_L93_25m_alpesIG.tif"
        # NWP
        prm["path_to_file_npy"] = None
        # Observations
        prm["BDclim_stations_path"] = prm["data_path"] + "BDclim/precise_localisation/liste_postes_alps_l93.csv"

    if GPU:
        # Data
        prm["data_path"] = "//scratch/mrmn/letoumelinl/predict_real/"
        # CNN
        prm["experience_path"] = "//scratch/mrmn/letoumelinl/predict_real/Model/"
        # Topography
        prm["topo_path"] = prm["data_path"] + "MNT/IGN_25m/preprocessed_MNT.nc"
        # NWP
        prm["path_to_file_npy"] = prm["data_path"] + "AROME/AROME_June_2019"
        # Observations
        prm["BDclim_stations_path"] = prm["data_path"] + "BDclim/pickle_stations.pkl"

    # NWP
    prm["AROME_path_1"] = prm["data_path"] + "AROME/FORCING_alp_2017080106_2018080106.nc"
    prm["AROME_path_2"] = prm["data_path"] + "AROME/FORCING_alp_2018080106_2019060106.nc"
    prm["AROME_path_3"] = prm["data_path"] + "AROME/FORCING_alp_2019060107_2019070106.nc"
    prm["AROME_path"] = [prm["AROME_path_1"], prm["AROME_path_2"], prm["AROME_path_3"]]

    if month_prediction:
        year, month, day = np.int16(end.split('-'))
        prm = update_selected_path(year, month, day, prm)

    # Z0
    prm["path_Z0_2018"] = prm["data_path"] + "AROME/Z0/Z0_alp_2018010100_2018120700.nc"
    prm["path_Z0_2019"] = prm["data_path"] + "AROME/Z0/Z0_alp_20190103_20191227.nc"
    prm["save_path"] = prm["data_path"] + "AROME/Z0/"

    # Observation
    prm["BDclim_data_path"] = prm["data_path"] + "BDclim/extract_BDClim_et_sta_alp_20171101_20190501.csv"
    prm["path_vallot"] = prm["data_path"] + "BDclim/Vallot/"
    prm["path_saint_sorlin"] = prm["data_path"] + "BDclim/Saint-Sorlin/"
    prm["path_argentiere"] = prm["data_path"] + "BDclim/Argentiere/"

    # CNN model
    prm['model_experience'] = "date_16_02_name_simu_FINAL_1_0_model_UNet/"
    prm["model_path"] = prm["experience_path"] + prm['model_experience']

    return(prm)

def update_selected_path(year, month, day, prm):
    current_date = datetime.datetime(year, month, day)
    d1 = datetime.datetime(2017, 8, 1, 6)
    d2 = datetime.datetime(2018, 8, 1, 6)
    d3 = datetime.datetime(2019, 6, 1, 6)
    d4 = datetime.datetime(2019, 6, 1, 7)
    d5 = datetime.datetime(2019, 7, 1, 6)

    if d1 < current_date < d2:
        prm["selected_path"] = prm["AROME_path_1"]
    if d2 < current_date < d3:
        prm["selected_path"] = prm["AROME_path_2"]
    if d4 < current_date < d5:
        prm["selected_path"] = prm["AROME_path_3"]
    return(prm)