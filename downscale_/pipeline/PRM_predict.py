from utils_prm import update_selected_path, select_path_to_coord_L93,\
    add_additional_stations, add_list_stations, check_expose_elevation, \
    check_extract_around_station_or_interpolated, create_begin_and_end_str, check_save_and_load, append_module_path


def create_prm(month_prediction=True):

    prm = {}

    """
    CPU or GPU
    """

    # GPU
    prm["GPU"] = True
    prm["horovod"] = True
    prm["results_name"] = "test_interpolation_final_is_false"

    """
    MNT, NWP and observations
    """

    # Observations
    prm["add_additional_stations"] = True
    prm["select_date_time_serie"] = True
    prm["fast_loading"] = True

    # MNT
    prm["name_mnt"] = "cen_gr"
    prm["resolution_mnt_x"] = 30
    prm["resolution_mnt_y"] = 30

    # NWP
    prm["name_nwp"] = "AROME"

    """
    Predictions
    """

    # General
    prm["launch_predictions"] = True
    prm["verbose"] = True

    # Z0
    prm["Z0"] = True
    prm["load_z0"] = True
    prm["save_z0"] = False
    prm["log_profile_3m_to_10m"] = False

    # Exposed wind
    prm["expose"] = "max altitude" #"peak valley", "max altitude", "10m", None
    prm["peak_valley"] = False
    prm["scale_at_10m"] = False # if peak_valley = True, scale_at_10m = False
    prm["scale_at_max_altitude"] = True # if peak_valley = True, scale_at_10m = False

    # Scaling
    prm["scaling_function"] = "Arctan_30_2" # linear, Arctan_30_1, Arctan_30_2, Arctan_10_2, Arctan_20_2, Arctan_40_2, Arctan_50_2

    # Get closer from learning conditions
    prm["get_closer_learning_condition"] = False

    # Profiling
    prm["line_profile"] = False
    prm["memory_profile"] = False

    # Ideal cases
    prm["ideal_case"] = False
    prm["input_speed"] = 3
    prm["input_direction"] = 270

    # For predictions at stations
    prm["stations_to_predict"] = "all"

    # For predictions long periods at stations
    prm["variable"] = ["W", "UV", "UVW", "UV_DIR_deg", "alpha_deg", "NWP_wind_speed", "exp_Wind", "acceleration_CNN", "Z0", "a4"]
    prm["station_similar_to_map"] = True  # If you work on interpolated AROME True, if stations False

    # For map prediction
    prm["type_rotation"] = "tfa" # "indexes" or "scipy" or "tfa"
    prm["interpolate_nwp"] = True
    prm["interp"] = 2
    prm["nb_pixels"] = 15#24 # Number of pixel extracted around each NWP pixel after prediction, default 15
    prm["interpolate_final_map"] = False
    prm["dx"] = 2_000
    prm["dy"] = 2_500
    prm["centered_on_interpolated"] = True  # If you work at stations, False, if interpolated AROME True
    prm["select_area"] = "coord"
    prm["coords_domain_for_map_prediction"] = [937875, 6439125, 973375, 6464125] #[959000, 6462000, 960000, 6464000]  #[xmin, ymin, xmax, ymax]
    prm["method"] = "linear"
    prm["nb_batch_sent_to_gpu"] = 10 # Number of batches sent from the cpu to the gpu to avoid out of memory error
    prm["batch_size_prediction"] = 2**10 # Number of batches in the gpu to accelerate computation

    # 2 August 2017 1h
    prm["hour_begin"] = 0#1
    prm["day_begin"] = 1#2
    prm["month_begin"] = 1#8
    prm["year_begin"] = 2019#2017

    # 31 May 2020 1h
    prm["hour_end"] = 0#1
    prm["day_end"] = 10#31
    prm["month_end"] = 1#5
    prm["year_end"] = 2019#2020

    prm["list_no_HTN"] = ["DIGNE LES BAINS", "LA MURE-ARGENS", "ARVIEUX", "EMBRUN", "LA FAURIE", "GAP", "ANTIBES-GAROUPE", "ASCROS", "CANNES", "CAUSSOLS", "PEILLE",
                          "CHAPELLE-EN-VER", "TRICASTIN", "ST ROMAN-DIOIS", "CREYS-MALVILLE", "ST-ALBAN", "ST-PIERRE-LES EGAUX", "LUS L CROIX HTE", "GRENOBLE–LVD", "ALBERTVILLE JO",
                          "MERIBEL BURGIN", "MONT DU CHAT", "FECLAZ_SAPC", "FREJUS", "LA MASSE", "RISTOLAS", "TALLARD", "ST MICHEL MAUR_SAPC", "TIGNES_SAPC", "VAL D’I SOLAISE",
                          "AGUIL. DU MIDI", "Vallot", "Saint-Sorlin", "Argentiere", "Dome Lac Blanc"]

    prm["list_additionnal"] = ['Vallot', 'Saint-Sorlin', 'Argentiere', 'Dome Lac Blanc', 'Col du Lac Blanc',
                               'La Muzelle Lac Blanc', 'Col de Porte']

    # Paths to files
    if not prm["GPU"]:

        # Parent directory
        working_directory = "C:/Users/louis/git/wind_downscaling_CNN/"
        # Data
        prm["data_path"] = working_directory + "Data/1_Raw/"
        # Synthetic topographies
        prm["gaussian_topo_path"] = prm["data_path"] + "ARPS/"
        # CNN
        prm["experience_path"] = working_directory + "Models/ARPS/"
        # Topography
        #prm["topo_path"] = prm["data_path"] + "MNT/IGN_25m/ign_L93_25m_alpesIG.tif"
        prm["topo_path"] = prm["data_path"] + "MNT/CEN/DEM_FRANCE_L93_30m_bilinear.tif"
        #prm["topo_path"] = prm["data_path"] + "MNT/CEN/grandes_rousses.tif"
        # Observations
        prm["BDclim_stations_path"] = prm["data_path"] + "BDclim/precise_localisation/liste_postes_alps_l93.csv"
        prm["height_sensor_path"] = prm["data_path"] + "BDclim/height_sensors.pkl"
        #prm["BDclim_stations_path"] = prm["data_path"] + "BDclim/04_Mai_2021/liste_postes_Alpes_LLT.csv"

        # 2017-2019
        #prm["BDclim_data_path"] = prm["data_path"] + "BDclim/extract_BDClim_et_sta_alp_20171101_20190501.csv"
        # 2015-2021
        #prm["BDclim_data_path"] = prm["data_path"] + "BDclim/extract_FF_T_RR1_alp_2015010100_2021013100.csv"
        # 2009-2021
        prm["BDclim_data_path"] = prm["data_path"] + "BDclim/04_Mai_2021/extract_FF_T_RR1_alp_2009010100_2021013100.csv"

        # NWP
        prm["AROME_path_1"] = prm["data_path"] + "AROME/FORCING_alp_2017080106_2018080106_32bits.nc"
        prm["AROME_path_2"] = prm["data_path"] + "AROME/FORCING_alp_2018080106_2019060106_32bits.nc"
        prm["AROME_path_3"] = prm["data_path"] + "AROME/FORCING_alp_2019060107_2019070106_32bits.nc"
        prm["AROME_path_4"] = prm["data_path"] + "AROME/FORCING_alp_2019060106_2020060206_32bits.nc"
        prm["AROME_path"] = [prm["AROME_path_1"], prm["AROME_path_2"], prm["AROME_path_3"], prm["AROME_path_4"]]

        prm["path_to_synthetic_topo"] = working_directory + "Data/2_Pre_processed/ARPS/fold/df_all.csv"

        # Path to python module downscale
        prm["path_module"] = working_directory + "src/downscale_/"

    if prm["GPU"]:
        # Data
        prm["data_path"] = "//scratch/mrmn/letoumelinl/predict_real/"
        # CNN
        prm["experience_path"] = "//scratch/mrmn/letoumelinl/predict_real/Model/"
        # Topography
        #prm["topo_path"] = prm["data_path"] + "MNT/IGN_25m/preprocessed_MNT.nc"
        prm["topo_path"] = prm["data_path"] + "MNT/CEN/grandes_rousses.nc"

        # Observations
        prm["BDclim_stations_path"] = prm["data_path"] + "BDclim/17_09_2021/stations.csv"
        # 2009-2021
        prm["BDclim_data_path"] = prm["data_path"] + "BDclim/17_09_2021/time_series.csv"

        # NWP
        #prm["AROME_path_1"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2017080106_2018080106_32bits.nc"
        #prm["AROME_path_2"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2018080106_2019050106_32bits.nc"
        #prm["AROME_path_3"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2019060107_2019070106_32bits.nc"
        #prm["AROME_path_4"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2019060106_2020060206_32bits.nc"

        prm["AROME_path_1"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2017080106_2018080106_32bits.nc"
        prm["AROME_path_2"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2018080106_2019050106_32bits.nc"
        prm["AROME_path_3"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2019050106_2019060106_32bits.nc"
        #prm["AROME_path_3"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2019060107_2019070106_32bits.nc"
        prm["AROME_path_4"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2019060106_2020060206_32bits.nc"

        prm["AROME_path"] = [prm["AROME_path_1"], prm["AROME_path_2"], prm["AROME_path_3"], prm["AROME_path_4"]]

        # Path to python module downscale
        prm["path_module"] = "//home/mrmn/letoumelinl/downscale_"

    # Z0
    prm["path_Z0_2018"] = prm["data_path"] + "AROME/Z0/Z0_alp_2018010100_2018120700.nc" if prm["Z0"] else None
    prm["path_Z0_2019"] = prm["data_path"] + "AROME/Z0/Z0_alp_20190103_20191227.nc" if prm["Z0"] else None
    prm["save_path"] = prm["data_path"] + "AROME/Z0/" if prm["Z0"] else None

    # SURFEX
    prm["path_SURFEX"] = prm["data_path"] + "/SURFEX/scriptMNT.nc"  # Surfex grid on the Grandes Rousses domain (from Ange)
    prm["path_save_prediction_on_surfex_grid"] = prm["data_path"] + "/SURFEX/"  # Where to save predictions on SURFECX grid

    # QC
    prm["QC_pkl"] = prm["data_path"] + "BDclim/QC/qc_59.pkl" # Quality controled time series (where to save the file)
    prm["path_fast_loading"] = prm["data_path"] + "BDclim/QC/qc_59.pkl" # Quality controled time series (where to load the file)

    # Observation
    prm["path_vallot"] = prm["data_path"] + "BDclim/Vallot/"
    prm["path_saint_sorlin"] = prm["data_path"] + "BDclim/Saint-Sorlin/"
    prm["path_argentiere"] = prm["data_path"] + "BDclim/Argentiere/"
    prm["path_Dome_Lac_Blanc"] = prm["data_path"] + "BDclim/Col du Lac Blanc/Dome/treated/dome_v2.csv"
    prm["path_Col_du_Lac_Blanc"] = prm["data_path"] + "BDclim/Col du Lac Blanc/Col/treated/col_v2.csv"
    prm["path_Muzelle_Lac_Blanc"] = prm["data_path"] + "BDclim/Col du Lac Blanc/Muzelle/treated/muzelle_v3.csv"
    prm["path_Col_de_Porte"] = prm["data_path"] + "BDclim/Col de Porte/treated/cdp.csv"
    prm["path_Col_du_Lautaret"] = prm["data_path"] + "BDclim/Col du Lautaret/lautaret.csv"

    # CNN model "date_16_02_name_simu_FINAL_1_0_model_UNet/"
    prm['model_experience'] = "date_20_05_name_simu_retrain_with_mean_each_sample_0_model_UNet/"
    prm["model_path"] = prm["experience_path"] + prm['model_experience']

    # Module path

    prm["name_prediction"] = prm["results_name"]

    # Do not modify
    append_module_path(prm)
    prm = create_begin_and_end_str(prm)
    prm = check_save_and_load(prm)
    prm = update_selected_path(prm, month_prediction)
    prm = select_path_to_coord_L93(prm)
    prm = add_additional_stations(prm)
    prm = add_list_stations(prm)
    prm = check_expose_elevation(prm)
    prm = check_extract_around_station_or_interpolated(prm)

    return prm



"""
# Détection des biais
from sklearn.neighbors import KernelDensity
from collections import defaultdict
import matplotlib.pyplot as plt

time_series = BDclim.time_series

def rmse(array_1, array_2):
    return(np.sqrt(np.mean((array_1-array_2)**2)))

wind_direction='winddir(deg)'
list_shift_direction = defaultdict(lambda: defaultdict(dict))
for station in ["LA MEIJE-NIVOSE"]:
    
    print(station)
    plt.figure()
    plt.title(station)
    # Select station
    time_serie_station = time_series[time_series["name"] == station]
    
    # Select wind
    wind = time_serie_station[wind_direction]
    
    # Select positive directions
    wind = wind[wind > 0]
    
    # Change range directions
    wind[wind == 360] = 0
    
    # Dropna
    wind = wind.dropna()
    
    # Constants
    nb_hour_in_year = 365*24
    
    
    list_years = wind.index.year.unique()
    
    year_plotted = []
    for index in range(len(list_years)-1):
        shift_detected = False
        print(f"Comparing {list_years[index]} and {list_years[index+1]}")
        # Select year and next year
        year_i = wind[(wind.index.year == list_years[index])].values
        year_after = wind[(wind.index.year == list_years[index + 1])].values
    
        # Filter on number of observations
        enough_data_year = len(year_i) >= 0.75 * nb_hour_in_year
        enough_data_year_after = len(year_after) >= 0.75 * nb_hour_in_year
    
        if enough_data_year & enough_data_year_after:
            
            print(f"Enough data in {list_years[index]} and {list_years[index+1]}")
            
            # Create density models
            model_year_i = KernelDensity(bandwidth=25, kernel='gaussian')
            
            # Fit density models
            model_year_i.fit(year_i.reshape(len(year_i), 1))
            
            # Evaluate distribution on range(0, 360)
            values = np.asarray([value for value in range(360)])
            values = values.reshape((len(values), 1))
            proba_year_i = model_year_i.score_samples(values)
            proba_year_i = np.exp(proba_year_i)
            
            def proba_year_after(year_after, values, shift):
                model_year_after = KernelDensity(bandwidth=25, kernel='gaussian')
                year_after = (year_after - shift)%360
                year_after = year_after.reshape(len(year_after), 1)
                model_year_after.fit(year_after)

                proba_year_after = model_year_after.score_samples(values)
                proba_year_after = np.exp(proba_year_after)
                return(proba_year_after)
            
            arg = np.argmin([rmse(proba_year_i, proba_year_after(year_after, values, shift)) for shift in range(0, 360, 20)])
            list_shift_direction[station][f"{list_years[index]} to {list_years[index+1]}"] = list(range(0, 360, 20))[arg]
            if list_shift_direction[station][f"{list_years[index]} to {list_years[index+1]}"] != 0:
                shift_detected = True
            print(shift_detected)
            linewidth = 3 if shift_detected else 1
            print(year_plotted)
            if ((list_years[index] not in year_plotted) and (list_years[index+1] not in year_plotted)) or shift_detected:
                print("Went here")
                plt.plot(values[:], proba_year_i, label=list_years[index], linewidth = linewidth)
                plt.plot(values[:], proba_year_after(year_after, values, 0), label=list_years[index+1], linewidth = linewidth)
                year_plotted.append(list_years[index])
                year_plotted.append(list_years[index+1])
                
        
    plt.legend()


list_shift_direction
"""
