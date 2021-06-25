import datetime

from downscale.Utils.Utils import check_save_and_load
from downscale.Utils.prm import update_selected_path, select_path_to_file_npy, add_additionnal_stations


def create_prm(month_prediction=True):

    prm = {}

    """
    Simulation parameters
    """

    prm["GPU"] = False
    prm["horovod"] = False
    prm["Z0"] = True
    prm["load_z0"] = True
    prm["save_z0"] = False
    prm["peak_valley"] = True
    prm["select_date_time_serie"] = True
    prm["verbose"] = True
    prm["line_profile"] = False
    prm["memory_profile"] = False
    prm["add_additionnal_stations"] = False
    prm["launch_predictions"] = True

    # For predictions at stations
    prm["stations_to_predict"] = ["Col du Lac Blanc"]
    prm["ideal_case"] = False

    # For predictions long periods
    prm["variable"] = "UV"

    # For map prediction
    prm["type_rotation"] = 'scipy'  # 'indexes' or 'scipy'
    prm["interp"] = 2
    prm["nb_pixels"] = 15
    prm["interpolate_final_map"] = True
    prm["dx"] = 2_000
    prm["dy"] = 2_500
    prm["extract_stations_only"] = True

    prm["hour_begin"] = 1
    prm["day_begin"] = 10
    prm["month_begin"] = 6
    prm["year_begin"] = 2019

    prm["hour_end"] = 1
    prm["day_end"] = 10
    prm["month_end"] = 6
    prm["year_end"] = 2019

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
        prm["BDclim_stations_path"] = prm["data_path"] + "BDclim/02_06_2021/stations.csv"
        # 2009-2021
        prm["BDclim_data_path"] = prm["data_path"] + "BDclim/02_06_2021/time_series.csv"

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

    # QC
    prm["QC_pkl"] = prm["data_path"] + "BDclim/QC/qc_57.pkl"

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

    prm["list_mf"] = ['BARCELONNETTE', 'DIGNE LES BAINS', 'LA MURE-ARGENS', 'ARVIEUX', 'EMBRUN',
       'LA FAURIE', 'GAP','RISTOLAS', 'ST JEAN-ST-NICOLAS', 'TALLARD', "VILLAR D'ARENE",
       'VILLAR ST PANCRACE', 'ANTIBES-GAROUPE', 'ASCROS', 'CANNES', 'CAUSSOLS', 'PEIRA CAVA', 'PEILLE', 'PEONE',
       'CHAPELLE-EN-VER', 'LUS L CROIX HTE', 'TRICASTIN', 'ST ROMAN-DIOIS', 'CREYS-MALVILLE',
       "ALPE-D'HUEZ", 'LA MURE- RADOME', 'GRENOBLE-ST GEOIRS', 'MERIBEL BURGIN',
       'ST-ALBAN', 'ST-PIERRE-LES EGAUX', 'GRENOBLE - LVD', 'VILLARD-DE-LANS', 'CHAMROUSSE', 'ALBERTVILLE JO',
       'MERIBEL BURGIN', 'MONT DU CHAT', 'FECLAZ_SAPC', 'COL-DES-SAISIES', 'FREJUS', 'LA MASSE',
       'ST MICHEL MAUR_SAPC', 'TIGNES_SAPC', "VAL D'I SOLAISE", 'LE TOUR',
       'AGUIL. DU MIDI', 'LE GRAND-BORNAND', 'MEYTHET', 'LE PLENAY', 'SEYNOD-AREA']

    prm["list_nivose"] = ['RESTEFOND-NIVOSE', 'PARPAILLON-NIVOSE', 'LA MEIJE-NIVOSE', 'COL AGNEL-NIVOSE',
       'GALIBIER-NIVOSE', 'ORCIERES-NIVOSE', 'MILLEFONTS-NIVOSE', 'AIGLETON-NIVOSE',
       'LE GUA-NIVOSE', 'LES ECRINS-NIVOSE', 'ST HILAIRE-NIVOSE', 'BONNEVAL-NIVOSE',
       'BELLECOTE-NIVOSE', 'GRANDE PAREI NIVOSE', 'ALLANT-NIVOSE', 'TIGNES_SAPC', 'LE CHEVRIL-NIVOSE',
       'LES ROCHILLES-NIVOSE', 'AIGUILLES ROUGES-NIVOSE']

    prm["list_additionnal"] = ['Vallot', 'Saint-Sorlin', 'Argentiere', 'Dome Lac Blanc', 'Col du Lac Blanc',
                               'La Muzelle Lac Blanc', 'Col de Porte']

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