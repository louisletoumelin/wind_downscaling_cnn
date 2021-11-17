import numpy as np
import datetime


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
        prm["selected_path"] = prm["AROME_path"]

    return prm


def update_selected_path_for_long_periods(begin, end, prm):

    if prm["GPU"]:
        d1 = datetime.datetime(2017, 8, 1, 6)
        d2 = datetime.datetime(2018, 8, 1, 6)
        d3 = datetime.datetime(2019, 5, 1, 6)
        d4 = datetime.datetime(2019, 6, 1, 6)
        d5 = datetime.datetime(2020, 6, 2, 6)

        prm["AROME_path_1"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2017080106_2018080106_32bits.nc"
        prm["AROME_path_2"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2018080106_2019050106_32bits.nc"
        prm["AROME_path_3"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2019050106_2019060106_32bits.nc"
        # prm["AROME_path_3"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2019060107_2019070106_32bits.nc"
        prm["AROME_path_4"] = prm["data_path"] + "AROME/32bits/FORCING_alp_2019060106_2020060206_32bits.nc"


        if (d1 <= begin < d2) and (d1 <= end < d2):
            prm["selected_path"] = prm["AROME_path_1"]
        elif (d2 <= begin < d3) and (d2 <= end < d3):
            prm["selected_path"] = prm["AROME_path_2"]
        elif (d3 <= begin < d4) and (d3 <= end < d4):
            prm["selected_path"] = prm["AROME_path_3"]
        elif (d4 <= begin < d5) and (d4 <= end < d5):
            prm["selected_path"] = prm["AROME_path_4"]
    else:
        d1 = datetime.datetime(2017, 8, 1, 6)
        d2 = datetime.datetime(2018, 8, 1, 6)
        d3 = datetime.datetime(2019, 6, 1, 6)
        d6 = datetime.datetime(2020, 7, 1, 6)

        if (d1 <= begin <= d2) and (d1 <= end <= d2):
            prm["selected_path"] = prm["AROME_path_1"]
        elif (d2 < begin <= d3) and (d2 < end <= d3):
            prm["selected_path"] = prm["AROME_path_2"]
        elif (d3 < begin <= d6) and (d3 < end <= d6):
            prm["selected_path"] = prm["AROME_path_4"]

    return prm


def update_selected_path(prm, month_prediction, year_end=None, month_end=None, day_end=None, force_date=False):
    if month_prediction:
        if force_date:
            prm = _update_selected_path(year_end, month_end, day_end, prm)
        else:
            prm = _update_selected_path(prm["year_end"], prm["month_end"], prm["day_end"], prm)
    else:
        prm["selected_path"] = prm["AROME_path"]
    return prm


def select_path_to_coord_L93(prm):

    if prm["GPU"]:
        prm_path = prm["selected_path"]
        path = "/".join(prm_path.split('/')[:-1]) + "/L93_npy/" + prm_path.split('/')[-1].split('.csv')[0].split('.nc')[
            0]
        prm["path_to_coord_L93"] = path
    else:
        prm["path_to_coord_L93"] = None

    return prm


def add_additional_stations(prm):
    if not prm["add_additional_stations"]:
        prm["path_vallot"] = None
        prm["path_saint_sorlin"] = None
        prm["path_argentiere"] = None
        prm["path_Dome_Lac_Blanc"] = None
        prm["path_Col_du_Lac_Blanc"] = None
        prm["path_Muzelle_Lac_Blanc"] = None
        prm["path_Col_de_Porte"] = None
        prm["path_Col_du_Lautaret"] = None

    return prm


def select_stations(prm, observation):
    if prm["stations_to_predict"] == "all":

        all_stations = observation.time_series["name"].unique()
        prm["stations_to_predict"] = all_stations

        prm["stations_to_predict"] = list(prm["stations_to_predict"])

        stations_to_reject = ['ANTIBES-GAROUPE', 'CANNES', 'SEYNOD-AREA', 'TIGNES_SAPC', 'ST MICHEL MAUR_SAPC',
                              'FECLAZ_SAPC', 'Dome Lac Blanc', 'MERIBEL BURGIN', "VAL D'I SOLAISE", 'CAP FERRAT',
                              'ALBERTVILLE', 'FREJUS', "VAL D'I BELLEVA", None]

        for station_to_reject in stations_to_reject:
            try:
                prm["stations_to_predict"].remove(station_to_reject)
            except ValueError:
                pass

        prm["stations_to_predict"] = np.array(prm["stations_to_predict"])

    return prm


def add_list_stations(prm):

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

    return prm


def check_expose_elevation(prm):

    if prm["peak_valley"]:
        prm["scale_at_10m"] = False
        prm["scale_at_max_altitude"] = False

    if prm["scale_at_max_altitude"]:
        prm["scale_at_10m"] = False

        return prm


def check_extract_around_station_or_interpolated(prm):

    prm["interp_str"] = "_interpolated" if prm["station_similar_to_map"] else ""
    prm["extract_around"] = "station" if not prm["centered_on_interpolated"] else "nwp_neighbor_interp"

    return prm


def create_begin_and_end_str(prm):

    prm["begin"] = str(prm["year_begin"]) + "-" + str(prm["month_begin"]) + "-" + str(prm["day_begin"])
    prm["begin_after"] = str(prm["year_begin"]) + "-" + str(prm["month_begin"]) + "-" + str(prm["day_begin"] + 1)
    prm["end"] = str(prm["year_end"]) + "-" + str(prm["month_end"]) + "-" + str(prm["day_end"])

    return prm


def check_save_and_load(prm):
    if prm["load_z0"] and prm["save_z0"]:
        prm["save_z0"] = False
    return prm


def append_module_path(prm):
    import sys
    sys.path.append(prm["path_module"])


def try_import_modules(prm):

    try:
        import dask
        prm["_dask"] = True
    except ModuleNotFoundError:
        prm["_dask"] = False

    try:
        import rasterio
        prm["_rasterio"] = True
    except ModuleNotFoundError:
        prm["_rasterio"] = False

    try:
        import pyproj
        prm["_pyproj"] = True
    except ModuleNotFoundError:
        prm["_pyproj"] = False

    try:
        from shapely.geometry import Point
        prm["_shapely_geometry"] = True
    except ModuleNotFoundError:
        prm["_shapely_geometry"] = False

    try:
        import concurrent.futures
        prm["_concurrent"] = True
    except:
        prm["_concurrent"] = False

    try:
        import geopandas as gpd
        prm["_geopandas"] = True
    except:
        prm["_geopandas"] = False

    try:
        import seaborn as sns
        sns.set(font_scale=2)
        sns.set_style("white")
        prm["_seaborn"] = True
    except ModuleNotFoundError:
        prm["_seaborn"] = False

    try:
        from numba import jit, prange, float64, float32, int32, int64

        prm["_numba"] = True
    except ModuleNotFoundError:
        prm["_numba"] = False

    try:
        import numexpr as ne

        prm["_numexpr"] = True
    except ModuleNotFoundError:
        prm["_numexpr"] = False

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        prm["_cartopy"] = True
    except ModuleNotFoundError:
        prm["_cartopy"] = False

    try:
        from shapely.geometry import Point
        from shapely.geometry import Polygon
        prm["_shapely_geometry"] = True
    except ModuleNotFoundError:
        prm["_shapely_geometry"] = False