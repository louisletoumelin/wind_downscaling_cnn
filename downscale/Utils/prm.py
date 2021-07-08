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


def select_path_to_file_npy(prm, GPU=False):
    if GPU:
        prm_path = prm["selected_path"]
        path = "/".join(prm_path.split('/')[:-1]) + "/L93_npy/" + prm_path.split('/')[-1].split('.csv')[0].split('.nc')[
            0]
        return path
    else:
        return None


def add_additionnal_stations(prm):
    if not prm["add_additionnal_stations"]:
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
                              'ALBERTVILLE','FREJUS', "VAL D'I BELLEVA", None]

        for station_to_reject in stations_to_reject:
            try:
                prm["stations_to_predict"].remove(station_to_reject)
            except ValueError:
                pass

        prm["stations_to_predict"] = np.array(prm["stations_to_predict"])

    return prm
