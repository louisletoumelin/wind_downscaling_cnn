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

    d1 = datetime.datetime(2017, 8, 1, 6)
    d2 = datetime.datetime(2018, 8, 1, 6)
    d3 = datetime.datetime(2019, 6, 1, 6)
    d6 = datetime.datetime(2020, 7, 1, 6)

    if (d1 < begin <= d2) and (d1 < end <= d2):
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
        path = "/".join(prm_path.split('/')[:-1]) + "/L93_npy/" + prm_path.split('/')[-1].split('.csv')[0].split('.nc')[0]
        return path
    else:
        return None


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

    return prm