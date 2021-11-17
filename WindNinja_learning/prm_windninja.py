from utils_prm import select_cfg_file


def create_prm():

    prm = {}

    prm["stations"] = ["Col du Lac Blanc"]

    # 2 August 2017 1h
    prm["hour_begin"] = 6
    prm["day_begin"] = 4
    prm["month_begin"] = 8
    prm["year_begin"] = 2017

    # 31 May 2020 1h
    prm["hour_end"] = 1
    prm["day_end"] = 5
    prm["month_end"] = 8
    prm["year_end"] = 2017

    prm["begin"] = f"{prm['year_begin']}-{prm['month_begin']}-{prm['day_begin']}"
    prm["end"] = f"{prm['year_end']}-{prm['month_end']}-{prm['day_end']}"

    # Parent directory
    prm["working_directory"] = "C:/Users/louis/git/wind_downscaling_CNN/"

    # Data
    prm["data_path"] = prm["working_directory"] + "Data/1_Raw/"

    # Topography
    prm["topo_path"] = prm["data_path"] + "MNT/Copernicus/COP30_L93_cropped.tif"
    prm["windninja_topo"] = 'C:/Users/louis/git/wind_downscaling_CNN/Data/2_Pre_processed/WindNinja/'

    # WindNinja
    prm["output_path"] = 'C:/Users/louis/git/wind_downscaling_CNN/Data/2_Pre_processed/WindNinja/'
    prm["path_to_WindNinja"] = "C:/WindNinja/WindNinja-3.7.2/bin/"
    prm["solver"] = "momentum" # "momentum" or "mass"

    # Observations
    prm["BDclim_stations_path"] = prm["working_directory"] + "Data/2_Pre_processed/WindNinja/" + "stations_with_nearest_neighbors.csv"

    # AROME
    prm["nwp_name"] = "AROME"
    prm["AROME_files"] = [prm["data_path"]+"AROME/FORCING_alp_2017080106_2018080106_32bits.nc", prm["data_path"]+"AROME/FORCING_alp_2018080106_2019060106_32bits.nc"]
    prm["AROME_files_temperature"] = [prm["data_path"]+"AROME/T2m_FORCING_alp_2017080106_2018080106_32bits.nc", prm["data_path"]+"AROME/T2m_FORCING_alp_2018080106_2019060106_32bits.nc"]

    prm = select_cfg_file(prm)

    return prm
