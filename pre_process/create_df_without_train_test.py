import os
import numpy as np
import pandas as pd

def create_list_simu_by_degree(degree, input_dir):
    """Create two list containing names for topographies and simulatins"""
    degree_str = str(degree) + 'degree/'

    # path to topographies files
    topo_dir = input_dir + "dem/" + degree_str

    # path to wind files
    wind_comp_list = ["ucompwind/", "vcompwind/", "wcompwind/"]
    wind_dir = input_dir + "Wind/" + "U_V_W/"
    wind_comp_dirs = [wind_dir + wind_comp_dir for wind_comp_dir in wind_comp_list]

    # List of filenames (sorted)
    topo_list = sorted(os.listdir(topo_dir))
    wind_list = list(zip(*(sorted(os.listdir(wind_comp_dirs[i] + degree_str)) for i in range(3))))

    return (topo_list, wind_list)


def get_name_ARPS_simulation(degree, simulation):
    """Get short name of ARPS files"""
    [topo_or_wind, N, dx, xi, sigma, ext] = simulation.split('_')
    name = str(degree) + 'degree' + '_' + xi + '_' + ext
    return (name)


def get_xi_from_ARPS_simulation(simulation):
    """Extract xi from full name of ARPS files"""
    [topo_or_wind, N, dx, xi, sigma, ext] = simulation.split('_')
    xi = xi.split('xi')[1]
    return (xi)


def check_names(degree, topo_list, wind_list):
    """"Check that we extract corresponding names"""
    topo_names = [get_name_ARPS_simulation(degree, topo) for topo in topo_list]
    no_duplicate_in_topo = len(set(topo_names)) == len(topo_names)

    u_v_w_names = [(get_name_ARPS_simulation(degree, u), get_name_ARPS_simulation(degree, v), get_name_ARPS_simulation(degree, w)) for (u, v, w)
                   in wind_list]
    wind_names = [get_name_ARPS_simulation(degree, u) for (u, v, w) in wind_list]

    assert no_duplicate_in_topo
    for (u, v, w) in u_v_w_names: assert u == v == w
    assert topo_names == wind_names


def create_df_with_simulation_name(input_dir):
    """
    Create an array with the name of the files

    Output:
    degree 	xi 	degree_xi 	topo_name 	wind_name
0 	5 	1000 	degree5_xi1000 	gaussiandem_N5451_dx30_xi1000_sigma88_r000.txt 	(gaussianu_N5451_dx30_xi1000_sigma88_r000.txt,...
1 	5 	1000 	degree5_xi1000 	gaussiandem_N5451_dx30_xi1000_sigma88_r001.txt 	(gaussianu_N5451_dx30_xi1000_sigma88_r001.txt,...
    """
    degrees = [5, 10, 13, 16, 20]
    all_info = np.array([])
    for index, degree in enumerate(degrees):

        # Create list of names for topographies and winds
        topo_names, wind_names = create_list_simu_by_degree(degree, input_dir)

        # Create list of degrees
        list_degree = [degree] * len(topo_names)

        # Check names are not duplicated and well orga,ized
        check_names(degree, topo_names, wind_names)
        name_simu = [get_name_ARPS_simulation(degree, topo) for topo in topo_names]

        # Create list of xi
        list_xi = [get_xi_from_ARPS_simulation(simu) for simu in topo_names]

        # Create list of degree_xi
        list_degree_xi = ['degree' + str(degree) + '_' + 'xi' + str(xi) for (degree, xi) in zip(list_degree, list_xi)]

        # Store the result
        array_to_add = np.array([list_degree, list_xi, list_degree_xi, topo_names, wind_names])
        if index == 0: all_info = np.array([list_degree, list_xi, list_degree_xi, topo_names, wind_names])
        if index >= 1: all_info = np.concatenate((all_info, array_to_add), axis=1)

    all_info = np.transpose(all_info)
    df_all = pd.DataFrame(all_info, columns=['degree', 'xi', 'degree_xi', 'topo_name', 'wind_name'])
    return (df_all)