def connect_GPU_to_horovod():
    import horovod.tensorflow.keras as hvd
    import tensorflow as tf
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    print(gpus)

def environment_GPU(GPU=True):
    if GPU:
        import os
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        #tf.get_logger().setLevel('WARNING')
        #tf.autograph.set_verbosity(0)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #tf.debugging.set_log_device_placement(True)

def select_range(month_begin, month_end, year_begin, year_end, date_begin, date_end):
    import pandas as pd
    if (month_end != month_begin) or (year_begin != year_end):
        dates = pd.date_range(date_begin, date_end, freq='M')
        iterator = zip(dates.day, dates.month, dates.year)
    else:
        dates = pd.to_datetime(date_end)
        iterator = zip([dates.day], [dates.month], [dates.year])
    return(iterator)


def check_save_and_load(load_z0, save_z0):
    if load_z0 and save_z0:
        load_z0 = True
        save_z0 = False
    return(load_z0, save_z0)


def print_current_line(time_step, nb_sim, division):
    nb_sim_divided = nb_sim // division
    for k in range(1, division+1):
        if time_step == k * nb_sim_divided: print(f" {k}/{division}")

def change_dtype_if_required(variable, dtype):
    if variable.dtype != dtype:
        variable = variable.astype(dtype, copy=False)
    return(variable)

def assert_equal_shapes(arrays, shape):
    assert arrays[0].shape == shape
    for k in range(len(arrays)-1):
        assert arrays[k].shape == arrays[k+1].shape



def _predict_at_stations(self, stations_name, fast=False, verbose=True, plot=False, Z0_cond=True, peak_valley=True,
                        log_profile_to_h_2=False, log_profile_from_h_2=False, log_profile_10m_to_3m=False,
                        ideal_case=False, input_speed=3, input_dir=270):
    """
    Wind downscaling operated at observation stations sites only.

    17 min on CPU
    3.7 minutes on GPU

    Parameters
    ----------
    stations_name : list of strings
        List containing station names
    Z0_cond : boolean
        To expose wind speed
    peak_valley : boolean
        Use mean peak valley height to expose wind. An other option is to use mean height.
    log_profile_ : boolean(s)
        If True, apply a log profile inside the function to adapt to calculation heights.
    ideal_case: boolean
        If True, run an ideal case during one day where the input speed and direction are specified by the user
    input_speed: float
        Input wind speed specified by the user for ideal cases (Default: 3 [m/s])
    input_dir: float
        Input wind direction specified by the user for ideal cases (Default: 270 [°], wind coming from the West)

    Returns
    -------
    array_xr : xarray DataFrame
        Result dataframe containing wind components, speeds, wind directions, accelerations and input data


    Exemple
    -------
    array_xr = p._predict_at_stations(['Col du Lac Blanc',
                         verbose=True,
                         Z0_cond=True,
                         peak_valley=True,
                         ideal_case=False)
    """

    # Select timeframe
    self._select_timeframe_nwp(ideal_case=ideal_case, verbose=True)

    # Simulation parameters
    time_xr = self._select_nwp_time_serie_at_pixel(random.choice(stations_name), time=True)
    nb_sim = len(time_xr)
    nb_station = len(stations_name)

    # initialize arrays
    topo, wind_speed_all, wind_dir_all, Z0_all, Z0REL_all, ZS_all, \
    peak_valley_height, mean_height, all_topo_HD, all_topo_x_small_l93, all_topo_y_small_l93, ten_m_array, \
    three_m_array = self._initialize_arrays(predict='stations_month', nb_station=nb_station, nb_sim=nb_sim)

    # Indexes
    nb_pixel = 70  # min = 116/2
    y_offset_left = nb_pixel - 39
    y_offset_right = nb_pixel + 40
    x_offset_left = nb_pixel - 34
    x_offset_right = nb_pixel + 35

    for idx_station, single_station in enumerate(stations_name):

        print(f"\nBegin downscaling at {single_station}")

        # Select nwp pixel
        wind_dir_all[idx_station, :], wind_speed_all[idx_station, :], Z0_all[idx_station, :], \
        Z0REL_all[idx_station, :], ZS_all[idx_station, :] = self._select_nwp_time_serie_at_pixel(single_station,
                                                                                            Z0=Z0_cond, time=False)
        # For ideal case, we define the input speed and direction
        if ideal_case:
            wind_speed_all[idx_station, :], \
            wind_dir_all[idx_station, :] = self._scale_wind_for_ideal_case(wind_speed_all[idx_station, :],
                                                                           wind_dir_all[idx_station, :],
                                                                           input_speed,
                                                                           input_dir)

        # Extract topography
        topo_HD, topo_x_l93, topo_y_l93 = self.observation.extract_MNT_around_station(single_station,
                                                                                      self.mnt,
                                                                                      nb_pixel,
                                                                                      nb_pixel)

        # Rotate topographies
        topo[idx_station, :, :, :, 0] = self.rotate_topography(topo_HD, wind_dir_all[idx_station, :], clockwise=False)[:, y_offset_left:y_offset_right, x_offset_left:x_offset_right]

        # Store results
        all_topo_HD[idx_station, :, :] = topo_HD[y_offset_left:y_offset_right, x_offset_left:x_offset_right]
        all_topo_x_small_l93[idx_station, :] = topo_x_l93[x_offset_left:x_offset_right]
        all_topo_y_small_l93[idx_station, :] = topo_y_l93[y_offset_left:y_offset_right]
        peak_valley_height[idx_station] = np.int32(2 * np.nanstd(all_topo_HD[idx_station, :, :]))
        mean_height[idx_station] = np.int32(np.nanmean(all_topo_HD[idx_station, :, :]))

    # Exposed wind
    if Z0_cond:
        peak_valley_height = peak_valley_height.reshape((nb_station, 1))
        Z0_all = np.where(Z0_all == 0, 1 * 10 ^ (-8), Z0_all)
        height = self.select_height_for_exposed_wind_speed(height=peak_valley_height,
                                                           zs=ZS_all,
                                                           peak_valley=peak_valley)
        wind1 = np.copy(wind_speed_all)

        # Log profile
        if log_profile_to_h_2:
            wind_speed_all = self.apply_log_profile(z_in=ten_m_array, z_out=height / 2, wind_in=wind_speed_all,
                                                    z0=Z0_all,
                                                    verbose=verbose, z_in_verbose="10m", z_out_verbose="height/2")
        a1 = self.wind_speed_ratio(num=wind_speed_all, den=wind1)
        wind2 = np.copy(wind_speed_all)

        # Expose wind
        exp_Wind, acceleration_factor = self.exposed_wind_speed(wind_speed=wind_speed_all,
                                                                z_out=height / 2,
                                                                z0=Z0_all,
                                                                z0_rel=Z0REL_all)
        a2 = self.wind_speed_ratio(num=exp_Wind, den=wind2)
        del wind2
        wind3 = np.copy(exp_Wind)

        # Log profile
        if log_profile_from_h_2:
            exp_Wind = self.apply_log_profile(z_in=height / 2, z_out=three_m_array, wind_in=exp_Wind, z0=Z0_all,
                                              verbose=verbose, z_in_verbose="height/2", z_out_verbose="3m")
        a3 = self.wind_speed_ratio(num=exp_Wind, den=wind3)
        del wind3

    # Normalize
    _, std = self._load_norm_prm()
    std = self.get_closer_from_learning_conditions(topo, mean_height, std)
    topo = self.normalize_topo(topo, mean_height.reshape((nb_station, 1, 1, 1, 1)), std)

    # Reshape for tensorflow
    topo = topo.reshape((nb_station * nb_sim, self.n_rows, self.n_col, 1))
    if verbose: print('__Reshaped tensorflow done')

    """
    Warning: change dependencies here
    """
    # Load model
    self.load_model(dependencies=True)

    # Predictions
    prediction = self.model.predict(topo)
    if verbose: print('__Prediction done')

    # Acceleration NWP to CNN
    UVW_int = self.compute_wind_speed(U=prediction[:, :, :, 0], V=prediction[:, :, :, 1], W=prediction[:, :, :, 2])
    acceleration_CNN = self.wind_speed_ratio(num=UVW_int, den=3 * np.ones(UVW_int.shape))

    # Reshape predictions for analysis
    prediction = prediction.reshape((nb_station, nb_sim, self.n_rows, self.n_col, 3))
    if verbose: print(f"__Prediction reshaped: {prediction.shape}")

    # Reshape for broadcasting
    wind_speed_all = wind_speed_all.reshape((nb_station, nb_sim, 1, 1, 1))
    if Z0_cond:
        exp_Wind, acceleration_factor, ten_m_array, three_m_array, Z0_all = self.reshape_list_array(
            list_array=[exp_Wind, acceleration_factor, ten_m_array, three_m_array, Z0_all],
            shape=(nb_station, nb_sim, 1, 1, 1))
        peak_valley_height = peak_valley_height.reshape((nb_station, 1, 1, 1, 1))
    wind_dir_all = wind_dir_all.reshape((nb_station, nb_sim, 1, 1))

    # Wind speed scaling
    scaling_wind = exp_Wind if Z0_cond else wind_speed_all
    prediction = self.wind_speed_scaling(scaling_wind, prediction, linear=True)

    # Copy wind variable
    wind4 = np.copy(prediction)

    if log_profile_10m_to_3m:
        # Apply log profile: 3m => 10m
        prediction = self.apply_log_profile(z_in=three_m_array, z_out=ten_m_array, wind_in=prediction, z0=Z0_all,
                                            verbose=verbose, z_in_verbose="3m", z_out_verbose="10m")

    # Acceleration a4
    a4 = self.wind_speed_ratio(num=prediction, den=wind4)
    del wind4

    # Wind computations
    U_old = prediction[:, :, :, :, 0].view()  # Expressed in the rotated coord. system [m/s]
    V_old = prediction[:, :, :, :, 1].view()  # Expressed in the rotated coord. system [m/s]
    W_old = prediction[:, :, :, :, 2].view()  # Good coord. but not on the right pixel [m/s]

    # Recalculate with respect to original coordinates
    UV = self.compute_wind_speed(U=U_old, V=V_old, W=None)  # Good coord. but not on the right pixel [m/s]
    alpha = self.angular_deviation(U_old, V_old)  # Expressed in the rotated coord. system [radian]
    UV_DIR = self.direction_from_alpha(wind_dir_all, alpha)  # Good coord. but not on the right pixel [radian]

    # Verification of shapes
    assert_equal_shapes([U_old, V_old, W_old, UV, alpha, UV_DIR], (nb_station, nb_sim, self.n_rows, self.n_col))

    # Calculate U and V along initial axis
    # Good coord. but not on the right pixel [m/s]
    prediction[:, :, :, :, 0], prediction[:, :, :, :, 1] = self.horizontal_wind_component(UV=UV,
                                                                                          UV_DIR=UV_DIR,
                                                                                          verbose=True)
    del UV_DIR

    # Rotate clockwise to put the wind value on the right topography pixel
    if verbose: print('__Start rotating to initial position')
    prediction = np.moveaxis(prediction, -1, 2)
    wind_dir_all = wind_dir_all.reshape((nb_station, nb_sim, 1))
    prediction = self.rotate_topography(prediction[:, :, :, :, :], wind_dir_all[:, :, :], clockwise=True, verbose=False)
    prediction = np.moveaxis(prediction, 2, -1)

    U = prediction[:, :, :, :, 0].view()
    V = prediction[:, :, :, :, 1].view()
    W = prediction[:, :, :, :, 2].view()

    if verbose: print('__Wind prediction rotated for initial topography')

    # Compute wind direction
    UV_DIR = self.direction_from_u_and_v(U, V)  # Good axis and pixel [degree]

    # UVW
    UVW = self.compute_wind_speed(U=U, V=V, W=W)

    # Acceleration NWP to CNN
    acceleration_all = self.wind_speed_ratio(num=UVW, den=wind1.reshape(
        (nb_station, nb_sim, 1, 1))) if Z0_cond else UVW * np.nan

    # Reshape after broadcasting
    wind_speed_all, wind_dir_all, Z0_all = self.reshape_list_array(list_array=[wind_speed_all, wind_dir_all, Z0_all],
                                                           shape=(nb_station, nb_sim))
    if Z0_cond:
        exp_Wind, acceleration_factor, a1, a2, a3 = self.reshape_list_array(
            list_array=[exp_Wind, acceleration_factor, a1, a2, a3],
            shape=(nb_station, nb_sim))
        a4, acceleration_CNN = self.reshape_list_array(list_array=[np.max(a4, axis=4), acceleration_CNN],
                                                       shape=(nb_station, nb_sim, self.n_rows, self.n_col))
        peak_valley_height = peak_valley_height.reshape((nb_station))

    # Verification of shapes
    assert_equal_shapes([U,V,W,UV_DIR], (nb_station, nb_sim, self.n_rows, self.n_col))
    assert_equal_shapes([wind_speed_all,wind_dir_all], (nb_station, nb_sim))

    if verbose: print('__Reshape final predictions done')

    # Store results
    if verbose: print('__Start creating array')
    array_xr = xr.Dataset(data_vars={"U": (["station", "time", "y", "x"], U),
                                     "V": (["station", "time", "y", "x"], V),
                                     "W": (["station", "time", "y", "x"], W),
                                     "UV": (["station", "time", "y", "x"], np.sqrt(U ** 2 + V ** 2)),
                                     "UVW": (["station", "time", "y", "x"], self.compute_wind_speed(U=U, V=V, W=W)),
                                     "UV_DIR_deg": (["station", "time", "y", "x"], UV_DIR),
                                     "alpha_deg": (["station", "time", "y", "x"],
                                                   wind_dir_all.reshape((nb_station, nb_sim, 1, 1)) - UV_DIR),
                                     "NWP_wind_speed": (["station", "time"], wind_speed_all),
                                     "NWP_wind_DIR": (["station", "time"], wind_dir_all),
                                     "ZS_mnt": (["station", "y", "x"], all_topo_HD,),
                                     "peak_valley_height": (["station"], peak_valley_height),
                                     "XX": (["station", "x"], all_topo_x_small_l93,),
                                     "YY": (["station", "y"], all_topo_y_small_l93,),
                                     "exp_Wind": (
                                         ["station", "time"],
                                         exp_Wind if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                     "acceleration_factor": (
                                         ["station", "time"],
                                         acceleration_factor if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                     "a1": (
                                         ["station", "time"],
                                         a1 if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                     "a2": (
                                         ["station", "time"],
                                         a2 if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                     "a3": (
                                         ["station", "time"],
                                         a3 if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                     "a4": (
                                         ["station", "time", "y", "x"],
                                         a4 if Z0_cond else np.zeros(
                                             (nb_station, nb_sim, self.n_rows, self.n_col)),),
                                     "acceleration_all": (
                                         ["station", "time", "y", "x"],
                                         acceleration_all if Z0_cond else np.zeros(
                                             (nb_station, nb_sim, self.n_rows, self.n_col)),),
                                     "acceleration_CNN": (
                                         ["station", "time", "y", "x"],
                                         acceleration_CNN if Z0_cond else np.zeros(
                                             (nb_station, nb_sim, self.n_rows, self.n_col)),),
                                     "Z0": (
                                         ["station", "time"],
                                         Z0_all if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                     "Z0REL": (["station", "time"],
                                               Z0REL_all if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                     },

                          coords={"station": np.array(stations_name),
                                  "time": np.array(time_xr),
                                  "x": np.array(list(range(self.n_col))),
                                  "y": np.array(list(range(self.n_rows)))})
    if verbose: print('__Creating array done')

    return (array_xr)