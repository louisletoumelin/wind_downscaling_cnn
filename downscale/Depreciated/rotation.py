def predict_map(self, station_name='Col du Lac Blanc', x_0=None, y_0=None, dx=10_000, dy=10_000, interp=3,
                year_0=None, month_0=None, day_0=None, hour_0=None,
                year_1=None, month_1=None, day_1=None, hour_1=None,
                Z0_cond=False, verbose=True, peak_valley=True):
    # Select NWP data
    if verbose: print("Selecting NWP")
    nwp_data = self._select_large_domain_around_station(station_name, dx, dy, type="NWP", additionnal_dx_mnt=None)
    begin = datetime.datetime(year_0, month_0, day_0, hour_0)
    end = datetime.datetime(year_1, month_1, day_1, hour_1)
    nwp_data = nwp_data.sel(time=slice(begin, end))
    nwp_data_initial = nwp_data

    # Calculate U_nwp and V_nwp
    if verbose: print("U_nwp and V_nwp computation")
    nwp_data = nwp_data.assign(theta=lambda x: (np.pi / 180) * (x["Wind_DIR"] % 360))
    nwp_data = nwp_data.assign(U=lambda x: -x["Wind"] * np.sin(x["theta"]))
    nwp_data = nwp_data.assign(V=lambda x: -x["Wind"] * np.cos(x["theta"]))
    nwp_data = nwp_data.drop_vars(["Wind", "Wind_DIR"])

    # Interpolate AROME
    if verbose: print("AROME interpolation")
    new_x = np.linspace(nwp_data["xx"].min().data, nwp_data["xx"].max().data, nwp_data.dims["xx"] * interp)
    new_y = np.linspace(nwp_data["yy"].min().data, nwp_data["yy"].max().data, nwp_data.dims["yy"] * interp)
    nwp_data = nwp_data.interp(xx=new_x, yy=new_y, method='linear')
    nwp_data = nwp_data.assign(Wind=lambda x: np.sqrt(x["U"] ** 2 + x["V"] ** 2))
    nwp_data = nwp_data.assign(Wind_DIR=lambda x: np.mod(180 + np.rad2deg(np.arctan2(x["U"], x["V"])), 360))

    # Time scale and domain length
    times = nwp_data.time.data
    nwp_x_l93 = nwp_data.X_L93
    nwp_y_l93 = nwp_data.Y_L93
    nb_time_step = len(times)
    nb_px_nwp_y, nb_px_nwp_x = nwp_x_l93.shape

    # Select MNT data
    if verbose: print("Selecting NWP")
    mnt_data = self._select_large_domain_around_station(station_name, dx, dy, type="MNT", additionnal_dx_mnt=2_000)
    xmin_mnt = np.nanmin(mnt_data.x.data)
    ymax_mnt = np.nanmax(mnt_data.y.data)

    # NWP forcing data
    if verbose: print("Selecting forcing data")
    wind_speed_nwp = nwp_data["Wind"].data
    wind_DIR_nwp = nwp_data["Wind_DIR"].data
    if Z0_cond:
        Z0_nwp = nwp_data["Z0"].data
        Z0REL_nwp = nwp_data["Z0REL"].data
        ZS_nwp = nwp_data["ZS"].data

    # Weight
    x, y = np.meshgrid(np.linspace(-1, 1, self.n_col), np.linspace(-1, 1, self.n_rows))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 0.025, 0
    gaussian_weight = 0.5 + 50 * np.array(np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))))

    # Initialize wind map
    if _dask:
        shape_x_mnt, shape_y_mnt = mnt_data.data.shape[1:]
        mnt_data_x = mnt_data.x.data
        mnt_data_y = mnt_data.y.data
        mnt_data = mnt_data.data
    else:
        mnt_data_x = mnt_data.x.data
        mnt_data_y = mnt_data.y.data
        mnt_data = mnt_data.__xarray_dataarray_variable__.data
        shape_x_mnt, shape_y_mnt = mnt_data[0, :, :].shape

    wind_map = np.zeros((nb_time_step, shape_x_mnt, shape_y_mnt, 3), dtype=np.float32)
    weights = np.zeros((nb_time_step, shape_x_mnt, shape_y_mnt), dtype=np.float32)
    peak_valley_height = np.empty((nb_px_nwp_y, nb_px_nwp_x), dtype=np.float32)
    topo_concat = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69), dtype=np.float32)

    # Concatenate topographies along single axis
    if verbose: print("Concatenate topographies along single axis")
    nb_pixel = 70
    for time_step, time in enumerate(times):
        for idx_y_nwp in range(nb_px_nwp_y):
            for idx_x_nwp in range(nb_px_nwp_x):
                # Select index NWP
                x_nwp_L93 = nwp_x_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data
                y_nwp_L93 = nwp_y_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data

                # Select indexes MNT
                idx_x_mnt = int((x_nwp_L93 - xmin_mnt) // self.mnt.resolution_x)
                idx_y_mnt = int((ymax_mnt - y_nwp_L93) // self.mnt.resolution_y)

                # Large topo
                topo_i = mnt_data[0, idx_y_mnt - nb_pixel:idx_y_mnt + nb_pixel,
                         idx_x_mnt - nb_pixel:idx_x_mnt + nb_pixel]
                # mnt_x_i = mnt_data.x.data[idx_x_mnt - nb_pixel:idx_x_mnt + nb_pixel]
                # mnt_y_i = mnt_data.y.data[idx_y_mnt - nb_pixel:idx_y_mnt + nb_pixel]

                # Mean peak_valley altitude
                if time_step == 0:
                    peak_valley_height[idx_y_nwp, idx_x_nwp] = np.int32(2 * np.nanstd(topo_i))

                # Wind direction
                wind_DIR = wind_DIR_nwp[time_step, idx_y_nwp, idx_x_nwp]

                # Rotate topography
                topo_i = self.rotate_topography(topo_i, wind_DIR)
                topo_i = topo_i[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]

                # Store result
                topo_concat[time_step, idx_y_nwp, idx_x_nwp, :, :] = topo_i

    # Reshape for tensorflow
    topo_concat = topo_concat.reshape((nb_time_step * nb_px_nwp_x * nb_px_nwp_y, self.n_rows, self.n_col, 1))

    # Normalize
    if verbose: print("Topographies normalization")
    mean, std = self._load_norm_prm()
    topo_concat = self.normalize_topo(topo_concat, mean, std).astype(dtype=np.float32, copy=False)

    # Load model
    self.load_model(dependencies=True)

    # Predictions
    if verbose: print("Predictions")
    prediction = self.model.predict(topo_concat)

    # Reshape predictions for analysis
    prediction = prediction.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, self.n_rows, self.n_col, 3)).astype(
        np.float32, copy=False)

    # Wind speed scaling for broadcasting
    wind_speed_nwp = wind_speed_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1)).astype(np.float32,
                                                                                                      copy=False)
    wind_DIR_nwp = wind_DIR_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1)).astype(np.float32,
                                                                                               copy=False)

    # Exposed wind speed
    if verbose: print("Exposed wind speed")
    if Z0_cond:

        Z0_nwp = Z0_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
        Z0REL_nwp = Z0REL_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
        ZS_nwp = ZS_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
        peak_valley_height = peak_valley_height.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))

        # Choose height in the formula
        if peak_valley:
            height = peak_valley_height
        else:
            height = ZS_nwp

        if self._numexpr:
            acceleration_factor = ne.evaluate(
                "log((height/2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp+Z0_nwp))**0.0706 / (log((height/2) / (Z0REL_nwp+Z0_nwp)))")
            acceleration_factor = ne.evaluate("where(acceleration_factor > 0, acceleration_factor, 1)")
            exp_Wind = ne.evaluate("wind_speed_all * acceleration_factor")
        else:
            acceleration_factor = np.log((height / 2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp + Z0_nwp)) ** 0.0706 / (
                np.log((height / 2) / (Z0REL_nwp + Z0_nwp)))
            acceleration_factor = np.where(acceleration_factor > 0, acceleration_factor, 1)
            exp_Wind = wind_speed_all * acceleration_factor

    # Wind speed scaling
    if verbose: print("Wind speed scaling")
    scaling_wind = exp_Wind if Z0_cond else wind_speed_nwp
    if self._numexpr:
        prediction = ne.evaluate("scaling_wind * prediction / 3")
    else:
        prediction = scaling_wind * prediction / 3

    # Wind computations
    if verbose: print("Wind computations")
    U_old = prediction[:, :, :, :, :, 0]  # Expressed in the rotated coord. system [m/s]
    V_old = prediction[:, :, :, :, :, 1]  # Expressed in the rotated coord. system [m/s]
    W_old = prediction[:, :, :, :, :, 2]  # Good coord. but not on the right pixel [m/s]

    if self._numexpr:
        UV = ne.evaluate("sqrt(U_old**2 + V_old**2)")  # Good coord. but not on the right pixel [m/s]
        alpha = ne.evaluate(
            "where(U_old == 0, where(V_old == 0, 0, V_old/abs(V_old) * 3.14159 / 2), arctan(V_old / U_old))")
        UV_DIR = ne.evaluate(
            "(3.14159/180) * wind_DIR_nwp - alpha")  # Good coord. but not on the right pixel [radian]
    else:
        UV = np.sqrt(U_old ** 2 + V_old ** 2)  # Good coord. but not on the right pixel [m/s]
        alpha = np.where(U_old == 0,
                         np.where(V_old == 0, 0, np.sign(V_old) * np.pi / 2),
                         np.arctan(V_old / U_old))  # Expressed in the rotated coord. system [radian]
        UV_DIR = (np.pi / 180) * wind_DIR_nwp - alpha  # Good coord. but not on the right pixel [radian]

    # float64 to float32
    UV_DIR = UV_DIR.astype(dtype=np.float32, copy=False)

    # Reshape wind speed and wind direction
    wind_speed_nwp = wind_speed_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x))
    wind_DIR_nwp = wind_DIR_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x))

    # Calculate U and V along initial axis
    if self._numexpr:
        U_old = ne.evaluate("-sin(UV_DIR) * UV")
        V_old = ne.evaluate("-cos(UV_DIR) * UV")
    else:
        U_old = -np.sin(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]
        V_old = -np.cos(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]

    # Final results
    print("Creating wind map")

    # Good axis and pixel location [m/s]
    for time_step, time in enumerate(times):
        for idx_y_nwp in range(nb_px_nwp_y):
            for idx_x_nwp in range(nb_px_nwp_x):
                wind_DIR = wind_DIR_nwp[time_step, idx_y_nwp, idx_x_nwp]
                U = self.rotate_topography(
                    U_old[time_step, idx_y_nwp, idx_x_nwp, :, :],
                    wind_DIR,
                    clockwise=True)
                V = self.rotate_topography(
                    V_old[time_step, idx_y_nwp, idx_x_nwp, :, :],
                    wind_DIR,
                    clockwise=True)
                W = self.rotate_topography(
                    W_old[time_step, idx_y_nwp, idx_x_nwp, :, :],
                    wind_DIR,
                    clockwise=True)

                # Select index NWP
                x_nwp_L93 = nwp_x_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data
                y_nwp_L93 = nwp_y_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data

                # Select indexes MNT
                idx_x_mnt = int((x_nwp_L93 - xmin_mnt) // self.mnt.resolution_x)
                idx_y_mnt = int((ymax_mnt - y_nwp_L93) // self.mnt.resolution_y)

                # Select center of the predictions
                wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 0] = U[39 - 8:40 + 8,
                                                                                                   34 - 8:35 + 8]

                wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 1] = V[39 - 8:40 + 8,
                                                                                                   34 - 8:35 + 8]

                wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 2] = W[39 - 8:40 + 8,
                                                                                                   34 - 8:35 + 8]

    # wind_map = wind_map / weights.reshape((nb_time_step, shape_x_mnt, shape_y_mnt, 1))
    return (wind_map, weights, nwp_data_initial, nwp_data, mnt_data)


def predict_map_tensorflow(self, station_name='Col du Lac Blanc', x_0=None, y_0=None, dx=10_000, dy=10_000,
                           interp=3,
                           year_0=None, month_0=None, day_0=None, hour_0=None,
                           year_1=None, month_1=None, day_1=None, hour_1=None,
                           Z0_cond=False, verbose=True, peak_valley=True):

    # Select NWP data
    if verbose: print("Selecting NWP")
    nwp_data = self._select_large_domain_around_station(station_name, dx, dy, type="NWP", additionnal_dx_mnt=None)
    begin = datetime.datetime(year_0, month_0, day_0, hour_0)  # datetime
    end = datetime.datetime(year_1, month_1, day_1, hour_1)  # datetime
    nwp_data = nwp_data.sel(time=slice(begin, end))
    nwp_data_initial = nwp_data

    # Calculate U_nwp and V_nwp
    if verbose: print("U_nwp and V_nwp computation")
    nwp_data = nwp_data.assign(theta=lambda x: (np.pi / 180) * (x["Wind_DIR"] % 360))
    nwp_data = nwp_data.assign(U=lambda x: -x["Wind"] * np.sin(x["theta"]))
    nwp_data = nwp_data.assign(V=lambda x: -x["Wind"] * np.cos(x["theta"]))
    nwp_data = nwp_data.drop_vars(["Wind", "Wind_DIR"])

    # Interpolate AROME
    if verbose: print("AROME interpolation")
    new_x = np.linspace(nwp_data["xx"].min().data, nwp_data["xx"].max().data, nwp_data.dims["xx"] * interp)
    new_y = np.linspace(nwp_data["yy"].min().data, nwp_data["yy"].max().data, nwp_data.dims["yy"] * interp)
    nwp_data = nwp_data.interp(xx=new_x, yy=new_y, method='linear')
    nwp_data = nwp_data.assign(Wind=lambda x: np.sqrt(x["U"] ** 2 + x["V"] ** 2))
    nwp_data = nwp_data.assign(Wind_DIR=lambda x: np.mod(180 + np.rad2deg(np.arctan2(x["U"], x["V"])), 360))

    # Time scale and domain length
    times = nwp_data.time.data
    nwp_x_l93 = nwp_data.X_L93
    nwp_y_l93 = nwp_data.Y_L93
    nb_time_step = len(times)
    nb_px_nwp_y, nb_px_nwp_x = nwp_x_l93.shape

    # Select MNT data
    if verbose: print("Selecting NWP")
    mnt_data = self._select_large_domain_around_station(station_name, dx, dy, type="MNT", additionnal_dx_mnt=2_000)
    xmin_mnt = np.nanmin(mnt_data.x.data)
    ymax_mnt = np.nanmax(mnt_data.y.data)
    if _dask:
        shape_x_mnt, shape_y_mnt = mnt_data.data.shape[1:]
        mnt_data_x = mnt_data.x.data
        mnt_data_y = mnt_data.y.data
        mnt_data = tf.constant(mnt_data.data, dtype=tf.float32)
    else:
        mnt_data_x = mnt_data.x.data
        mnt_data_y = mnt_data.y.data
        mnt_data = tf.constant(mnt_data.__xarray_dataarray_variable__.data, dtype=tf.float32)
        shape_x_mnt, shape_y_mnt = mnt_data[0, :, :].shape

    # NWP forcing data
    if verbose: print("Selecting forcing data")
    wind_speed_nwp = tf.constant(nwp_data["Wind"].data, dtype=tf.float32)
    wind_DIR_nwp = tf.constant(nwp_data["Wind_DIR"].data, dtype=tf.float32)
    if Z0_cond:
        Z0_nwp = tf.constant(nwp_data["Z0"].data, dtype=tf.float32)
        Z0REL_nwp = tf.constant(nwp_data["Z0REL"].data, dtype=tf.float32)
        ZS_nwp = tf.constant(nwp_data["ZS"].data, dtype=tf.float32)

    # Initialize wind map
    wind_map = np.empty((nb_time_step, shape_x_mnt, shape_y_mnt, 3))

    # Concatenate topographies along single axis
    if verbose: print("Concatenate topographies along single axis")
    topo_concat = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69), dtype=np.float32)
    peak_valley_height = np.empty((nb_px_nwp_y, nb_px_nwp_x), dtype=np.float32)

    nb_pixel = 70
    for time_step, time in enumerate(times):
        for idx_y_nwp in range(nb_px_nwp_y):
            for idx_x_nwp in range(nb_px_nwp_x):
                # Select index NWP
                x_nwp_L93 = nwp_x_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data
                y_nwp_L93 = nwp_y_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data

                # Select indexes MNT
                idx_x_mnt = int((x_nwp_L93 - xmin_mnt) // self.mnt.resolution_x)
                idx_y_mnt = int((ymax_mnt - y_nwp_L93) // self.mnt.resolution_y)

                # Large topo
                topo_i = mnt_data[0, idx_y_mnt - nb_pixel:idx_y_mnt + nb_pixel,
                         idx_x_mnt - nb_pixel:idx_x_mnt + nb_pixel]
                # mnt_x_i = mnt_data.x.data[idx_x_mnt - nb_pixel:idx_x_mnt + nb_pixel]
                # mnt_y_i = mnt_data.y.data[idx_y_mnt - nb_pixel:idx_y_mnt + nb_pixel]

                # Mean peak_valley altitude
                if time_step == 0:
                    peak_valley_height[idx_y_nwp, idx_x_nwp] = np.int32(2 * np.nanstd(topo_i))

                # Wind direction
                wind_DIR = wind_DIR_nwp[time_step, idx_y_nwp, idx_x_nwp]

                # Rotate topography
                topo_i = self.rotate_topography(topo_i.numpy(), wind_DIR.numpy())
                topo_i = topo_i[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]

                # Store result
                topo_concat[time_step, idx_y_nwp, idx_x_nwp, :, :] = topo_i

    with tf.device('/GPU:0'):
        # Reshape for tensorflow
        topo_concat = tf.constant(topo_concat, dtype=tf.float32)
        topo_concat = tf.reshape(topo_concat,
                                 [nb_time_step * nb_px_nwp_x * nb_px_nwp_y, self.n_rows, self.n_col, 1])

        # Normalize
        if verbose: print("Topographies normalization")
        mean, std = self._load_norm_prm()
        topo_concat = self.normalize_topo(topo_concat, mean, std, librairie='tensorflow')

        # Load model
        self.load_model(dependencies=True)

        # Predictions
        if verbose: print("Predictions")
        prediction = self.model.predict(topo_concat)
        del topo_concat
        # Reshape predictions for analysis
        prediction = tf.reshape(prediction, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x, self.n_rows, self.n_col, 3])

        # Wind speed scaling for broadcasting
        wind_speed_nwp = tf.reshape(wind_speed_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1])
        wind_DIR_nwp = tf.reshape(wind_DIR_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1])

        # Exposed wind speed
        if verbose: print("Exposed wind speed")
        if Z0_cond:
            Z0_nwp = tf.reshape(Z0_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1])
            Z0REL_nwp = tf.reshape(Z0REL_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1])
            peak_valley_height = tf.reshape(peak_valley_height, [1, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1])

            # Choose height in the formula
            if peak_valley:
                height = peak_valley_height
            else:
                height = ZS_nwp

            acceleration_factor = tf.math.log((height / 2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp + Z0_nwp)) ** 0.0706 / (
                np.log((height / 2) / (Z0REL_nwp + Z0_all)))
            acceleration_factor = np.where(acceleration_factor > 0, acceleration_factor, 1)
            exp_Wind = wind_speed_all * acceleration_factor

        # Wind speed scaling
        if verbose: print("Wind speed scaling")
        scaling_wind = exp_Wind if Z0_cond else wind_speed_nwp
        prediction = scaling_wind * prediction / 3

        # Wind computations
        if verbose: print("Wind computations")
        U_old = prediction[:, :, :, :, :, 0]  # Expressed in the rotated coord. system [m/s]
        V_old = prediction[:, :, :, :, :, 1]  # Expressed in the rotated coord. system [m/s]
        W_old = prediction[:, :, :, :, :, 2]  # Good coord. but not on the right pixel [m/s]
        del prediction
        UV = tf.math.sqrt(tf.square(U_old) + tf.square(V_old))  # Good coord. but not on the right pixel [m/s]
        alpha = tf.where(U_old == 0,
                         tf.where(V_old == 0, 0, tf.math.sign(V_old) * 3.14159 / 2),
                         tf.math.atan(V_old / U_old))  # Expressed in the rotated coord. system [radian]
        UV_DIR = (3.14159 / 180) * wind_DIR_nwp - alpha  # Good coord. but not on the right pixel [radian]

        # Reshape wind speed and wind direction
        wind_speed_nwp = tf.reshape(wind_speed_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x])
        wind_DIR_nwp = tf.reshape(wind_DIR_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x])

        # Calculate U and V along initial axis
        U_old = -tf.math.sin(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]
        V_old = -tf.math.sin(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]
        del UV
        del UV_DIR
        del alpha
        del wind_speed_nwp
        # Final results
        print("Creating wind map")

    # Good axis and pixel location [m/s]
    for time_step, time in enumerate(times):
        for idx_y_nwp in range(nb_px_nwp_y):
            for idx_x_nwp in range(nb_px_nwp_x):
                wind_DIR = wind_DIR_nwp[time_step, idx_y_nwp, idx_x_nwp].numpy()
                U = self.rotate_topography(
                    U_old[time_step, idx_y_nwp, idx_x_nwp, :, :].numpy(),
                    wind_DIR,
                    clockwise=True)
                V = self.rotate_topography(
                    V_old[time_step, idx_y_nwp, idx_x_nwp, :, :].numpy(),
                    wind_DIR,
                    clockwise=True)
                W = self.rotate_topography(
                    W_old[time_step, idx_y_nwp, idx_x_nwp, :, :].numpy(),
                    wind_DIR,
                    clockwise=True)

                # Select index NWP
                x_nwp_L93 = nwp_x_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data
                y_nwp_L93 = nwp_y_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data

                # Select indexes MNT
                idx_x_mnt = int((x_nwp_L93 - xmin_mnt) // self.mnt.resolution_x)
                idx_y_mnt = int((ymax_mnt - y_nwp_L93) // self.mnt.resolution_y)

                # Select center of the predictions
                wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 0] = U[39 - 8:40 + 8,
                                                                                                   34 - 8:35 + 8]

                wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 1] = V[39 - 8:40 + 8,
                                                                                                   34 - 8:35 + 8]

                wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 2] = W[39 - 8:40 + 8,
                                                                                                   34 - 8:35 + 8]

    return (wind_map, [], nwp_data_initial, nwp_data, mnt_data)