"""
# Interpolate final map
print("Final interpolation")
for time_step, time in enumerate(times):
    for component in range(3):

        # Select component to interpolate
        wind_component = wind_map[time_step, :, :, component]

        # Create x and y axis
        x = np.arange(0, wind_component.shape[1])
        y = np.arange(0, wind_component.shape[0])

        # Mask invalid values
        wind_component = np.ma.masked_invalid(wind_component)
        xx, yy = np.meshgrid(x, y)

        # Get only the valid values
        x1 = xx[~wind_component.mask]
        y1 = yy[~wind_component.mask]
        newarr = wind_component[~wind_component.mask]

        # Interpolate
        wind_map[time_step, :, :, component] = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
"""