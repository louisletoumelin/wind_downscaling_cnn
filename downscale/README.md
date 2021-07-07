## How to launch predictions

First specify the paths to data and specificities of the prediction (timescale, domain etc)

```
PRM_predict.py
```

To launch predictions at the *observation stations for a single month*

```
python pipeline_predict_stations.py
```

To launch predictions at the *observation stations for several months* (less data stored)

```
python pipeline_predict_long_periods.py
```

To launch predictions on *maps*

```
python pipeline_predict_map.py
```

To launch predictions using the method described in *Helbig et al. 2017*

```
python pipeline_downscale_helbig.py
```

To launch predictions at the *observation stations for several months calculated accordingly to the map production method* (less data stored). This method gives the data that will be evaluated.

```
python pipeline_predict_station_similar_to_map.py
```

To launch quality control of wind speed and direction observations. I intend to turn it into a module.

```
python pipeline_QC.py
```

