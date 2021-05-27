## How to start training

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

