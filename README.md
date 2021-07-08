# Wind fields downscaling in complex terrain using convolutional neural networks

Work in progress

Ongoing study - PhD Louis Le Toumelin - louis.letoumelin@gmail.com

**Supervisors:** Isabelle Gouttevin, Fatima Karbou

*Centre Etudes de la Neige (Snow Research Center) - CNRM - CNRS  - Météo-France*


## How it works

*Learning*

**Input** : Synthetic topographies

**Output** : ARPS wind on synthetic topographies. Wind comes from the West at 3 m/s.

*Prediction*

**Input**:  
- NWP (ex: AROME) wind field (speed + direction)
- Topography (ex: IGN 25m)

**Output**:  
- Downscaled wind fields (U, V, W)


## Structure

```
├── LICENSE                              <- To be created
│
├── README.md                            <- File describing the github repository. You are reading it right now.
│
├── downscale                            <- Apply downscaling predictions 
│   ├── Analysis                         <- Analyse and visualize predictions
│      ├── Evaluation.py                 <- Evaluate predictions (RMSE, bias...etc)
│      ├── MidpointNormalize.py          <- Script to plot beautiful centered colorbars
│      └── Visualization.py              <- Plot results
│
│   ├── Data_family                      <- Process data according to its nature (model data, DEM, observation...)
│      ├── Data_2D.py                    <- Parent class of MNT and NWP
│      ├── MNT.py                        <- Treat Digital Elevation Models (DEM)
│      ├── NWP.py                        <- Treat Numerical Weather Prediction (NWP) models outputs 
│      └── Observation.py                <- Treat observations datasets such as Automatic Weather Stations (AWS)
│
│   ├── Depreciated                      <- Old code that I don't want to get rid off 
│
│   ├── Operators                        <- Process data to make predictions
│      ├── Helbig.py                     <- Functions to downscale wind fields according to Helbig et al. (2017)
│      ├── Micro_Met.py                  <- Functions to downscale wind fields according to MicroMet model from Liston and Elder (2006)
│      ├── Processing.py                 <- Functions to downscale wind fiels ccording to my method
│      ├── Rotation.py                   <- Functions to rotate images (e.g. topography maps) including numpy vectorized rotations
│      ├── topo_utils.py                 <- Functions to calcuate parameters on topography (e.g. Laplacian, tpi, sx, peak-valley elevation)
│      └── wind_utils.py                 <- Functions specific to wind calculations
│
│   ├── Utils                            <- Utility functions
│      ├── Decorators.py                 <- Some decorators
│      ├── GPU.py                        <- Some functions used when working with GPU
│      ├── prm.py                        <- Funcitons to treat input parameters
│      └── Utils.py                      <- Utility functions
│
│   ├── scripts        <- To be removed
│
│   └── test           <- test the code with pytest
│
├── pre_process        <- Pre-process data before training
│
├── train                  <- Train models
│   ├── Metrics            <- RMSE, bias etc
│   ├── Models             <- UNet, VCD
│   ├── Prm                <- Define parameters for the training
│   ├── Slurm              <- Commands to launch training on supercomputers
│   ├── Test               <- Evaluate training
│   ├── Type_of_training   <- Different ways to categorize data and then launch training
│   └── Utils              <- Utility functions
│
├── .gitignore         <- Files ignored during git version control
│
├── __init__.py        <- Python file to create a python module
│
└──  environment.yml    <- Description of the python environment to use. This file has to be updated.
```


## Code architecture

Where I do some pre-processing to organize data
> pre_process/

Training (performed on GPU)

> train/

Predictions on real topographies

> predict_real_topo/

