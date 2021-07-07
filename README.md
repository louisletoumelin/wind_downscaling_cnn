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
├── LICENSE            <- To be created

├── README.md          <- File describing the github repository. You are reading it right now.

├── downscale          <- Apply downscaling predictions 
│   ├── Analysis       <- Analyse and visualize predictions
│   ├── Data_family    <- Process data according to its nature (model data, DEM, observation...)
│   ├── Depreciated    <- Old code that I don't want to get rid off now
│   ├── Operators      <- Process data to make predictions
│   ├── Utils          <- Utility functions
│   ├── scripts        <- To be removed
│   └── test           <- test the code with pytest
│
├── pre_process        <- Pre-process data before training
│
├── train          <- Apply downscaling predictions 
│   ├── Metrics            <- RMSE, bias etc
│   ├── Models             <- UNet, VCD
│   ├── Prm                <- Define parameters for the training
│   ├── Slurm              <- sh and slurm commands to laucnh on GPU HPC
│   ├── Test               <- Evaluate training
│   ├── Type_of_training   <- Train on folds, class, degree, xi, all data
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

