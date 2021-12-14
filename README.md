# Wind fields downscaling in complex terrain using convolutional neural networks

Work in progress

Ongoing study - PhD Louis Le Toumelin - louis.letoumelin@gmail.com

**Supervisors:** Isabelle Gouttevin, Fatima Karbou

*Centre Etudes de la Neige (Snow Research Center) - CNRM - CNRS  - Météo-France*


## How it works

We try to replicate the behavior of te ARPS model by using a CNN.
The process is divided in two distinct steps: first learinng from the model outputs, then generating real case scenario.

**Learning**

*Input* : Synthetic topographies (7260 gridded outputs containing topographies, each map having a size 79x69)

**Output** : ARPS wind on synthetic topographies. Wind comes from the West at 3 m/s. The wind vector is three dimensional (hence 7360 outputs are generated, and each output is a 3D map)

*Prediction*

**Input**:  
- Numerical Weather Prediction (ex: AROME model) wind field (speed + direction)
- Topography (ex: Digital Elevation Model from IGN 30m)

**Output**:  
- Downscaled wind fields on real topographies (U, V, W)


## Structure

```
├── LICENSE                               <- To be created
│
├── README.md                             <- File describing the github repository. You are reading it right now.
│
├── downscale_                            
│   ├── downscale                         <- downscale module
│   │  ├── data_source                    <- Process different types of data (NWP, DEM, observations...)
│   │  ├── eval                           <- Evaluate predictions (process bias, RMSE...)
│   │  ├── operators                      <- Downscaling strategy (Helbig, Devine, MicroMet...)
│   │  ├── test                           <- Test the code
│   │  ├── utils                          <- Utils function
│   │  └── visu                           <- Visualize plots
│   │  ├── __init__.py                    <- To create a module
│   ├── pipeline                          <- Scripts using the downscale module
│
│
├── pre_process                           <- Pre-process data before training
│
├── train                                 <- Train models
│   ├── Metrics                           <- RMSE, bias etc
│   ├── Models                            <- UNet, VCD
│   ├── Prm                               <- Define parameters for the training
│   ├── Slurm                             <- Commands to launch training on supercomputers
│   ├── Test                              <- Evaluate training
│   ├── Type_of_training                  <- Different ways to categorize data and then launch training
│   └── Utils                             <- Utility functions
│
├── WindNinja_learning                    <- Launch WindNinja simulation with python
│
├── .gitignore                            <- Files ignored during git version control
│
├── __init__.py                           <- To create a module
│
└──  cnn4.yml                             <- Description of the python environment to use. This file has to be updated.
```


## Code architecture

Where I do some pre-processing to organize data
> pre_process/

Training (performed on GPU)

> train/

Predictions on real topographies

> downscale_/

