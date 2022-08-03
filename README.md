# Emulating an atmospheric model with deep learning to downscale wind fields in complex terrain

![image](/images/example_downscaling.jpg)

Work in progress

Ongoing study - PhD Louis Le Toumelin - louis.letoumelin@gmail.com

**Supervisors:** Isabelle Gouttevin, Fatima Karbou

*Centre Etudes de la Neige (Snow Research Center) - CNRM - CNRS - Météo-France*

## Method

![image](/images/SchemeDevine.png)


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
│   ├── pipeline                          <- Scripts using the downscale module: good starting point
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
├── WindNinja_learning                    <- Launch WindNinja simulation with python (ongoing project)
│
├── .gitignore                            <- Files ignored during git version control
│
├── __init__.py                           <- To create a module
│
├── model                                 <- Tensorflow model
│
└──  devine_cnn.yml                       <- An example of working conda environment for this project
```


## Code architecture

Where I do some pre-processing to organize data
> pre_process/

Training (performed on GPU)

> train/

Predictions on real topographies + result analysis

> downscale_/

