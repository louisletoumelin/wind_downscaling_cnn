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


## Code architecture

Where I do some pre-processing to organize data
> pre_process/

Training (performed on GPU)

> train/

Predictions on real topographies

> predict_real_topo/

