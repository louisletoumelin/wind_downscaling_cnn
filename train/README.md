## How to start training

All parameters, hyperparameters, path to data etc are specified in the following file

```
prm/create_prm.py
```

To launch the training, use command

```
python train_models.py
```

If you want to launch several trainings in parallel on different GPUs, each training can be launched individually. Please, specify parameters, hyperparameters etc in the following file. 

```
python prm/create_prm_2.py
```

To launch the second training, use command

```
python experience_2.py
```

For GPU selection, see Slurm folder. I trained my model on a GPU cluster and had to execute .slurm files locates in Slurm/ to submit job, batches and get a GPU allocation. If you are working on your personal PC it is useless.