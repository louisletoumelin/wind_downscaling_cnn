## How to start training

I used to train the CNN on a GPU cluster. Each training was lauched on one GPU. Sometimes, I launched several trainings on several GPUs manually.

To laucnh first experience, I used

```
my_scipt.slurm
```

To laucnh second experience, I used

```
experience_2.slurm
```

The code is not developped to be launched on several GPUs on your local machine but could be easily adapted by selecting distinctively different GPU to launch training. The code should work as specified in the parent folder on your local machine with only one GPU.

Initially, horovod was used to paralellize computation on several GPUs (not to lauch several distinct training on distinct GPUs). I found not clear benefit with this method and do not use it anymore. However, it should still work.