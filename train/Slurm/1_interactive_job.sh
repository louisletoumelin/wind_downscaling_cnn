#!/bin/bash

#Shell script to run a slurm interactive session 

####Parameters to modify####
JOB_NAME="my_interactive_job"   #job name
NB_GPU="2"                      #number of GPU 
TIME="01:00:00"               #maximum time of the job (format 'DAY-HH:MM:SS') -> depends on the limit time of the partition 
###############

echo run a slurm interactive session 
echo srun --job-name=${JOB_NAME} --partition=interactive --gres=gpu:v100:${NB_GPU} --time=${TIME} --ntasks-per-node=1 --pty bash

#run the slurm interactive session and you go directly on the allocated node 

srun --job-name=${JOB_NAME} --partition=interactive --gres=gpu:v100:${NB_GPU} --time=${TIME} --ntasks-per-node=1 --pty bash


