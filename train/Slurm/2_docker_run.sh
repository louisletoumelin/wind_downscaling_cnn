#!/bin/bash

#Shell script to run a docker container in interactive mode 

####Parameters to modify####
HOME_DIR="/home/<my_group>/<my_user>/<my_code_directory>/"      #home directory to mount into the container 
DATA_DIR="/scratch/<my_group>/<my_user>/<my_data_directory>/"                   #data directory to mount into the container 
CONTAINER_NAME="horovod:env_DLv2"                                 #container name
###########################

# Define Docker options
DOCKER_MOUNTS_OPTS="-v ${HOME_DIR}:${HOME_DIR} -v ${DATA_DIR}:${DATA_DIR}"  #mount the different directories
MLX_OPTS="--network=host --cap-add=IPC_LOCK --device=/dev/infiniband"       #options about the infiniband network
GPU_OPTS='--gpus '\"device=${SLURM_STEP_GPUS}\"''                           #allocated GPU(s) 

echo 'allocated GPU(s):' "${SLURM_STEP_GPUS}"

echo docker run -it ${GPU_OPTS} ${MLX_OPTS} ${DOCKER_MOUNTS_OPTS} ${CONTAINER_NAME}

docker run -it ${GPU_OPTS} ${MLX_OPTS} ${DOCKER_MOUNTS_OPTS} ${CONTAINER_NAME}



