#!/bin/bash

#Shell script to modify the rights on the created files and repositories during the DL experiences (from root to user)
#/!\#Execute this script inside the container ONLY IF YOU DO NOT USE HOROVOD

####Parameters to modify####
HOME_DIR="/home/<my_group>/<my_user>/<my_code_directory>/"     #home directory
DATA_DIR="/scratch/<my_group>/<my_user>/<my_data_directory>/"                  #data directory

uid=XXXXX  #user id : to know it, execute this command as user : id -u 
gid=XXXXX  #group id :  to know it, execute this command as user : id -g 
############

echo chown -R ${uid}:${gid} ${HOME_DIR} ${DATA_DIR}

chown -R ${uid}:${gid} ${HOME_DIR} ${DATA_DIR}
