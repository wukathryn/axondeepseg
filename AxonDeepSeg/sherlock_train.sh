#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1-0

model_dir=''
while getopts 'm:' flag; do
    case "${flag}" in
        m) model_dir=${OPTARG};;
    esac
done

echo 'This script is running on:'
hostname

echo $model_dir

source ~/.bash_profile
conda activate ads_venv
python ./train_from_config.py -m $model_dir


