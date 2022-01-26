#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1-0

model_dir=''
trainingset_path=''

while getopts 'm:t:' flag; do
    case "${flag}" in
        m) model_dir=${OPTARG};;
        t) trainingset_path=${OPTARG};;
    esac
done

echo "model directory: " $model_dir
echo "training set: " $trainingset_path
echo 'This script is running on:'
hostname


source ~/.bashrc
conda activate ads_venv
python ../AxonDeepSeg/train_from_config.py -m $model_dir -t $trainingset_path

