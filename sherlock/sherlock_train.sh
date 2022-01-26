#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1-0

model_dir=''
trainingset=''
while getopts 'm:t' flag; do
    case "${flag}" in
        m) model_dir=${OPTARG};;
        t) trainingset=${OPTARG};;
    esac
done

echo "model directory: " + $model_dir
echo "training set: " + $trainingset
echo 'This script is running on:'
hostname


source ~/.bashrc
conda activate ads_venv
python ../AxonDeepSeg/train_from_config.py -m $model_dir -t $trainingset

