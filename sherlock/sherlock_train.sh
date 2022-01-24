#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1-0

while getopts 'm' flag; do
    case "${flag}" in
        d) model_dir =${OPTARG};;
    esac
done

echo 'This script is running on:'
hostname

source ~/.bash_profile
conda activate ads_venv
python ../AxonDeepSeg/train_from_config.py -m model_dir


