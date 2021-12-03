#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 1-0

echo 'This script is running on:'
hostname

source ~/.bash_profile
conda activate ads_venv
python ./KW-training-savedconfig.py

