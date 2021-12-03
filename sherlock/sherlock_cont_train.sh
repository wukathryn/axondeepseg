#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 0-6

echo 'This script is running on:'
hostname

source ~/.bash_profile
conda activate ads_venv
python ./KW-trainingfromcheckpoint.py
