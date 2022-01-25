#!/usr/bin/env python
# coding: utf-8

# # Train a new model using a saved config file containing hyperparameters
# place in the same folder as train_network.py

# -*- coding: utf-8 -*-


import sys
import os
import json

def compute_training(path_model, path_trainingset, path_model_init = None):

    os.chdir(sys.path[0]) # Necessary to fix the directory we are working in
    path_configfile = os.path.join(path_model, 'config_network.json')
    with open(path_configfile, 'r') as fd:
        config_network = json.loads(fd.read())

    #if not os.path.exists(path_model): # Already created before. But can be useful if we want to create models with date of launch.
    #    os.makedirs(path_model)

    #with open(path_model + configname, 'w') as f:
    #    json.dump(config, f, indent=2)

    #with open(path_model + filename, 'r') as fd:
    #    config_network = json.loads(fd.read())

    # Training
    from AxonDeepSeg.train_network import train_model
    train_model(path_trainingset, path_model, config_network)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--path_model", required=True, help="")
    ap.add_argument("-t", "--path_trainingset",
                    default='/home/groups/bzuchero/axondeepseg/data/processed/cns/na_nyu_combined', help="")
    ap.add_argument("-i", "--path_model_init", required=False, default = None, help="")

    args = vars(ap.parse_args())
    path_model = str(args["path_model"])
    path_trainingset = str(args["path_trainingset"])
    path_model_init = str(args["path_model_init"])

    compute_training(path_model, path_trainingset, path_model_init)

if __name__ == '__main__':
    main()



