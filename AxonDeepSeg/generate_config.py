# -*- coding: utf-8 -*-
#This file generates model path folders and config files within them

import os
import json
from pathlib import Path

def make_model_directory():

    # Define path to where the trained model will be saved
    tissue = "CNS"
    change_from_default_config = "testconfig"
    dir_name = Path(tissue + "_" + change_from_default_config)
    path_model = "../models" / dir_name

    print("Model path: " + str(path_model.resolve().absolute()))

    # Create directory for model
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    return path_model

def make_config_file(path_model):
    config = {
        # General parameters:
        "n_classes": 3,
        # Number of classes. For this application, the number of classes should be set to **3** (i.e. axon pixel, myelin pixel, or background pixel).
        "thresholds": [0, 0.2, 0.8],  # Thresholds for the 3-class classification problem. Do not modify.
        "trainingset_patchsize": 512,
        # Patch size of the training set in pixels (note that the patches have the same size in both dimensions).
        "trainingset": "TEM",  # Name of the training set.
        "batch_size": 8,
        # Batch size, i.e. the number of training patches used in one iteration of the training. Note that a larger batch size will take more memory.
        "epochs": 1000,
        "checkpoint_period": 5,  # Number of epoch after which the model checkpoint is saved.
        "checkpoint": None,  # Checkpoint to use to resume training. Option: "loss", "accuracy" or None.

        # Network architecture parameters:
        "depth": 4,  # Depth of the network (i.e. number of blocks of the U-net).
        "convolution_per_layer": [2, 2, 2, 2],  # Number of convolution layers used at each block.
        "size_of_convolutions_per_layer": [[5, 5], [3, 3], [3, 3], [3, 3]],
        # Kernel size of each convolution layer of the network.
        "features_per_convolution": [[[1, 16], [16, 16]], [[16, 32], [32, 32]], [[32, 64], [64, 64]],
                                     [[64, 128], [128, 128]]],  # Number of features of each convolution layer.
        "downsampling": "convolution",
        # Type of downsampling to use in the downsampling layers of the network. Option "maxpooling" for standard max pooling layer or option "convolution" for learned convolutional downsampling.
        "dropout": 0.75,
        # Dropout to use for the training. Note: In TensorFlow, the keep probability is used instead. For instance, setting this param. to 0.75 means that 75% of the neurons of the network will be kept (i.e. dropout of 25%).

        # Learning rate parameters:
        "learning_rate": 0.001,  # Learning rate to use in the training.
        "learning_rate_decay_activate": True,  # Set to "True" to use a decay on the learning rate.
        "learning_rate_decay_period": 24000,
        # Period of the learning rate decay, expressed in number of images (samples) seen.
        "learning_rate_decay_type": "polynomial",
        # Type of decay to use. An exponential decay will be used by default unless this param. is set to "polynomial" (to use a polynomial decay).
        "learning_rate_decay_rate": 0.99,
        # Rate of the decay to use for the exponential decay. This only applies when the user does not set the decay type to "polynomial".

        # Batch normalization parameters:
        "batch_norm_activate": True,  # Set to "True" to use batch normalization during the training.
        "batch_norm_decay_decay_activate": True,
        # Set to "True" to activate an exponential decay for the batch normalization step of the training.
        "batch_norm_decay_starting_decay": 0.7,  # The starting decay value for the batch normalization.
        "batch_norm_decay_ending_decay": 0.9,  # The ending decay value for the batch normalization.
        "batch_norm_decay_decay_period": 16000,
        # Period of the batch normalization decay, expressed in number of images (samples) seen.

        # Weighted cost parameters:
        "weighted_cost-activate": True,
        # Set to "True" to use weights based on the class in the cost function for the training.
        "weighted_cost-balanced_activate": True,
        # Set to "True" to use weights in the cost function to correct class imbalance.
        "weighted_cost-balanced_weights": [1.1, 1, 1.3],
        # Values of the weights for the class imbalance. Typically, larger weights are assigned to classes with less pixels to add more penalty in the cost function when there is a misclassification. Order of the classes in the weights list: background, myelin, axon.
        "weighted_cost-boundaries_sigma": 2,
        # Set to "True" to add weights to the boundaries (e.g. penalize more when misclassification happens in the axon-myelin interface).
        "weighted_cost-boundaries_activate": False,
        # Value to control the distribution of the boundary weights (if activated).

        # Data augmentation parameters:
        "da-type": "all",
        # Type of data augmentation procedure. Option "all" applies all selected data augmentation transformations sequentially, while option "random" only applies one of the selected transformations (randomly) to the sample(s). List of available data augmentation transformations: 'random_rotation', 'noise_addition', 'elastic', 'shifting', 'rescaling' and 'flipping'.
        "da-0-shifting-activate": True,
        "da-1-rescaling-activate": False,
        "da-2-random_rotation-activate": False,
        "da-3-elastic-activate": True,
        "da-4-flipping-activate": True,
        "da-6-reflection_border-activate": False
    }


    # Define file name of network configuration
    file_config = 'config_network.json'

    # Load/Write config file (depending if it already exists or not)
    fname_config = os.path.join(path_model, file_config)
    with open(fname_config, 'w') as f:
        json.dump(config, f, indent=2)

def main():
    path_model = make_model_directory()
    make_config_file(path_model)
    return path_model

if __name__ == '__main__':
    main()