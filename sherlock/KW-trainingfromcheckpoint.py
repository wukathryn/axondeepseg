#!/usr/bin/env python
# coding: utf-8

# # Train from an existing checkpoint
# 
# - note: only works if there is a model.hdf5 file in the model directory

import os
import json
import time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry

# Scientific package imports
import imageio
import numpy as np
import tensorflow as tf
from skimage import io
import matplotlib.pyplot as plt

# Utils import
from shutil import copy
import zipfile
from tqdm import tqdm
import cgi
import tempfile

# AxonDeepSeg imports
try:
    from AxonDeepSeg.ads_utils import download_data
except ModuleNotFoundError:
    # Change cwd to project main folder 
    os.chdir("..")
    try :
        from AxonDeepSeg.ads_utils import download_data
    except:
        raise
except:
    raise
# If no exceptions were raised import all folders        
from AxonDeepSeg.config_tools import validate_config
from AxonDeepSeg.train_network import train_model
from AxonDeepSeg.apply_model import axon_segmentation
import AxonDeepSeg.ads_utils as ads 
from config import axonmyelin_suffix

# reset the tensorflow graph for new training
tf.reset_default_graph()

# Before running this notebook, please make sure you have split, patched, and organized your data in the correct way by running the [guide_dataset_building.ipynb](guide_dataset_building.ipynb) tutorial.

# In[2]:

#PNS training images
path_training = '/home/groups/bzuchero/axondeepseg/data/processed/pns'

#CNS training images
#path_training = Path("./training")


# ### Training Image Description:
# 
# Used 9 labelled TEM images from Nick
# (1500x magnification)
# 
# And images from NYU from White Matter Microscopy Database
# 
# 

# #### 1.2. Define the U-Net architecture and hyper-parameters
# 
# Here we defined the network and training parameters (i.e. hyperparameters). We use a lighter network than the ones used in the original [AxonDeepSeg article](https://www.nature.com/articles/s41598-018-22181-4), because they require large GPU memory (>12GB). The network below runs on an NVIDIA TitanX in ~2h. Note that the architecture below might not produce satisfactory results on your data so you likely need to play around with the architecture and hyperparameters.
# 
# **Important:** The pixel size is not defined at the training step. During inference however, the parameter `-t {SEM,TEM,OM}` sets the resampling resolution to 0.1µm or 0.01µm depending on the model (i.e., implying the pixel size of the training data should be around 0.1µm for SEM and OM, and 0.01µm for TEM). This is definitely a limitation of the current version of AxonDeepSeg, which we are planning to solve at some point (for more info, see [Issue #152](https://github.com/neuropoly/axondeepseg/issues/152)). 
# 
# **Note about data augmentation:**
# For each type of data augmentation, the order needs to be specified if you decide to apply more than one transformation sequentially. For instance, setting `da-0-shifting-activate` to `True` means that the shifting is the first transformation that will be applied to the sample(s) during training. The default ranges of transformations are:
# - **Shifing**: Random horizontal and vertical shifting between 0 and 10% of the patch size, sampled from a uniform distribution.
# - **Rotation**: Random rotation, angle between 5 and 89 degrees, sampled from a uniform distribution.
# - **Rescaling**: Random rescaling of a randomly sampled factor between 1/1.2 and 1.2.
# - **Flipping**: Random fipping: vertical fipping or horizontal fipping.
# - **Elastic deformation**: Random elastic deformation with uniformly sampled deformation coefficient α=[1–8] and fixed standard deviation σ=4.
# 
# You can find more information about the range of transformations applied to the patches for each data augmentation technique in the file [data_augmentation.py](https://github.com/neuropoly/axondeepseg/blob/master/AxonDeepSeg/data_management/data_augmentation.py).

# In[3]:


# Example of network configuration for TEM data (small network trainable on a Titan X GPU card)
config = {
    
# General parameters:    
  "n_classes": 3,  # Number of classes. For this application, the number of classes should be set to **3** (i.e. axon pixel, myelin pixel, or background pixel).
  "thresholds": [0, 0.2, 0.8],  # Thresholds for the 3-class classification problem. Do not modify.  
  "trainingset_patchsize": 512,  # Patch size of the training set in pixels (note that the patches have the same size in both dimensions).  
  "trainingset": "PNS_TEM",  # Name of the training set.
  "batch_size": 8,  # Batch size, i.e. the number of training patches used in one iteration of the training. Note that a larger batch size will take more memory.
  "epochs":2000,
  "checkpoint_period": 5, # Number of epoch after which the model checkpoint is saved.
  "checkpoint": "loss", # Checkpoint to use to resume training. Option: "loss", "accuracy" or None.

# Network architecture parameters:     
  "depth": 4,  # Depth of the network (i.e. number of blocks of the U-net).
  "convolution_per_layer": [2, 2, 2, 2],  # Number of convolution layers used at each block.
  "size_of_convolutions_per_layer": [[5, 5], [3, 3], [3, 3], [3, 3]],  # Kernel size of each convolution layer of the network.
  "features_per_convolution": [[[1, 16], [16, 16]], [[16, 32], [32, 32]], [[32, 64], [64, 64]], [[64, 128], [128, 128]]],  # Number of features of each convolution layer.
  "downsampling": "convolution",  # Type of downsampling to use in the downsampling layers of the network. Option "maxpooling" for standard max pooling layer or option "convolution" for learned convolutional downsampling.
  "dropout": 0.75,  # Dropout to use for the training. Note: In TensorFlow, the keep probability is used instead. For instance, setting this param. to 0.75 means that 75% of the neurons of the network will be kept (i.e. dropout of 25%).
     
# Learning rate parameters:    
  "learning_rate": 0.001,  # Learning rate to use in the training.  
  "learning_rate_decay_activate": True,  # Set to "True" to use a decay on the learning rate.  
  "learning_rate_decay_period": 24000,  # Period of the learning rate decay, expressed in number of images (samples) seen.
  "learning_rate_decay_type": "polynomial",  # Type of decay to use. An exponential decay will be used by default unless this param. is set to "polynomial" (to use a polynomial decay).
  "learning_rate_decay_rate": 0.9,  # Rate of the decay to use for the exponential decay. This only applies when the user does not set the decay type to "polynomial".
    
# Batch normalization parameters:     
  "batch_norm_activate": True,  # Set to "True" to use batch normalization during the training.
  "batch_norm_decay_decay_activate": True,  # Set to "True" to activate an exponential decay for the batch normalization step of the training.  
  "batch_norm_decay_starting_decay": 0.7,  # The starting decay value for the batch normalization. 
  "batch_norm_decay_ending_decay": 0.9,  # The ending decay value for the batch normalization.
  "batch_norm_decay_decay_period": 1600,  # Period of the batch normalization decay, expressed in number of images (samples) seen.
        
# Weighted cost parameters:    
  "weighted_cost-activate": True,  # Set to "True" to use weights based on the class in the cost function for the training.
  "weighted_cost-balanced_activate": True,  # Set to "True" to use weights in the cost function to correct class imbalance. 
  "weighted_cost-balanced_weights": [1.1, 1, 1.3],  # Values of the weights for the class imbalance. Typically, larger weights are assigned to classes with less pixels to add more penalty in the cost function when there is a misclassification. Order of the classes in the weights list: background, myelin, axon.
  "weighted_cost-boundaries_sigma": 2,  # Set to "True" to add weights to the boundaries (e.g. penalize more when misclassification happens in the axon-myelin interface).
  "weighted_cost-boundaries_activate": False,  # Value to control the distribution of the boundary weights (if activated). 
    
# Data augmentation parameters:
  "da-type": "all",  # Type of data augmentation procedure. Option "all" applies all selected data augmentation transformations sequentially, while option "random" only applies one of the selected transformations (randomly) to the sample(s). List of available data augmentation transformations: 'random_rotation', 'noise_addition', 'elastic', 'shifting', 'rescaling' and 'flipping'. 
  "da-0-shifting-activate": True, 
  "da-1-rescaling-activate": True,
  "da-2-random_rotation-activate": True,
  "da-3-elastic-activate": True, 
  "da-4-flipping-activate": True, 
  "da-6-reflection_border-activate": True
}


# #### 1.3. Define training path and save configuration parameters
# 
# Here we define the path where the new model will be saved. It is useful to add date+time in path definition in case multiple training are launched (to avoid conflicts).
# 
# The network configuration parameters defined at 1.2. are saved into a .json file in the model folder. This .json file keeps tract of the network and model parameters in a structured way.

# In[4]:


# Define path to where the trained model will be saved
#tissue = "CNS"
#dir_name = Path(tissue + '_' + config["trainingset"] + '_' + time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S"))
#path_model = "../models" / dir_name

#print("This training session's model will be saved in the folder: " + str(path_model.resolve().absolute()))

# Create directory
#if not os.path.exists(path_model):
#    os.makedirs(path_model)

dir_name = Path('PNS_TEM_2021-12-01_01-23-00')
path_model = path_model = "../models" / dir_name


# In[5]:


# Define file name of network configuration
file_config = 'config_network.json'

# Load/Write config file (depending if it already exists or not)
fname_config = os.path.join(path_model, file_config)
if os.path.exists(fname_config):
    with open(fname_config, 'r') as fd:
        config_network = json.loads(fd.read())
else:
    with open(fname_config, 'w') as f:
        json.dump(config, f, indent=2)
    with open(fname_config, 'r') as fd:
        config_network = json.loads(fd.read())


# #### 1.4. Launch the training procedure
# 
# The training can be launched by calling the *'train_model'* function. After each epoch, the function will display the loss and accuracy of the model. The model checkpoints will be saved according to the "checkpoint_period" parameter in "config".

# In[6]:


# reset the tensorflow graph for new testing
tf.reset_default_graph()

print(str(path_training))
print(str(path_model))

train_model(str(path_training), str(path_model), config)
# Note: For multi-OS compatibility of this notebook, paths were defined as Path objects from the pathlib module.
# They need to be converted into strings prior to be given as arguments to train_model(), as some old-style string
# concatenation (e.g. "+") are still used in it.
# In your own application, simply defining paths with correct syntax for your OS as strings instead of Path
# objects would be sufficient.

