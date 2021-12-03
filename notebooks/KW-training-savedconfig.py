#!/usr/bin/env python
# coding: utf-8

# # Train a new model using a saved config file
# 
# #### Config files contain the hyperparameters for training the model

# In[1]:


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

# Input path of training images
path_training = Path('/home/groups/bzuchero/axondeepseg/data/processed/cns/na')
print("Training folder path:",path_training)

# Load save configuration file
model_name = 'CNS_TEM_2021-12-02_03-25-00'
path_model = os.path.join('..','models',model_name)
path_configfile = os.path.join(path_model,'config_network.json')
with open(path_configfile, 'r') as fd:
    config = json.loads(fd.read())

# reset the tensorflow graph for new testing
tf.reset_default_graph()

train_model(str(path_training), str(path_model), config)



