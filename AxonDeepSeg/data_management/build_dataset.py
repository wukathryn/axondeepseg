#!/usr/bin/env python
# coding: utf-8

# ## Dataset creation notebook
# 
# This notebook shows how to build a dataset for the training of a new model in AxonDeepSeg. It covers the following steps:
# 
# * How to structure the raw data.
# * How to define the parameters of the patch extraction and divide the raw labelled dataset into patches.
# * How to generate the training dataset of patches by combining all raw data patches.
# 
# 

# ### STEP 0: IMPORTS.

# In[19]:


from AxonDeepSeg.data_management.dataset_building import split_data, raw_img_to_patches, patched_to_dataset
from AxonDeepSeg.ads_utils import download_data
import os, shutil
from pathlib import Path

#modules for visualization at the end
import cv2
from matplotlib import pyplot as plt

import PIL
from PIL import Image, ImageOps
PIL.Image.MAX_IMAGE_PIXELS = None


# ### STEP 1: GENERATE THE DATASET.
# 
# ### Suggested procedure for training/validation split of the dataset:
# 
# * **Example use case:** we have 6 labelled samples in our dataset. To respect the split convention (between 10-30% of samples kept for validation), we can keep 5 samples for the training and the remaining one for the validation. 
# 
# ---
# ##### The folder structure *before* the training/validation split:
# 
# * ***folder_of_your_raw_data***
# 
#      * **sample1**
#           * *image.png*
#           * *mask.png*
#           * *pixel_size_in_micrometer.txt*
#      * **sample2**
#           * *image.png*
#           * *mask.png*
#           * *pixel_size_in_micrometer.txt*
#             
#             ...
#             
#      * **sample6**
#           * *image.png*
#           * *mask.png*
#           * *pixel_size_in_micrometer.txt*
#             
# ---
# #### The folder structure *after* the training/validation split:
# 
# * ***folder_of_your_raw_data***
# 
#     * **Train**
#         * **sample1**
#             * *image.png*
#             * *mask.png*
#             * *pixel_size_in_micrometer.txt*
#         * **sample2**
#             * *image.png*
#             * *mask.png*
#             * *pixel_size_in_micrometer.txt*
#             
#             ...
#             
#         * **sample5**
#             * *image.png*
#             * *mask.png*
#             * *pixel_size_in_micrometer.txt*
#             
#     * **Validation**
#         * **sample6**
#             * *image.png*
#             * *mask.png*
#             * *pixel_size_in_micrometer.txt*
# ---         

# For this notebook, we'll download a sample dataset hosted on OSF.io that's in the correct raw data folder structure , and use our AxonDeepSeg tools to split the dataset. In the next sections, this notebook will resample the images into patches and group them together in the correct directory structure needed to run the [training guideline notebook](training_guideline.ipynb).

# In[20]:


#import path of training images
train_data_orig = "/Users/kwu2/Library/CloudStorage/Box-Box/Box Files/Lab_files/TEM data/6_16_21_foranalysis/p5-4-KW20210318/p5-4-KW20210318_PNGs copy/Training/Training"

#copy data to new folder inside of $GROUP_HOME/axondeepseg/notebooks/training
#note that data in this folder will be removed after processing
get_ipython().system('cp -a "$train_data_orig" ./train_data')
train_data_path = Path("./train_data")


# In[21]:


# Define the directory names for the dataset building folders

data_split_path = Path("./split")
data_patched_path = Path("./patched")
data_training_path = Path("./training")

# If dataset building folders already exist, remove them.
if data_split_path.exists():
    shutil.rmtree(data_split_path)
if data_patched_path.exists():
    shutil.rmtree(data_patched_path)
if data_training_path.exists():
    shutil.rmtree(data_training_path)


# In[22]:


# Set seed (changing value will change how the dataset is split)
seed = 2019

# Set dataset split fraction [Train, Validation]
split = [0.8, 0.2]

# Split data into training and validation datasets 
split_data(train_data_path, data_split_path, seed=seed, split=split)


# #### 1.1. Define the parameters of the patch extraction.
# 
# * **path_raw_data**: Path of the folder that contains the raw data. Each labelled sample of the dataset should be in a different subfolder. For each sample (and subfolder), the expected files are the following:
#     * *"image.png"*: The microscopy sample image (uint8 format).
#     * *"mask.png"*: The microscopy sample image (uint8 format).
#     * *"pixel_size_in_micrometer.txt"*: A one-line text file with the value of the pixel size of the sample. For instance, if the pixel size of the sample is 0.02um, the value in the text file should be **"0.02"**.
#     
# * **path_patched_data**: Path of the folder that will contain the raw data divided into patches. Each sample (i.e. subfolder) of the raw dataset will be divided into patches and saved in this folder. For instance, if a sample of the original data is divided into 10 patches, the corresponding folder in the **path_patched_dataset** will contain 10 image and mask patches, named **image_0.png** to **image_9.png** and **mask_0.png** to **mask_9.png**, respectively. 
# 
# * **patch_size**: The size of the patches in pixels. For instance, a patch size of **128** means that each generated patch will be 128x128 pixels.
# 
# * **general_pixel_size**: The pixel size (i.e. resolution) of the generated patches in micrometers. The pixel size will be the same for all generated patches. If the selected pixel size is different from the native pixel sizes of the samples, downsampling or upsampling will be performed. Note that the pixel size should be chosen by taking into account the modality of the dataset and the patch size.  

# In[23]:


# Define the paths for the training samples
path_raw_data_train = data_split_path / 'Train'
path_patched_data_train = data_patched_path / 'Train'

# Define the paths for the validation samples
path_raw_data_validation = data_split_path / 'Validation'
path_patched_data_validation = data_patched_path / 'Validation'

patch_size = 512

#default was originally 0.1
general_pixel_size = 0.01


# In[31]:


with open('training_data_list.txt', 'w') as f:
    f.write("Training files:" + '\n')
    for d in os.scandir(path_raw_data_train):
        f.write(d.name + '\n')
    f.write('\n' + "Validation files:" + '\n')
    for d in os.scandir(path_raw_data_validation):
        f.write(d.name + '\n')
    f.write('\n' + "General pixel size: " + str(general_pixel_size))
    f.close()


# #### 1.2. Divide the training/validation samples into patches.
# 
# In the **path_patched_data** folder defined above, the original samples are going to be split into patches of same size. For instance, the sample 1 of the training set of the example use case above will be split into *n* patches and its corresponding subfolder in the **path_patched_data** folder will have the following structure:
# 
# ---
# * ***folder_of_your_patched_data***
# 
#     * **Train**
#         * **sample1**
#             * *image_0.png*
#             * *mask_0.png*
#             * *image_1.png* 
#             * *mask_1.png*
#             * *image_2.png*
#             * *mask_2.png*
#             
#             ...
#             
#             * *image_n.png* 
#             * *mask_n.png*        
# ---
# 
# 
# * Run the *raw_img_to_patches* function on both *Train* and *Validation* subfolders to split the data into patches. Note the input param. **thresh_indices** is a list of the threshold values to use in order to generate the classes of the training masks. The default value is [0, 0.2, 0.8], meaning that the mask labels (background=0, myelin=0.5, axon=1) will be split into our 3 classes.

# In[32]:


# Split the *Train* dataset into patches
raw_img_to_patches(path_raw_data_train, path_patched_data_train, thresh_indices = [0, 0.2, 0.8], patch_size=patch_size, resampling_resolution=general_pixel_size)

# Split the *Validation* dataset into patches
raw_img_to_patches(path_raw_data_validation, path_patched_data_validation, thresh_indices = [0, 0.2, 0.8], patch_size=patch_size, resampling_resolution=general_pixel_size)


# #### 1.3. Regroup all the divided patches in the same training/validation folder.
# 
# Finally, to build the dataset folder that is going to be used for the training, all patches obtained from the different samples are regrouped into the same folder and renamed. The final training and validation folders will have the following structure (*m* is the total number of training patches and *p* is the total number of validation patches):
# 
# ---
# * ***folder_of_your_final_patched_data***
# 
#     * **Train**
#          * *image_0.png*
#          * *mask_0.png*
#          * *image_1.png* 
#          * *mask_1.png*
#          * *image_2.png*
#          * *mask_2.png*
#             
#          ...
#            
#          * *image_m.png* 
#          * *mask_m.png*   
#          
#     * **Validation**
#          * *image_0.png*
#          * *mask_0.png*
#          * *image_1.png* 
#          * *mask_1.png*
#          * *image_2.png*
#          * *mask_2.png*
#             
#          ...
#            
#          * *image_p.png* 
#          * *mask_p.png*         
#          
# ---
# 
# Note that we define a random seed in the input of the *patched_to_dataset* function in order to reproduce the exact same images each time we run the function. This is done to enable the generation of the same training and validation sets (for reproducibility). Also note that the **type_** input argument of the function can be set to **"unique"** or **"mixed"** to specify if the generated dataset comes from the same modality, or contains more than one modality.

# In[33]:


# Path of the final training dataset
path_final_dataset_train = data_training_path / 'Train'

# Path of the final validation dataset
path_final_dataset_validation = data_training_path / 'Validation'


# In[34]:


# Regroup all training patches
patched_to_dataset(path_patched_data_train, path_final_dataset_train, type_='unique', random_seed=2017)

# Regroup all validation patches
patched_to_dataset(path_patched_data_validation, path_final_dataset_validation, type_='unique', random_seed=2017)


# In[35]:


# Remove intermediate dataset building folders
if train_data_path.exists():
    shutil.rmtree(train_data_path)
if data_split_path.exists():
    shutil.rmtree(data_split_path)
if data_patched_path.exists():
    shutil.rmtree(data_patched_path)


# In[36]:


#Optional: visualize one of the image patches

from IPython.display import Image
Image(filename='./training/Train/image_0.png') 


# ## Next step
# 
# Now that you've resampled, patched, and organized your data correctly, the next step is to run the [training guideline notebook](training_guideline.ipynb).

# In[ ]:




