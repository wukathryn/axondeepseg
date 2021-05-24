# organize_domains.py
# This script moves the images of a specific genotype in one of the domains: VM, G/C, CST

import os
import shutil
import glob
from PIL import Image
from tqdm import tqdm 


if __name__ == "__main__":
    directory_name = '~/duke/histology/McGill/EM 2020-2021/SC P30'
    subdirectory_list = os.listdir(directory_name)

    for subdirectory in subdirectory_list:

        # split the directory 
        image_list = os.path.join(directory_name, subdirectory)
        directories_list = ['G_C', 'CST', 'VM']

        for directory in directories_list:
            if not os.path.isdir(os.path.join(directory_name, subdirectory, directory)):
                os.mkdir(os.path.join(directory_name, subdirectory, directory))
        
        # iterate over the images
        for image in tqdm(os.listdir(image_list)):

            if not os.path.isdir(os.path.join(directory_name, subdirectory, image)):
                if image.startswith('Gracilus') or image.startswith('Gracillus') or image.startswith('Cuneate'):
                    shutil.move(os.path.join(directory_name, subdirectory, image), os.path.join(directory_name, subdirectory, 'G_C', image))
                   
                if image.startswith('CST'):
                    shutil.move(os.path.join(directory_name, subdirectory, image), os.path.join(directory_name, subdirectory, 'CST', image))
           
                if image.startswith('VM'):
                    shutil.move(os.path.join(directory_name, subdirectory, image), os.path.join(directory_name, subdirectory, 'VM', image))

    print('Preprocessing done')

    
    