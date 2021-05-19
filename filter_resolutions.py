# filter_resolutions.py
# This script filter unwanted resolutions of the images. We're only interested in Gracilus x4800 or Cuneate x4800, VM x2900 and CST x9300

import os
import shutil

def filter_resolution(directory):
    list_images = os.listdir(directory) # list of image names

    print('The number of images before were : ', len(list_images))

    image_resolution_keep = ['Gracilus x4800', 'CST x9300', 'VM x2900', 'Gracillus x4800', 'Cuneate x4800'] # image resolution to keep
    
    for image in list_images: # iterating over all the images name in the directory
        if image_resolution_keep[0] not in image:
            if image_resolution_keep[1] not in image:
                if image_resolution_keep[2] not in image:
                    if image_resolution_keep[3] not in image:
                        if image_resolution_keep[4] not in image:
                            os.unlink(os.path.join(directory, image)) # remove the images of  
    
    print('The unnecessary resolution images are removed')
    print('The number of images after filtering were', len(os.listdir(directory)))

if __name__ == "__main__":
    directory_name = '/home/GRAMES.POLYMTL.CA/p112473/p112473/axondeepseg/SC P30_lastone'
    subdirectory_list = os.listdir(directory_name)



    for subdirectory in subdirectory_list:
        print('The directory is ', subdirectory)
        filter_resolution(os.path.join(directory_name, subdirectory))
        print("\n\n")
