# batch_morphometrics.py

# A simple script to generate batches of morphmetrics

import os
from AxonDeepSeg.morphometrics import launch_morphometrics_computation
from config import axonmyelin_suffix, axon_suffix, myelin_suffix
from tqdm import tqdm 


if __name__ == '__main__':
    directory_name = '/home/GRAMES.POLYMTL.CA/p112473/duke/projects/axondeepseg/20210415_peterson/20210516_segmentations/SC P30' # Name of the directory for which you want to do generates

    list_genotypes = os.listdir(directory_name) # list of genotypes
    
    for genotype in tqdm(list_genotypes): # iterate across all the genotypes

        if os.path.isdir(os.path.join(directory_name, genotype)):
            domain_list = os.listdir(os.path.join(directory_name, genotype))

            for domain in domain_list: # iterate over the domains
                if os.path.isdir(os.path.join(directory_name, genotype, domain)):
                    list_images =  os.listdir(os.path.join(directory_name, genotype, domain))

                    for image in list_images: # loop over the images in a particular domain
                        if not image.endswith(axonmyelin_suffix.name) or not image.endswith(axon_suffix.name) or image.endswith(myelin_suffix.name): 
                            image_name = image[:-4] # get the image name only 

                            if (image_name + axonmyelin_suffix.name) in list_images:

                                morphometrics_file = image_name + "_axon_morphometrics.csv"
                                launch_morphometrics_computation.main(["-i", os.path.join(directory_name, genotype, domain, image), "-f", morphometrics_file])

    print("********** Batch Morphometrics computed ********************")