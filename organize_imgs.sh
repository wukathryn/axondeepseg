#!/bin/bash

#This script takes PNG image files and organizes them in the structure recommended for axondeepseg training:
# with a directory containing the file name, the image labelled image.png, and a text file with the
# pixels in micrometer information

#Traverse directory
#for each PNG file
#   makedir with that name
#    rename image to "image.png"
#    move image to that folder
#     add file pixel_size_in_micrometer.txt

while getopts 'd:' flag; do
    case "${flag}" in
        d) directory=${OPTARG};;
    esac
done

for file in $directory/*.png;
do
    newdir="${file%.*}"; echo newdir: $newdir;
    mkdir $newdir;
    mv "$file" "${newdir}/image.png";

    #get magnification from an image labelled in the following format: condition_mag_number (ex: p10-1_3000x_0010.png)
    mag="${newdir%x*}";
    mag="${mag##*_}";

  echo $mag > "$newdir/pixel_size_in_micrometer.txt";
done

