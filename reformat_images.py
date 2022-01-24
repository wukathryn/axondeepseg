#!/usr/bin/python
import os
import os.path
#from skimage.transform import rescale
from PIL import Image
import argparse


def scale_images(file_path, newsize):
    if file_path.lower().endswith(('.jpg', '.jpeg', '.tif', '.bmp', '.png')):
        img = Image.open(file_path)
        img = img.resize(newsize)
        img.save(file_path)


def convert_to_png(file_path):
    if file_path.lower().endswith(('.jpg', '.jpeg', '.tif', '.bmp')):
        img = Image.open(file_path)
        file_dir, file_name = os.path.split(file_path)
        file_name, _ = os.path.splitext(os.path.basename(file_name))
        img.save(os.path.join(file_dir, file_name) + '.png', format='png')
        print(file_name + ' converted to PNG in ' + file_dir)


def batch_png_convert(folder_path, new_folder='True'):
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.bmp')):
            file_path = os.path.join(folder_path, f)
            if new_folder == 'False':
                save_folder = folder_path
            elif new_folder == 'True':
                save_folder = folder_path + '_png'
                print(save_folder)
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)
            convert_to_png(file_path, save_folder)


def batch_scale(folder_path, scaling_factor, new_folder='True'):
    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path, f)
        scale_images(file_path, scaling_factor, new_folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True)  # indicate name of file or folder
    ap.add_argument("-f", "--function", required=True)
    ap.add_argument("-s", "--scaling_factor", required=False)

    args = vars(ap.parse_args())
    name = args["name"]
    function = args["function"]
    scaling_factor = args["scaling_factor"]

    if os.path.isdir(name):
        if function == "rescale":
            batch_scale(name, scaling_factor)
        if function == "PNGconvert":
            batch_png_convert(name)
    if os.path.isfile(name):
        if function == "rescale":
            scale_images(name, scaling_factor)
        if function == "PNGconvert":
            convert_to_png(name)


if __name__ == "__main__":
    main()