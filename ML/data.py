# Imports
from PIL import Image
import numpy as np
from patchify import patchify
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import tensorflow as tf
from colorama import Fore, Style

def make_patches(source_path: str, save_path: str, image_size=(200,200), image_type="png", max_files = None):
    """Function to create patches from the original images, in order to be able to feed them to the model
        Args:
            source_path: path to the original dataset
            save_path: path where the patches should be saved
            image_size: (width, height) of patches
            image_type: file type that the patches should be outputted in
            max_files: maximum number of images to process per subfolder (selected randomly), None = no limit
    """

    # Checking if directories exist, if not, create them
    for folder in ['train','test']:
        subfolders = ['gt', 'images'] if folder == "train" else ['images']
        for subfolder in subfolders:
            load_path = f'{source_path}/{folder}/{subfolder}'
            save_path_n = f'{save_path}/{folder}/{subfolder}'
            # Check whether the specified path exists or not
            isExist = os.path.exists(save_path_n)
            if not isExist:
                os.makedirs(save_path_n)

    # Checking the local data
    sets = []
    print(Fore.BLUE + "\nChecking local data..." + Style.RESET_ALL)

    # Get # of images from save path
    train_images = len(os.listdir(f"{save_path}/train/images"))
    train_gt = len(os.listdir(f"{save_path}/train/gt"))
    test_images = len(os.listdir(f"{save_path}/test/images"))

    print(f"{train_images} Patches found for X_train")
    print(f"{train_gt} Patches found for y_train")
    print(f"{test_images} Patches found for X_test")

    # Check if >0 images are there for each set (meaning we have data) otherwise add it to the sets that we want to patch
    if train_images == 0 or train_gt == 0:
        print("❗ Train folders contain 0 images")
        sets.append("train")
    else:
        print("ℹ️ Train patches already exist")

    if test_images == 0:
        print("❗ Test folders contain 0 images")
        sets.append("test")
    else:
        print("ℹ️ Test patches already exist")


    # Iterate through the sets
    for set in sets:

        # Subfolders to iterate through, if the set is train, iterate also iterate through GT folder
        subfolders = ['images']
        if set == 'train':
            subfolders.append('gt')

        # Iterate through each subfolder

        for subfolder in subfolders:
            load_path = f'{source_path}/{set}/{subfolder}'
            save_path_n = f'{save_path}/{set}/{subfolder}'

            print(Fore.BLUE + f"\nIterating through folder: {set}/{subfolder}" + Style.RESET_ALL)

            dimensions = image_size if subfolder == "gt" else (image_size[0],image_size[1],3)
            file_count = 0

            files = os.listdir(load_path)

            if max_files is not None:
                files = random.sample(files, max_files)

            for filename in tqdm(files):
                im = Image.open(f'{load_path}/{filename}')
                imarray = np.array(im)
                patches = patchify(imarray, dimensions, step=200)
                for i, row in enumerate(patches):
                    for j, col in enumerate(row):
                        if subfolder == "gt":
                            im = Image.fromarray(col)
                        else:
                            im = Image.fromarray(col[0])
                        im.save(f'{save_path_n}/{filename}__{i}__{j}.{image_type}')

    # Final Output
    # Get # of images from save path
    train_images = len(os.listdir(f"{save_path}/train/images"))
    train_gt = len(os.listdir(f"{save_path}/train/gt"))
    test_images = len(os.listdir(f"{save_path}/test/images"))

    if len(sets) > 0:
        print(Fore.BLUE + f"\nCounting local images..." + Style.RESET_ALL)
        print(f"{train_images} Patches found for X_train")
        print(f"{train_gt} Patches found for y_train")
        print(f"{test_images} Patches found for X_test")
    print("✅ Patches loaded")
