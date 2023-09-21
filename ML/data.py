# Imports
from PIL import Image
import numpy as np
from patchify import patchify
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tensorflow as tf
from colorama import Fore, Style

def make_patches(source_path, save_path, image_size=(200,200), image_type="png"):
    isExist = os.path.exists(save_path)

    sets = []

    # Check if save path exists
    if isExist:

        print(Fore.BLUE + "\nChecking local data..." + Style.RESET_ALL)
        # Get # of images from save path
        train_images = len(os.listdir(f"{save_path}/train/images"))
        train_gt = len(os.listdir(f"{save_path}/train/gt"))
        test_images = len(os.listdir(f"{save_path}/test/images"))

        print(f"{train_images} Patches found for X_train")
        print(f"{train_gt} Patches found for y_train")
        print(f"{test_images} Patches found for X_test")


        print("✅ Patches already exist")
        return
    else:
        # No folder exist yet, so we need to make sets for both train & test

        sets = ["train", "test"]

    for set in sets:
        subfolders = ['images']
        if set == 'train':
            subfolders.append('gt')

        for subfolder in subfolders:
            load_path = f'{source_path}/{set}/{subfolder}'
            save_path_n = f'{save_path}/{set}/{subfolder}'
            # Check whether the specified path exists or not
            isExist = os.path.exists(save_path_n)
            if not isExist:
                os.makedirs(save_path_n)
            print("-"*10)
            print(f" Iterating through folder: {subfolder}")
            print("-"*10)

            dimensions = image_size if subfolder == "gt" else (image_size[0],image_size[1],3)

            for filename in tqdm(os.listdir(load_path)):
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
