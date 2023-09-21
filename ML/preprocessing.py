import os
import tensorflow as tf
from colorama import Fore, Style
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(image_path, mask_path=None):
    # Load and preprocess the image
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels=3)
    image = image/255
    #image = tf.pad(image, [[12,12], [12,12], [0,0]], mode='REFLECT') #To be removed and model adjusted


    # Load and preprocess the mask image
    if mask_path is not None:
        mask = tf.io.read_file(mask_path)
        mask = tf.io.decode_png(mask, channels=1)
        mask = mask/255
        #mask = tf.pad(mask, [[12,12], [12,12], [0,0]], mode='REFLECT') #To be removed and model adjusted

        return image, mask
    else:
        return image

def create_datasets(save_path, set="train", test_size=.1, batch_size=64, data_size=1):

    print(Fore.BLUE + "\nCreating dataset for {set}" + Style.RESET_ALL)

    if set == "predict":
        image_dir = save_path
    else:
        image_dir = f'{save_path}/{set}/images'
    mask_dir = f'{save_path}/{set}/gt' if set == "train" else None

    image_path = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    mask_path = [os.path.join(mask_dir, filename) for filename in os.listdir(mask_dir)] if mask_dir else None

    image_path_subset, _, mask_path_subset, _ = train_test_split(image_path, mask_path, test_size=data_size)

    if mask_path:
        dataset = tf.data.Dataset.from_tensor_slices((image_path_subset, mask_path_subset))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(image_path_subset)

    dataset = dataset.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

    #Shuffling full dataset
    buffer_size = len(image_path)
    shuffled_dataset = dataset.shuffle(buffer_size) if set == "train" else dataset

    #Splitting dataset into train and test

    if test_size == 0:
        dataset = shuffled_dataset
        dataset_batches = (dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

        pics_test = len(dataset)
        print(f"✅ Dataset_batches with {pics_test} images ({len(dataset_batches)} batches) created")

        return dataset_batches

    else:
        train_size = int((1-test_size) * buffer_size)
        train_dataset = shuffled_dataset.take(train_size)
        test_dataset = shuffled_dataset.skip(train_size)

        train_batches = (train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
        test_batches = (test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

        pics_train = len(train_dataset)
        pics_test = len(test_dataset)

        print(f"✅ Train_batches with {pics_train} images ({len(train_batches)} batches) created")
        print(f"✅ Test_batches with {pics_test} images ({len(test_batches)} batches) created")

        return train_batches, test_batches
