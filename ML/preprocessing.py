import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(image_path, mask_path=None):
    """Function to load images from local storage and preprocess them
        Args:
            image_path: list of all image paths
            mask_path: list of all mask paths (None for tets/predict sets)
    """
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

def create_datasets(save_path, set="train", test_size=.1, batch_size=64, data_size=1, val_size=.3):
    """Function to create dataset batches
        Args:
            save_path: path of images
            set: kind of input dataset --> train, test or predict (save_path for predict is end folder)
            test_size: size of test_set, ignored for set in ["predict", "test"] --> float in range (0,1)
            batch_size= size of batches --> int
            data_size: size of original data to be used --> float in range (0,1), 1 = total data
            val_size: size of validation set, ignored for set in ["predict", "test"] --> float in range (0,1)
    """
    if set == "predict":
        image_dir = save_path
    else:
        image_dir = f'{save_path}/{set}/images'
    mask_dir = f'{save_path}/{set}/gt' if set == "train" else None

    image_path = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    mask_path = [os.path.join(mask_dir, filename) for filename in os.listdir(mask_dir)] if mask_dir else None

    if set in ["predict", "test"] :
        dataset = tf.data.Dataset.from_tensor_slices(image_path)
        dataset = dataset.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

        predict_batches = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        pics_test = len(dataset)
        print(f"✅ Predict_batches with {pics_test} images ({len(predict_batches)} batches) created")

        return predict_batches

    else:
        #test split
        image_path_rest, image_path_test, mask_path_rest, mask_path_test = train_test_split(image_path, mask_path, test_size=test_size, random_state=42) if test_size != 0 else (None, None, None, None)

        #data size split
        if test_size == 0:
            image_path_rest = image_path
            mask_path_rest = mask_path
        image_path_subset, _, mask_path_subset, _ = train_test_split(image_path_rest, mask_path_rest, train_size=data_size) if data_size != 1 else (None, None, None, None)

        #train val split
        if data_size == 1:
            image_path_subset = image_path_rest
            mask_path_subset = mask_path_rest
        image_path_train, image_path_val, mask_path_train, mask_path_val = train_test_split(image_path_subset, mask_path_subset, train_size=val_size)

        train_dataset = tf.data.Dataset.from_tensor_slices((image_path_train, mask_path_train))
        train_dataset = train_dataset.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.cache("/Users/bastianberger/code/bergerbastian/inria1358/Cache/train")
        train_dataset = train_dataset.shuffle(len(image_path_train)*.2)
        train_batches = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((image_path_val, mask_path_val))
        val_dataset = val_dataset.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.cache("/Users/bastianberger/code/bergerbastian/inria1358/Cache/val")
        val_dataset = val_dataset.shuffle(len(image_path_val)*.2)
        val_batches = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((image_path_test, mask_path_test)) if test_size != 0 else None
        test_dataset = test_dataset.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE) if test_size != 0 else None
        test_batches = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) if test_size != 0 else None

        pics_train = len(train_dataset)
        pics_val = len(val_dataset)
        pics_test = len(test_dataset) if test_size != 0 else None

        print(f"✅ Test_batches with {pics_test} images ({len(test_batches)} batches) created") if test_size != 0 else None
        print(f"✅ Train_batches with {pics_train} images ({len(train_batches)} batches) created")
        print(f"✅ Val_batches with {pics_val} images ({len(val_batches)} batches) created")

        if test_size != 0:
            return test_batches, train_batches, val_batches
        else:
            return train_batches, val_batches
