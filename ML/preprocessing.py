import os
import tensorflow as tf

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

def create_datasets(save_path, set="train", test_size=.1, batch_size=64):

    image_dir = f'{save_path}/{set}/images'
    mask_dir = f'{save_path}/{set}/gt' if set == "train" else None

    image_path = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    mask_path = [os.path.join(mask_dir, filename) for filename in os.listdir(mask_dir)] if mask_dir else None

    if mask_path:
        dataset = tf.data.Dataset.from_tensor_slices((image_path, mask_path))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(image_path)

    dataset = dataset.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

    #Shuffling full dataset
    buffer_size = len(image_path)
    shuffled_dataset = dataset.shuffle(buffer_size)

    #Splitting dataset into train and test

    if test_size == 0:
        train_dataset = shuffled_dataset
        train_batches = (train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

        return train_batches

    else:
        train_size = int(test_size * buffer_size)
        train_dataset = shuffled_dataset.take(train_size)
        test_dataset = shuffled_dataset.skip(train_size)

        train_batches = (train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
        test_batches = (test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

        return train_batches, test_batches
