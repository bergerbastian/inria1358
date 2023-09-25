from ML.preprocessing import load_and_preprocess_data
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import numpy as np
from patchify import patchify

#
#   !!WARNING: These functions do not yet work as expected!!
#

def predict_full_image_from_patches(save_path, model, image_name):
    """
        Combines image arrays (for instance from predict) to full image

        Args:
            images: 2D Array with rows cols containing tensors/images arrays
    """
    path = f"{save_path}/train/images"

    predict_data = [] # 2D Array (rows, cols) holding the predictions
    for r in tqdm(range(25)):
        col = []
        for c in range(25):
            # Load Image
            img_path = f"{path}/{image_name}__{r}__{c}.png"

            # Preparing for prediction
            image = load_and_preprocess_data(img_path)
            image = tf.expand_dims(image, axis=0)

            # Predict
            predict_mask = model.predict(image, verbose=0)

            # Remove batch
            predict_mask = tf.squeeze(predict_mask)

            col.append(predict_mask)
        predict_data.append(col)

    return predict_data

def patchify_image(image_path, dimensions=(200,200,3)):
    patch_list = []
    im = Image.open(f'{image_path}')
    imarray = np.array(im)
    patches = patchify(imarray, dimensions, step=dimensions[0])
    for i, row in enumerate(patches):
        cols = []
        for j, col in enumerate(row):
            if len(dimensions) == 2:
                im = Image.fromarray(col)
            else:
                im = Image.fromarray(col[0])
            cols.append(np.array(im))
        patch_list.append(cols)
    return patch_list

def predict_mask_from_full_image(model, image_path, dimensions=(200,200,3)) -> Image:
    predict_data = [] # 2D Array (rows, cols) holding the predictions
    patch_list = patchify_image(image_path, dimensions=dimensions)
    for r in tqdm(range(25)):
        col = []
        for c in range(25):
            # Load Image
            # Preparing for prediction
            image = patch_list[r][c]
            image = image/255
            image = tf.expand_dims(image, axis=0)

            # Predict
            predict_mask = model.predict(image, verbose=0)

            # Remove batch
            predict_mask = tf.squeeze(predict_mask)

            col.append(predict_mask)
        predict_data.append(col)

    rows = [np.hstack(predict_data[r]) for r in tqdm(range(25))]

    return Image.fromarray(np.vstack(rows))
