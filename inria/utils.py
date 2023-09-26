from inria.ml_logic.preprocessing import load_and_preprocess_data
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import numpy as np
from patchify import patchify
import urllib
import time
from inria.params import MAPS_API_KEY, LOCAL_API_DATA_FOLDER

def predict_image_maps(lat, lon, model, zoom=17, return_ground_truth=True):
    """Returns original image and prediction matrix for a google maps image from the specified lat,lon. If zoom >= 17 and return_ground_truth = True, it also returns a GT generated from google maps api.
    """
    image_url=f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=640x640&scale=2&maptype=satellite&key={MAPS_API_KEY}"
    dimensions = (200,200, 3)

    image_path = LOCAL_API_DATA_FOLDER
    image_filename = f"{str(lat).replace('.','_')}__{str(lon).replace('.','_')}"
    image_type = "png"

    urllib.request.urlretrieve(image_url, f"{image_path}/input/{image_filename}.{image_type}")
    # Open the downloaded image in PIL

    left = 40
    top = 40
    right = 1280-40
    bottom = 1280-40

    my_img = Image.open(f"{image_path}/input/{image_filename}.{image_type}").crop((left, top, right, bottom)).convert("RGB")

    patch_list = []
    #im = Image.open(f'{image_path}')
    imarray = np.array(my_img)
    patches = patchify(imarray, dimensions, step=dimensions[0])

    predict_data = []
    for r_ind in tqdm(range(patches.shape[0])):
        col_predict = []
        for c_ind in range(patches.shape[1]):
            image = patches[r_ind][c_ind]
            image = image/255

            # Predict
            predict_mask = model.predict(image, verbose=0)

            # Remove batch
            predict_mask = tf.squeeze(predict_mask)
            col_predict.append(predict_mask)
        predict_data.append(col_predict)

    rows = [np.hstack(predict_data[r]) for r in range(patches.shape[1])]
    prediction = np.vstack(rows)

    if return_ground_truth and int(zoom) >= 17:
        return imarray, get_ground_truth(lat,lon, zoom), prediction
    else:
        return imarray, prediction

def save_image_prediction(prediction, path, custom_suffix=""):
    """saves image at the specified path"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    Image.fromarray(np.uint8(prediction*255)).save(f"{path}/{timestamp}_{custom_suffix}.png")
    return f"{path}/{timestamp}_{custom_suffix}.png"


def get_ground_truth(lat, lon, zoom=17):
    gt_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=640x640&scale=2&map_id=c2e4254a97b86e42&style=feature:all|element:labels|visibility:off&key={MAPS_API_KEY}"

    image_path = LOCAL_API_DATA_FOLDER
    image_filename = f"{str(lat).replace('.','_')}__{str(lon).replace('.','_')}"
    image_type = "png"
    urllib.request.urlretrieve(gt_url, f"{image_path}/{image_filename}.{image_type}")
    # Open the downloaded image in PIL

    left = 40
    top = 40
    right = 1280-40
    bottom = 1280-40

    my_img = Image.open(f"{image_path}/{image_filename}.{image_type}").crop((left, top, right, bottom)).convert("RGB")

    # Load or create your image as a NumPy array
    image = np.array(my_img)  # Replace 'your_image' with your actual image array

    # Define the desired color and a tolerance
    desired_color = (191, 48, 191)  # Red in RGB
    tolerance = 80  # Adjust this tolerance as needed

    # Create a mask for the desired color
    lower_bound = np.array(desired_color) - tolerance
    upper_bound = np.array(desired_color) + tolerance
    mask = np.all((image >= lower_bound) & (image <= upper_bound), axis=2)

    # Color the masked pixels white
    result = np.zeros_like(image)
    result[mask] = (255,255,255)

    return result

def compute_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) of two binary segmentation masks.

    Args:
        mask1 (numpy.ndarray): First binary mask.
        mask2 (numpy.ndarray): Second binary mask.

    Returns:
        float: IoU score.
    """
    # Ensure the masks are binary
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Intersection and Union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union



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
