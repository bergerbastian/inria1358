from fastapi import FastAPI
from fastapi.responses import FileResponse

import json

from inria.ml_logic.registry import load_model
from inria.utils import predict_image_maps, save_image_prediction
from inria.params import LOCAL_API_DATA_FOLDER

import numpy as np

import tensorflow as tf

app = FastAPI()

app.state.model = tf.keras.models.load_model("/Users/paulrenger/code/Paukhard/inria1358/mlops/unet") #load_model()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

@app.get('/predict-maps')
async def predict_map(lat, lon, zoom):
    if int(zoom) >= 17:
        original, gt, prediction = predict_image_maps(lat, lon, app.state.model, zoom=zoom)
        path = save_image_prediction(prediction, f"{LOCAL_API_DATA_FOLDER}/pred", f"_{lat}__{lon}")
        return json.dumps({"original_image": original.tolist(), "gt": gt.tolist(), "predicted_mask": np.uint8(prediction*255).tolist()})
    else:
        original, prediction = predict_image_maps(lat, lon, app.state.model, zoom=zoom)
        path = save_image_prediction(prediction, f"{LOCAL_API_DATA_FOLDER}/pred", f"_{lat}__{lon}")
        return json.dumps({"original_image": original.tolist(), "predicted_mask": np.uint8(prediction*255).tolist()})
    #return FileResponse(path)
    #return {"original_image": original.tolist(), "predicted_mask": np.uint8(prediction*255).tolist()}
