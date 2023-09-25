from fastapi import FastAPI
from fastapi.responses import FileResponse

import json

from inria.ml_logic.registry import load_model
from inria.utils import predict_image_maps, save_image_prediction
from inria.params import LOCAL_API_DATA_FOLDER

import numpy as np

app = FastAPI()

app.state.model = load_model()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

@app.get('/predict-maps')
async def predict_map(lat, lon):
    original, prediction = predict_image_maps(lat,lon, app.state.model)
    path = save_image_prediction(prediction, LOCAL_API_DATA_FOLDER, f"_{lat}__{lon}")
    #return FileResponse(path)
    return json.dumps({"original_image": original.tolist(), "predicted_mask": np.uint8(prediction*255).tolist()})
    #return {"original_image": original.tolist(), "predicted_mask": np.uint8(prediction*255).tolist()}
