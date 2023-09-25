from fastapi import FastAPI
from fastapi.responses import FileResponse

from inria.ml_logic.registry import load_model


from inria.utils import predict_image_maps, save_image_prediction
from inria.params import LOCAL_API_DATA_FOLDER

app = FastAPI()

app.state.model = load_model()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

@app.get('/predict-maps')
async def predict_map(lat, lon):
    path = save_image_prediction(predict_image_maps(lat,lon, app.state.model), LOCAL_API_DATA_FOLDER, f"_{lat}__{lon}")
    return FileResponse(path)
