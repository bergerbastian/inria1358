from fastapi import FastAPI
from patchify import patchify
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ML.registry import load_model
import tensorflow as tf
from tqdm import tqdm
import urllib.request
import time

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

@app.get('/predict-maps')
def predict_map(lat, lon):
    return {'lat': lat, 'lon':lon}
