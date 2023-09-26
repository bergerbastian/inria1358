import streamlit as st
import pandas as pd
import requests
import numpy as np
import json

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import tensorflow as tf

from inria.utils import compute_iou

st.set_page_config(layout="wide")



street = st.sidebar.text_input("Address", "Schützenstraße 40, Berlin")
zoom_level = st.sidebar.number_input("Zoom", min_value=2, max_value=20, value=17, format="%i")
threshold = st.sidebar.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.5)
def prediction():
    st.header(f"Prediction for {street}")
    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street)

    lat = location.latitude
    lon = location.longitude

    columns = st.columns(3)

    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})



    with st.spinner('Wait for it...'):
        url = f'http://localhost:8000/predict-maps?lat={lat}&lon={lon}&zoom={zoom_level}'
        r = requests.get(url=url)
        dict = json.loads(r.json())


        predicted_mask = np.asarray(dict['predicted_mask']).astype(np.uint8)
        columns[2].write("Segmentation Mask")
        columns[2].image(((predicted_mask/255)>threshold)*255)
        sub_columns = columns[2].columns(3)

        sub_columns[1].metric(label="Max value", value=f"{np.round(np.max(predicted_mask)/255,3)}")
        sub_columns[2].metric(label="Min value", value=f"{np.round(np.min(predicted_mask)/255,3)}")

        if zoom_level >= 17:

            columns[0].write("Input Image")
            columns[0].image(np.asarray(dict['original_image']).astype(np.uint8))

            gt_mask = np.asarray(dict['gt']).astype(np.uint8)
            columns[1].write("Ground Truth")
            columns[1].image(gt_mask)

            columns[1].metric(label="IOU", value=f"{compute_iou(predicted_mask/255>threshold, np.array(tf.squeeze(tf.image.rgb_to_grayscale(gt_mask))))}")

        else:
            columns[1].write("Input Image")
            columns[1].image(np.asarray(dict['original_image']).astype(np.uint8))

    st.header("Location")
    st.map(map_data)



st.sidebar.button("Predict Buildings", on_click=prediction)
