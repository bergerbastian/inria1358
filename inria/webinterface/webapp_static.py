import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import tensorflow as tf
from google.cloud import storage

from inria.utils import compute_iou, predict_image_maps



st.set_page_config(layout="wide")


# SIDEBAR
street = st.sidebar.text_input("Address", "Schützenstraße 40, Berlin")
zoom_level = st.sidebar.number_input("Zoom", min_value=2, max_value=20, value=17, format="%i")
threshold = st.sidebar.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.5)
model_selection = st.sidebar.selectbox('What model do you want to use?', ('unet', 'segnet'))
show_iou = st.sidebar.checkbox('Show IOU graph')

storage_client = storage.Client(project="le-wagon-bootcamp-398616")
buckets = storage_client.list_buckets()

print("Buckets:")
for bucket in buckets:
    print(bucket.name)
print("Listed all storage buckets.")

# PREDICT FUNCTION
def prediction():

    # SET HEADER
    st.header(f"Prediction for {street}")

    # GET LOCATION DATA
    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street)

    lat = location.latitude
    lon = location.longitude

    columns = st.columns(3)

    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})


    # PROCESS COMPUTE INTENSIVE TASK
    with st.spinner('Wait for it...'):

        model = tf.keras.models.load_model(f"/Users/paulrenger/code/Paukhard/inria1358/mlops/{model_selection}")

        original, gt, prediction = predict_image_maps(lat, lon, model, zoom=zoom_level)

        # SHOW PREDICT MASK
        columns[2].write("Segmentation Mask")
        columns[2].image(((prediction)>threshold)*255)
        sub_columns = columns[2].columns(3)

        sub_columns[1].metric(label="Max value", value=f"{np.round(np.max(prediction)/255,3)}")
        sub_columns[2].metric(label="Min value", value=f"{np.round(np.min(prediction)/255,3)}")

        # IF ZOOM >= 17, THEN WE HAVE A GROUND TRUTH, SHOW GROUND TRUTH
        if zoom_level >= 17:

            columns[0].write("Input Image")
            columns[0].image(original)

            columns[1].write("Ground Truth")
            columns[1].image(gt)

            # CALCULATE CHANGE TO PREVIOUS METRIC
            delta = None
            current_iou = compute_iou(prediction>threshold, np.array(tf.squeeze(tf.image.rgb_to_grayscale(gt))))

            if 'prev_iou' in st.session_state:
                delta = f"{np.round((current_iou/st.session_state['prev_iou']-1)*100)}%"
                st.session_state['prev_iou'] = current_iou
            else:
                st.session_state['prev_iou'] = current_iou

            columns[1].metric(label="IOU", value=f"{current_iou}", delta=delta)

        # NO GROUND TRUTH
        else:
            columns[1].write("Input Image")
            columns[1].image(original)

        # IF WE WANT TO SHOW IOU GRAPH
        if show_iou:
            fig, ax = plt.subplots()

            linspace = np.linspace(0.05, 0.95, num=50)
            y = [compute_iou(prediction>ts, np.array(tf.squeeze(tf.image.rgb_to_grayscale(gt)))) for ts in linspace]
            ax.plot(linspace, y)
            ax.set_ylabel("IOU")
            ax.set_xlabel("Threshold")

            columns[0].pyplot(fig)

    st.header("Location")
    st.map(map_data)



st.sidebar.button("Predict Buildings", on_click=prediction)
