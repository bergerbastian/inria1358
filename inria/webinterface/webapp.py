import streamlit as st
import pandas as pd
import requests
import numpy as np
import json

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(layout="wide")

street = st.sidebar.text_input("Address", "Schützenstraße 40, Berlin")

def prediction():
    st.header(f"Prediction for {street}")
    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street)

    lat = location.latitude
    lon = location.longitude

    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    with st.spinner('Wait for it...'):
        columns = st.columns(3)
        columns[0].write("Location")
        columns[0].map(map_data)
        url = f'http://localhost:8000/predict-maps?lat={lat}&lon={lon}'
        r = requests.get(url=url)
        dict = json.loads(r.json())
        columns[1].write("Input Image")
        columns[1].image(np.asarray(dict['original_image']).astype(np.uint8))
        columns[2].write("Segmentation Mask")
        columns[2].image(np.asarray(dict['predicted_mask']).astype(np.uint8))

st.sidebar.button("Predict Buildings", on_click=prediction)
