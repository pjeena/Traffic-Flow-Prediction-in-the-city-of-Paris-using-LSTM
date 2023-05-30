import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import os
import logging
import os
import json
import sys
import glob
import pickle
import altair as alt
from datetime import date, timedelta
from pathlib import Path
import streamlit as st
import tensorflow as tf
import locale
import pytz  
import pydeck as pdk
import plotly.express as px
import folium
from folium.plugins import HeatMap
import folium.plugins as plugins
from streamlit_folium import st_folium, folium_static
from src.data_fetching_util import GetDataFromAPI
from src.utils import (
    preprocess_fetched_data_from_API,
    feature_build_from_preprocessed_data,
    standarize_data,
)
from tensorflow.keras.models import load_model
from app_utils import (
    define_constant_attributes,
    get_live_raw_data,
    get_preprocessed_raw_data,
    build_features,
    get_predictions,
)


(
    junction_labels,
    window_lag,
    window_lead,
    num_components_in_one_time_step,
    dates,
    current_hour,
    past_week_hour,
) = define_constant_attributes()
coordinates = pd.read_csv('coordinates.csv',index_col=0)
#dir_predictions = os.path.join('output', 'predictions', '*' , '*.csv')
#file_names = sorted(glob.glob(dir_predictions))




################################## Pipeline component #####################################

url = 'https://opendata.paris.fr/explore/dataset/comptage-multimodal-comptages/information/?disjunctive.label&disjunctive.mode&disjunctive.voie&disjunctive.sens&disjunctive.trajectoire'
get_live_raw_data(dates)
get_preprocessed_raw_data(dates, junction_labels)
build_features(window_lag, window_lead)
get_predictions(current_hour, past_week_hour, window_lag, num_components_in_one_time_step)

