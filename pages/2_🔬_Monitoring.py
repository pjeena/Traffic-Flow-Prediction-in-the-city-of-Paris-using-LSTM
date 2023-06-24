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
from datetime import date, timedelta, datetime
from pathlib import Path
import streamlit as st
import tensorflow as tf
import locale
import pytz  
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
import folium.plugins as plugins
from streamlit_folium import st_folium, folium_static
from src.data_fetching_util import GetDataFromAPI



st.set_page_config(layout="wide")

# title
current_date = pd.to_datetime(datetime.utcnow()).floor('H')
st.title(f'Monitoring dashboard 🔎')

with st.sidebar:
    st.warning('*Mean Absolute Error (MAE) per junction and hour', icon="📌")

dir_path = 'output/predictions/*/*.csv'
file_names = sorted(glob.glob(dir_path))


rmse_junctions = []
junction_labels = []


for file in file_names:
    df = pd.read_csv(file, index_col=0)
    y_pred = (
        df[df["type"] == "Prediction"].reset_index(drop=True)["no_of_vehicles"].values
    )
    y_actual = (
        df[df["type"] == "Actual"].reset_index(drop=True)["no_of_vehicles"].values
    )

    mae = np.abs(y_pred-y_actual)
    date_time = list(df["datetime"].drop_duplicates())

    df_rmse = pd.DataFrame(list(zip(date_time, mae)), columns=["datetime", "mae"])
    rmse_junctions.append(df_rmse)
    junction_labels.append(file.split('/')[-2])






for i,junction in enumerate(junction_labels):
    fig = px.line(rmse_junctions[i], x="datetime", y="mae", title=junction)
    st.plotly_chart(fig,theme='streamlit',use_container_width=True)
