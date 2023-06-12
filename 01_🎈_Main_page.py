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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


st.set_page_config(layout="wide")
reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)
hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)



col1, col2 = st.columns([6.7,5])
with col1:
        st.title(f'Hourly Traffic rate in Paris 🚕')
with col2:
        option_hr = st.radio(
            "",
            ('Current hr', 'After 1 hr', 'After 2 hrs'), horizontal=True)
st.write('*This app forecasts the traffic for the next 3 hours from now in real time at 8 major junctions of Paris*')

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




################################## UI Template component #####################################
#st.set_page_config(layout="wide")




progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 4


url = 'https://opendata.paris.fr/explore/dataset/comptage-multimodal-comptages/information/?disjunctive.label&disjunctive.mode&disjunctive.voie&disjunctive.sens&disjunctive.trajectoire'
with st.spinner('Fetching live data from [OpenDataParis](%s) API.....' %url ):
#    get_live_raw_data(dates)
    time.sleep(1.0)
    st.sidebar.write('✅ Data fetched from API')
    progress_bar.progress(1/N_STEPS)


with st.spinner('Preprocessing raw data.....'):
#    get_preprocessed_raw_data(dates, junction_labels)
    time.sleep(0.0)
    st.sidebar.write('✅ Data preprocessing completed')
    progress_bar.progress(2/N_STEPS)


with st.spinner('Creating features and targets from time series data.....'):
 #   build_features(window_lag, window_lead)
    time.sleep(0.0)
    st.sidebar.write('✅ Feature engineering completed')
    progress_bar.progress(3/N_STEPS)


with st.spinner('Getting live predictions from the models.....'):
#    get_predictions(current_hour, past_week_hour, window_lag, num_components_in_one_time_step)
    time.sleep(0.0)
    st.sidebar.write('✅ Predictions arrived')
    progress_bar.progress(4/N_STEPS)



with st.sidebar:
    st.warning('*Hover over the markers to see the junction name, traffic and change in % from prior hr*', icon="📌")
    st.sidebar.markdown("**:red[*Scroll the main page below to see predictions for the 8 junctions in Paris*]**")




##################################### MAP Generation *******************************
dir_path = 'output/predictions/*/*.csv'
file_names = sorted(glob.glob(dir_path))

df_previous_hour = pd.DataFrame()
df_current_hour = pd.DataFrame()
df_next_hour = pd.DataFrame()
df_next2next_hour = pd.DataFrame()

for file in file_names: 
        df = pd.read_csv(file,index_col=0)
        df_pred = df[df['type'] == 'Prediction']
        df_pred['junction'] = file.split('/')[-2]

        check_if_real_data_is_present_or_not = df[df.type == 'Actual']['no_of_vehicles']
        if (check_if_real_data_is_present_or_not == 0.0).sum() > 50:
                    #if there are more than 50 data points with 0.0 as number of vehicles, we skip that junction
                    continue
                    
        df_minus_1_hr = pd.DataFrame( [df_pred.iloc[-4] ] )
        df_previous_hour = pd.concat([df_previous_hour, df_minus_1_hr])
        df_previous_hour  = df_previous_hour.reset_index(drop=True)


        df_0_hr = pd.DataFrame( [df_pred.iloc[-3] ] )
        df_current_hour = pd.concat([df_current_hour, df_0_hr])
        df_current_hour  = df_current_hour.reset_index(drop=True)

        df_1_hr = pd.DataFrame( [df_pred.iloc[-2] ] )
        df_next_hour = pd.concat([ df_next_hour, df_1_hr])
        df_next_hour  = df_next_hour.reset_index(drop=True)

        df_2_hr = pd.DataFrame( [df_pred.iloc[-1] ] )
        df_next2next_hour = pd.concat([df_next2next_hour, df_2_hr])
        df_next2next_hour  = df_next2next_hour.reset_index(drop=True)



coordinates = pd.read_csv('coordinates.csv',index_col=0).reset_index(names='junction')

to_plot_minus_1_hr = pd.merge( df_previous_hour, coordinates , on='junction')
to_plot_0_hr = pd.merge( df_current_hour, coordinates , on='junction')
to_plot_1_hr = pd.merge( df_next_hour, coordinates , on='junction')
to_plot_2_hr = pd.merge( df_next2next_hour, coordinates , on='junction')

#to_plot_minus_1_hr.to_csv('output/predictions/map_minus_1_hr')
#to_plot_0_hr.to_csv('output/predictions/map_0_hr')
#to_plot_1_hr.to_csv('output/predictions/map_1_hr')
#to_plot_2_hr.to_csv('output/predictions/map_2_hr')


if option_hr == 'Current hr':
            to_plot = to_plot_0_hr
            change = list( ( to_plot['no_of_vehicles'] - to_plot_minus_1_hr['no_of_vehicles'] )/ to_plot_minus_1_hr['no_of_vehicles']  * 100 )
elif option_hr == 'After 1 hr':
            to_plot = to_plot_1_hr
            change = list( ( to_plot['no_of_vehicles'] - to_plot_0_hr['no_of_vehicles'] )/ to_plot_0_hr['no_of_vehicles']  * 100 )

elif option_hr == 'After 2 hrs':
            to_plot = to_plot_2_hr
            change = list( ( to_plot['no_of_vehicles'] - to_plot_1_hr['no_of_vehicles'] )/ to_plot_1_hr['no_of_vehicles']  * 100 )




heatmap_data = list(map(list, zip(to_plot["latitudes"],
                          to_plot["longitudes"],
                          to_plot["no_of_vehicles"]
                         )
               )
           )



map_hr = folium.Map(location=[  48.8705  , 2.34639   ], zoom_start=14.0)
            #tiles="CartoDB positron",

HeatMap(heatmap_data).add_to(map_hr)
for i in range(0,len(to_plot)):
                        text = 'junction : <b>{}</b> , vehicles : <b> {}</b>'.format(to_plot.iloc[i]['junction'],int(to_plot.iloc[i]['no_of_vehicles'])) 

                        title_html = '''<h3 align="center" style="font-size:15px"> Junction : <b>{}</b> <br>
                                        Traffic (No of vehicles) : <b>{}</b>  <br>
                                        Change from prior hour : <b>{}</b> % </h3>
                        '''.format(to_plot.iloc[i]['junction'],int(to_plot.iloc[i]['no_of_vehicles']), int( np.round(change[i])) )
            #           text = folium.map.Tooltip( text,   style='width:300px; height:300px; white-space:normal;')
                        folium.Marker(
                            location=[to_plot.iloc[i]['latitudes'], to_plot.iloc[i]['longitudes']],
                            tooltip =   title_html,
                            radius=13,
                            color='red',
#                            fill_color='red',
 #                           fill_opacity= to_plot.iloc[i]['no_of_vehicles']/to_plot['no_of_vehicles'].max()
                        ).add_to(map_hr)

folium_static(map_hr, width=1000,height=540)




col3, col4, col5 = st.columns([1.0, 10 , 1.0])
with col4:
        st.markdown('**Hourly Traffic prediction plots obtained from LSTM model since the past 7 days and upto 3 hours in future:**')
        for file in file_names:
                print(file)
                df = pd.read_csv(file,index_col=0)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df["datetime"] = df["datetime"].dt.tz_convert(pytz.timezone("Europe/Berlin"))
#                chart = get_chart(df,file.split('/')[2])


                check_if_real_data_is_present_or_not = df[df.type == 'Actual']['no_of_vehicles']
                if (check_if_real_data_is_present_or_not == 0.0).sum() > 50:
                    #if there are more than 50 data points with 0.0 as number of vehicles, we skip that junction
                    st.warning('Real time data is not available for :red[**{}**]'.format(file.split('/')[-2]), icon="⚠️")
                    continue
                    

                trace1 = go.Scatter(
                    x=  df[df.type == 'Prediction']['datetime'] ,
                    y= df[df.type == 'Prediction']['no_of_vehicles'],
                    name='Prediction', mode='lines+markers', marker_color='red',
                    
                )
                trace2 = go.Scatter(
                    x= df[df.type == 'Actual'].iloc[:-2 , :]['datetime'] ,
                    y= df[df.type == 'Actual'].iloc[:-2 , :]['no_of_vehicles'],
                    name='Actual',
                    yaxis='y2', mode='lines+markers', marker_color='aqua'
                )

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(trace1)
                fig.add_trace(trace2,secondary_y=True)
                fig['layout'].update(height = 500, width = 1000, title = file.split('/')[-2], yaxis_title='No of vehicles passing each hour',xaxis_title='Datetime')


                fig.update_traces(marker_size=8)
                fig.update_xaxes(title_font=dict(size=16,color="red"))
                fig.update_yaxes(title_font=dict(size=16,color="green"))

                st.plotly_chart(fig, theme="streamlit", use_container_width=True)


