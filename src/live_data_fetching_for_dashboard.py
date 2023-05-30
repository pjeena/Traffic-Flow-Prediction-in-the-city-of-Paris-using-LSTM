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
from datetime import date, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import tensorflow as tf
from src.data_fetching_util import GetDataFromAPI
from src.utils import preprocess_fetched_data_from_API, feature_build_from_preprocessed_data, standarize_data
from tensorflow.keras.models import load_model


def define_constant_attributes():

        junction_labels = ['[Paris] Rivoli x Nicolas Flamel',
                            '[Paris] Amsterdam x Clichy',
                            '[Paris] Quai de Valmy',
                            '[Paris] Quai de Jemmapes',
                            '[Paris] CF892 Rivoli x Bourdonnais',
                            '[Paris] CF318 Rivoli x Lobau',
                            '[Paris] CF1 Rivoli x Sébastopol',
   #                         '[Paris] CF4 Poissonnière x Montmartre EST',
                            '[Paris] CF4 Poissonnière x Montmartre OUEST']

        window_lag, window_lead = 24,3
        num_components_in_one_time_step = 7
        today_date = (date.today()).strftime("%Y-%m-%d")
        week_ago_date = (pd.to_datetime( today_date )-timedelta(days=10)).strftime("%Y-%m-%d")
        dates = pd.date_range( pd.to_datetime(week_ago_date), pd.to_datetime( today_date ) + timedelta(1) , freq='d').astype(str)

        current_hour = pd.to_datetime(pd.Timestamp.now(tz='UTC').to_pydatetime()).floor("H")
        #current_hour = current_hour + timedelta(hours = 10)
        past_week_hour = current_hour - timedelta(days=7) - timedelta(hours=1)

        return junction_labels, window_lag, window_lead, num_components_in_one_time_step, dates, current_hour, past_week_hour





def get_live_raw_data(dates):
        dir_data = os.path.join('output', 'raw_data')
        CHECK_FOLDER = os.path.isdir(dir_data)
        if not CHECK_FOLDER:
                    os.makedirs(dir_data)

        
        for date in dates:
                get_data_class = GetDataFromAPI(date,api_key=None)
                data_vehicle_hourly = get_data_class.get_data_from_open_data_paris()

                path_to_data_file = os.path.join(dir_data,'vehicle_count_hourly_' +  date + '.parquet')
                data_vehicle_hourly.to_parquet(path_to_data_file)

                
                
get_live_raw_data(dates)
