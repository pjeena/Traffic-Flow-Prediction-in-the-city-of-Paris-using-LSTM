
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


## defining important elements required for prediction
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




def get_preprocessed_raw_data(dates,junction_labels):

        dir_data = os.path.join('output', 'raw_data' , '*.parquet')
        file_names = sorted(glob.glob(dir_data))
        df = []
        for file in file_names:
                df.append(pd.read_parquet(file))
                print(file)
        df = pd.concat(df)



        dir_preprocessed_data = os.path.join('output', 'preprocessed')
        CHECK_FOLDER = os.path.isdir(dir_preprocessed_data)
        if not CHECK_FOLDER:
                    os.makedirs(dir_preprocessed_data)

        for junction in junction_labels:
                print(junction)
                df_junction = preprocess_fetched_data_from_API(df,junction)
                if not df_junction.empty:
                        df_junction['t'] = pd.to_datetime(df_junction['t'])
                        df_junction = df_junction.set_index('t')
                        index_dates = pd.date_range(start = dates[0], end = dates[-1] ,freq='H', tz='UTC').delete(-1)
                        df_junction = df_junction.reindex(index_dates) 
                        df_junction = df_junction.groupby(df_junction.index.time).ffill()
                        print(index_dates.shape, df_junction.shape)
                
                
                        path_to_preprocessed_data_file = os.path.join( dir_preprocessed_data,  junction + '.parquet')
                        print(path_to_preprocessed_data_file)
                        df_junction.to_parquet(path_to_preprocessed_data_file)





def build_features(window_lag, window_lead):
        dir_preprocessed_data = os.path.join('output', 'preprocessed',  '*.parquet')
        file_names = sorted(glob.glob(dir_preprocessed_data))


        dir_test_data = os.path.join('output', 'input_data_to_model')
        CHECK_FOLDER = os.path.isdir(dir_test_data)
        if not CHECK_FOLDER:
                    os.makedirs(dir_test_data)

        for file in file_names:
                df = pd.read_parquet(file)
                df_test_set = feature_build_from_preprocessed_data(df,window_lag, window_lead)


                path_to_test_data_file = os.path.join( dir_test_data,  '_test_'  + file.split('/')[2].split('.')[0]  )
                print(path_to_test_data_file)
                df_test_set.to_parquet(path_to_test_data_file + '.parquet')



@st.cache_data()
def get_predictions(current_hour, past_week_hour, window_lag, num_components_in_one_time_step):
        dir_input_to_model_data = os.path.join('output', 'input_data_to_model',  '*.parquet')
        file_names = sorted(glob.glob(dir_input_to_model_data))


        dir_predictions = os.path.join('output', 'predictions')
        CHECK_FOLDER = os.path.isdir(dir_predictions)
        if not CHECK_FOLDER:
                    os.makedirs(dir_predictions)



        for file in file_names:
                print(file)
                df_test = pd.read_parquet(file)
                df_test_upto_current_hour = df_test[(df_test.index < current_hour) & (df_test.index >= past_week_hour)]
                save_index = df_test_upto_current_hour.index  # save index for later use


                X_test = df_test_upto_current_hour.iloc[
                :, 0 : window_lag * num_components_in_one_time_step
                ].values.reshape(
                df_test_upto_current_hour.shape[0], window_lag, num_components_in_one_time_step
                )



                y_test = df_test_upto_current_hour.iloc[
                :, window_lag * num_components_in_one_time_step :
                ].values


                scaler = pickle.load(open(os.path.join ( 'preprocessor' , file.split('.')[0].split('_test_')[1] ,  'preprocessor.p' ), 'rb'))
                model = tf.keras.models.load_model(  os.path.join('models', file.split('.')[0].split('_test_')[1].replace(' ', '') + '.h5'  )  )


                X_test, y_test = standarize_data(X_test,y_test, scaler=scaler)


                y_pred = scaler.inverse_transform( model.predict(X_test) )
                y_test = scaler.inverse_transform( y_test )


                df_y_pred = pd.DataFrame(y_pred).add_prefix('hour_ahead_3*').round().set_index(save_index).reset_index(names="datetime")
                df_y_test = pd.DataFrame(y_test).add_prefix('hour_ahead_3*').round().set_index(save_index).reset_index(names='datetime')



                df_y_pred['datetime'] = pd.to_datetime(df_y_pred['datetime'])
                index_mod =   list( df_y_pred['datetime'] + timedelta(hours=1) )  + [ df_y_pred['datetime'].iloc[-1] + timedelta(hours=2) ,df_y_pred['datetime'].iloc[-1] + timedelta(hours=3) ]
                values_mod =  list(df_y_pred['hour_ahead_3*0'].values) +  [ df_y_pred['hour_ahead_3*1'].values[-1],  df_y_pred['hour_ahead_3*2'].values[-1]  ]
                df_pred = pd.DataFrame(list(zip(index_mod, values_mod)),
                        columns =['datetime', 'no_of_vehicles'])


                df_y_test['datetime'] = pd.to_datetime(df_y_test['datetime'])
                index_mod =   list( df_y_test['datetime'] + timedelta(hours=1) )  + [ df_y_test['datetime'].iloc[-1] + timedelta(hours=2) ,df_y_test['datetime'].iloc[-1] + timedelta(hours=3) ]
                values_mod =  list(df_y_test['hour_ahead_3*0'].values) +  [ df_y_test['hour_ahead_3*1'].values[-1],  df_y_test['hour_ahead_3*2'].values[-1]  ]
                df_test = pd.DataFrame(list(zip(index_mod, values_mod)),
                        columns =['datetime', 'no_of_vehicles'])



                df_pred['type'] = 'Prediction'
                df_test['type'] = 'Actual'
                df_pred_and_test = pd.concat([df_pred, df_test])


                print(df_pred.shape, df_test.shape)
                path_prediction =  os.path.join('output', 'predictions', file.split('.')[0].split('_test_')[1])
                Path(path_prediction).mkdir(parents=True, exist_ok=True)
        #        df_pred.to_csv(  os.path.join(path_prediction,'df_pred.csv' ) )
        #       df_test.to_csv( os.path.join(path_prediction,'df_test.csv' ))
                df_pred_and_test.to_csv(  os.path.join(path_prediction,'df_pred_and_test.csv' ) )




