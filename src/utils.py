import pandas as pd
import numpy as np
import requests
import time
import os
import glob
from datetime import date, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


def preprocess_fetched_data_from_API(df : pd.DataFrame, junction : str) -> pd.DataFrame:
        '''
        -> This function extracts the data for a particular traffic junction
        -> It also calculates the no_of_vehicles for a particular type such as motorcycles, velos, big vehicles etc
        -> Finally it return the modified dataframe with relevant columns
        '''

        df_junction = df[df['label'] == junction].reset_index(drop=True)
        df_junction['latitude'] = df_junction['coordonnees_geo'].apply(lambda x : x[0])
        df_junction['longitude'] = df_junction['coordonnees_geo'].apply(lambda x : x[1])
        df_junction = df_junction.sort_values(by='t').reset_index(drop=True)

        df_junction = df_junction.groupby(['label','latitude','longitude','t'])['nb_usagers'].sum().reset_index()


 #       df_junction = df_junction.groupby(['label','latitude','longitude','t','mode'])['nb_usagers'].sum().reset_index()
 #       df_junction = df_junction.groupby(['label','latitude','longitude','t'])[['mode','nb_usagers']].agg(list).reset_index()

 #       df_junction['mode_&_nb_usagers'] = [list(x) for x in map(zip, df_junction['mode'], df_junction['nb_usagers'])]
 #       df_junction['mode_&_nb_usagers'] = df_junction['mode_&_nb_usagers'] .apply(lambda x : dict(x))


 #       df_junction['2 roues motorisées'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['2 roues motorisées'] if '2 roues motorisées' in x.keys() else np.nan)
 #       df_junction['Trottinettes'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Trottinettes'] if 'Trottinettes' in x.keys() else np.nan)
 #       df_junction['Véhicules légers < 3,5t'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Véhicules légers < 3,5t'] if 'Véhicules légers < 3,5t' in x.keys() else np.nan)
 #       df_junction['Vélos'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Vélos'] if 'Vélos' in x.keys() else np.nan)
 #       df_junction['Autobus et autocars'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Autobus et autocars'] if 'Autobus et autocars' in x.keys() else np.nan)
 #       df_junction['Véhicules lourds > 3,5t'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Véhicules lourds > 3,5t'] if 'Véhicules lourds > 3,5t' in x.keys() else np.nan)
 #       df_junction['Trottinettes + vélos'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Trottinettes + vélos'] if 'Trottinettes + vélos' in x.keys() else np.nan)

 #       columns_of_interest = ['label', 'latitude', 'longitude', 't', '2 roues motorisées', 'Trottinettes',
 #       'Véhicules légers < 3,5t', 'Vélos', 'Autobus et autocars',
 #       'Véhicules lourds > 3,5t', 'Trottinettes + vélos']
 #       df_junction = df_junction[columns_of_interest]

        return df_junction




def feature_build_from_preprocessed_data(df : pd.DataFrame, window_lag, window_lead) -> pd.DataFrame :
        '''
        -> This function builds time based features using sin and cos 
        -> It also groups all the vehicles count into two categories : 2 wheeler and 4 wheeler since thats what we want to forecast  
        -> Finally it returns features 
        '''

        df = df.fillna(0.0)

 #       df['2_wheelers'] = df['2 roues motorisées'] + df['Trottinettes'] + df['Trottinettes + vélos'] + df['Vélos']
 #       df['4_wheelers'] = df['Véhicules légers < 3,5t'] + df['Véhicules lourds > 3,5t'] + df['Autobus et autocars']

 #       df = df.drop(df.columns[4:11],axis = 1)
 #       df['t'] = pd.to_datetime(df['t'])

        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day'] = df.index.day

        df['sin_hour'] =  np.sin(2 * np.pi * df['hour']/ df['hour'].max())
        df['cos_hour'] =  np.cos(2 * np.pi * df['hour']/ df['hour'].max())

        df['sin_month'] =  np.sin(2 * np.pi * df['month']/ df['month'].max())
        df['cos_month'] =  np.cos(2 * np.pi * df['month']/ df['month'].max())


        df['sin_day'] =  np.sin(2 * np.pi * df['day']/ df['day'].max())
        df['cos_day'] =  np.cos(2 * np.pi * df['day']/ df['day'].max())



#        df = df.set_index('t')
        df = df.drop(['label'	,'latitude',	'longitude', 'hour', 'month', 'day'],axis=1)


############################ Test data input creation################
        df_original = df.copy()   # to access labels for training
        df_lagged_input = pd.DataFrame()
        print('Lagged features generation')
        for i in range(1,window_lag+1):
            print('time step : t[{}]'.format(-i))
            df_lagged_input = pd.concat([df_lagged_input, df_original.shift(i).add_suffix('_shift_[{}]'.format(-i))], axis =1)


############################ Test data ouptut creation to validate results ################
        df_output = pd.DataFrame(df_original['nb_usagers'])
        df_lagged_output = pd.DataFrame()
        print('Lead features generation')
        for i in range(1,window_lead+1):
                print('time step : t[{}]'.format(i))
                df_lagged_output = pd.concat([df_lagged_output, df_output.shift(-i).add_suffix('_shift_[{}]'.format(i))], axis =1)



        df_combined = pd.concat([df_lagged_input,df_lagged_output],axis=1)
        df_combined = df_combined.dropna()

        return df_combined












def standarize_data(X_sample, y_sample, scaler ):
    X_sample =  scaler.transform(X_sample.reshape(-1,1)).reshape(X_sample.shape)
    y_sample =  scaler.transform(y_sample.flatten().reshape(-1,1)).reshape(y_sample.shape)

    return X_sample,y_sample















#def model_train(X_train,y_train, X_val, y_val,num_epochs):
#        model = Sequential()
#        model.add(InputLayer((X_train.shape[1], X_train.shape[2])))
#        model.add(LSTM(32, return_sequences=True))
#        model.add(LSTM(64))
#        model.add(Dense(8, 'relu'))
#        model.add(Dense(y_train.shape[1], 'linear'))

#        path_model = os.path.join('checkpoints', file.split('/')[2])
#        cp = ModelCheckpoint(path_model, save_best_only=True)
#        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

#        model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=num_epochs, verbose=1)

#        return model






#def get_predictions(model, X, y, preprocessor_obj):
#        predictions = model.predict(X)
#        num_wheeler_2_predictions = predictions[:, 0 : predictions.shape[1] : 2]
#        num_wheeler_4_predictions = predictions[:, 1 : predictions.shape[1] : 2]

#        wheeler_2_train_mean, wheeler_2_train_std, wheeler_4_train_mean, wheeler_4_train_std = preprocessor_obj   
        
#        num_wheeler_2_predictions, num_wheeler_4_predictions = revert_standarize_data(num_wheeler_2_predictions, num_wheeler_4_predictions, wheeler_2_train_mean, wheeler_2_train_std, wheeler_4_train_mean, wheeler_4_train_std)
#        num_wheeler_2_actuals, num_wheeler_4_actuals = revert_standarize_data(y[:, 0 : y.shape[1] : 2], y[:, 1 : y.shape[1] : 2], wheeler_2_train_mean, wheeler_2_train_std, wheeler_4_train_mean, wheeler_4_train_std)

        #    df_2_wheeler = 
        #    df = pd.DataFrame(data={'Temperature Predictions': temp_preds,
        #                            'Temperature Actuals':temp_actuals,
        #                            'Pressure Predictions': p_preds,
        #                            'Pressure Actuals': p_actuals
        #                            })
        #    plt.plot(df['Temperature Predictions'][start:end])
        #    plt.plot(df['Temperature Actuals'][start:end])
        #plt.plot(df['Pressure Predictions'][start:end])
        #plt.plot(df['Pressure Actuals'][start:end])
 #       return num_wheeler_2_actuals, num_wheeler_4_actuals, num_wheeler_2_predictions, num_wheeler_4_predictions
    
