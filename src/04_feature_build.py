import pandas as pd
import numpy as np
import requests
import time
import os
import json
import glob
from datetime import date, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

## junctions in the city of paris
junction_labels = ['[Paris] Rivoli x Nicolas Flamel',
                    '[Paris] Amsterdam x Clichy',
                    '[Paris] Quai de Valmy',
                    '[Paris] Quai de Jemmapes',
                    '[Paris] CF892 Rivoli x Bourdonnais',
                    '[Paris] CF318 Rivoli x Lobau',
                    '[Paris] CF1 Rivoli x Sébastopol',
                    '[Paris] CF4 Poissonnière x Montmartre EST',
                    '[Paris] CF4 Poissonnière x Montmartre OUEST']


def build_feature(window_lag, window_lead):

    dir_path = 'data/vehicles_preprocessed/*.parquet'
    file_names = sorted(glob.glob(dir_path))

    for file in file_names:
        df = pd.read_parquet(file)

        df = df.fillna(0.0)

        df['t'] = pd.to_datetime(df['t'])
        df = df.set_index('t')
        index_dates = pd.date_range(start = df.index.min(), end = df.index.max() ,freq='H', tz='UTC')
        df = df.reindex(index_dates)
        
        df = df.groupby(df.index.time).ffill()


#        df['2_wheelers'] = df['2 roues motorisées'] + df['Trottinettes'] + df['Trottinettes + vélos'] + df['Vélos']


#        df['4_wheelers'] = df['Véhicules légers < 3,5t'] + df['Véhicules lourds > 3,5t'] + df['Autobus et autocars']







#       df = df.drop(df.columns[4:11],axis = 1)

#        df['t'] = pd.to_datetime(df['t'])

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

        df_original = df.copy()   # to access labels for training


    #      window = 6  # no of past time steps considered
        df_lagged_input = pd.DataFrame()
        print('Lagged features generation')
        for i in range(1,window_lag+1):
            print('time step : t[{}]'.format(-i))
            df_lagged_input = pd.concat([df_lagged_input, df_original.shift(i).add_suffix('_shift_[{}]'.format(-i))], axis =1)
#        df_lagged = df_lagged.dropna()


#        X = df_lagged.values
#        X = X.reshape(X.shape[0], window,df_original.shape[1])



#        df_output = df_original[['2_wheelers', '4_wheelers']]
        df_output = pd.DataFrame(df_original['nb_usagers'])
#        df_lagged_output = df_output.copy()
        df_lagged_output = pd.DataFrame()
        print('Lead features generation')
        for i in range(1,window_lead+1):
                print('time step : t[{}]'.format(i))
                df_lagged_output = pd.concat([df_lagged_output, df_output.shift(-i).add_suffix('_shift_[{}]'.format(i))], axis =1)



        df_combined = pd.concat([df_lagged_input,df_lagged_output],axis=1)

        df_combined = df_combined.dropna()

    #    y = y[window:]

        X = df_combined.iloc[:,0 : window_lag * df_original.shape[1]].values
        y = df_combined.iloc[:, window_lag * df_original.shape[1]:].values


        X = X.reshape(X.shape[0], window_lag,df_original.shape[1])
        y = y
    #      X.to_parquet(PATH/file.split('/')[2].split('.')[0]/'X.parquet')
    #      y.to_parquet(PATH/file.split('/')[2].split('.')[0]/'y.parquet')


        path_sep = PATH/file.split('/')[2].split('.')[0]
        (path_sep).mkdir(exist_ok=True)
        np.savez_compressed(path_sep/'X.npz',X)
        np.savez_compressed(path_sep/'y.npz',y)





if __name__ == '__main__':
    PATH = Path('data/vehicles_train')
    print("creating directory structure...")
#    (PATH).mkdir(exist_ok=True)

    build_feature(window_lag=24, window_lead=3)
