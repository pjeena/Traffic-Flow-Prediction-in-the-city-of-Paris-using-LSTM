
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
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#path_constant = '/Users/piyush/Desktop/dsml_Portfolio/kafka_project/'


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#my_rank = 8

def standarize_data(X_sample, y_sample, scaler ):
    X_sample =  scaler.transform(X_sample.reshape(-1,1)).reshape(X_sample.shape)
    y_sample =  scaler.transform(y_sample.flatten().reshape(-1,1)).reshape(y_sample.shape)

    return X_sample,y_sample


def model_train(X_train,y_train, X_val, y_val,num_epochs):
    model = Sequential()
    model.add(InputLayer((X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(y_train.shape[1], 'linear'))

    path_model = os.path.join('checkpoints', file.split('/')[2])
    cp = ModelCheckpoint(path_model, save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[cp],epochs=num_epochs, verbose=1)

    return model


#def get_predictions(model, X, y):
#    predictions = model.predict(X)
#    wheeler_2_predictions = predictions[:, 0 : predictions.shape[1] : 2]
#    wheeler_4_predictions = predictions[:, 1 : predictions.shape[1] : 2]

#    wheeler_2_predictions, wheeler_4_predictions = postprocess_data(wheeler_2_predictions, wheeler_4_predictions, wheeler_2_train_mean, wheeler_2_train_std, wheeler_4_train_mean, wheeler_4_train_std)
    
#    wheeler_2_actuals, wheeler_4_actuals = postprocess_data(y[:, 0 : y.shape[1] : 2], y[:, 1 : y.shape[1] : 2], wheeler_2_train_mean, wheeler_2_train_std, wheeler_4_train_mean, wheeler_4_train_std)


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
#   return wheeler_2_actuals, wheeler_4_actuals, wheeler_2_predictions, wheeler_4_predictions
    




if __name__ == '__main__':
    dir_path = 'data/vehicles_train/*/'
    file_names = sorted(glob.glob(dir_path))
#    file_names = file_names[0:1]

#    file = file_names[my_rank]
    for file in file_names:
            

            print(file)
            X = np.load(Path(file)/'X.npz')['arr_0']
            y = np.load(Path(file)/'y.npz')['arr_0']


            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=1)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=1)      
                  
            
 #           X_to_standarize = X_train[:,:,0]
            scaler = StandardScaler()
            scaler.fit(X_train[:,:,0].flatten().reshape(-1,1))

            path_preprocessor = os.path.join('preprocessor',file.split('/')[2])
            Path(path_preprocessor).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(path_preprocessor,'preprocessor.p'), 'wb') as f:
                pickle.dump(scaler, f)



            X_train, y_train = standarize_data(X_train,y_train, scaler)
            X_val, y_val = standarize_data(X_val,y_val, scaler)
            X_test, y_test = standarize_data(X_test,y_test, scaler)


            start = time.time()
            model = model_train(X_train,y_train, X_val, y_val,num_epochs=200)
            end = time.time()


            path_model = os.path.join( file.split('/')[2].replace(' ', '')  )
            model.save( os.path.join(  'models' ,  file.split('/')[2].replace(' ', '')  + '.h5'  )  )



            model = tf.keras.models.load_model(  os.path.join(  'models' ,  file.split('/')[2].replace(' ', '')  + '.h5'  )  )


            y_pred = scaler.inverse_transform( model.predict(X_test) )
            y_test = scaler.inverse_transform( y_test )



            df_y_pred = pd.DataFrame(y_pred).add_prefix('hour_ahead_3*')
            df_y_test = pd.DataFrame(y_test).add_prefix('hour_ahead_3*')


#            df_2_wheeler_actual = pd.DataFrame(wheeler_2_actuals).add_prefix('hour_3*')
#            df_4_wheeler_actual = pd.DataFrame(wheeler_4_actuals).add_prefix('hour_3*')
#            df_2_wheeler_prediction = pd.DataFrame(wheeler_2_predictions).add_prefix('hour_3*')
#            df_4_wheeler_prediction = pd.DataFrame(wheeler_4_predictions).add_prefix('hour_3*')

            print(df_y_pred.shape, df_y_test.shape)
        #    (PATH).mkdir(exist_ok=True)
            path_prediction =  os.path.join('predictions',file.split('/')[2])
            Path(path_prediction).mkdir(parents=True, exist_ok=True)

            df_y_pred.to_csv(  os.path.join(path_prediction,'df_y_pred.csv' ) )
            df_y_test.to_csv( os.path.join(path_prediction,'df_y_test.csv' ))
#            df_2_wheeler_prediction.to_csv( os.path.join(path_prediction,'df_2_wheeler_prediction.csv' ) )
#            df_4_wheeler_prediction.to_csv( os.path.join(path_prediction,'df_4_wheeler_prediction.csv' ) )

    #    hour_num = 0

    #    plt.plot(   df_2_wheeler_actual['hour_{}'.format(hour_num)][0:100]   )
    #    plt.plot(   df_2_wheeler_prediction['hour_{}'.format(hour_num)][0:100]   )

            print('Time : {}'.format(end-start)) 

