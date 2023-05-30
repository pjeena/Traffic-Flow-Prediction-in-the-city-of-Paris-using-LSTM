


import streamlit as st
import os 
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model

dir_input_to_model_data = os.path.join('output', 'input_data_to_model',  '*.parquet')
file_names = sorted(glob.glob(dir_input_to_model_data))

st.write(file_names)    

for file in file_names:
        model = tf.keras.models.load_model(  os.path.join('models', file.split('.')[0].split('_test_')[1].replace(' ', '') + '.h5'  )  )
        
        
st.write(model)
