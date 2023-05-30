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
from google.cloud import bigquery


## junctions in the city of paris
junction_labels = ['[Paris] Rivoli x Nicolas Flamel',
                    '[Paris] Amsterdam x Clichy',
                    '[Paris] Quai de Valmy',
                    '[Paris] Quai de Jemmapes',
                    '[Paris] CF892 Rivoli x Bourdonnais',
                    '[Paris] CF318 Rivoli x Lobau',
                    '[Paris] CF1 Rivoli x Sébastopol',
#                    '[Paris] CF4 Poissonnière x Montmartre EST',
                    '[Paris] CF4 Poissonnière x Montmartre OUEST']



def get_preprocessed_data():
    '''
    This function preprocesses raw data for each junction
    
    '''
    dir_path = 'data/vehicles/*.parquet'
    file_names = sorted(glob.glob(dir_path))

    df = []
    for file in file_names:
        df.append(pd.read_parquet(file))
        print(file)
    df = pd.concat(df)

    for junction in junction_labels:

        df_junction = df[df['label'] == junction].reset_index(drop=True)
        df_junction['latitude'] = df_junction['coordonnees_geo'].apply(lambda x : x[0])
        df_junction['longitude'] = df_junction['coordonnees_geo'].apply(lambda x : x[1])
        df_junction = df_junction.sort_values(by='t').reset_index(drop=True)

#        df_junction = df_junction.groupby(['label','latitude','longitude','t','mode'])['nb_usagers'].sum().reset_index()
#        df_junction = df_junction.groupby(['label','latitude','longitude','t'])[['mode','nb_usagers']].agg(list).reset_index()

        df_junction = df_junction.groupby(['label','latitude','longitude','t'])['nb_usagers'].sum().reset_index()
#        df_junction = df_junction.groupby(['label','latitude','longitude','t'])[['mode','nb_usagers']].agg(list).reset_index()


#        df_junction['mode_&_nb_usagers'] = [list(x) for x in map(zip, df_junction['mode'], df_junction['nb_usagers'])]
#        df_junction['mode_&_nb_usagers'] = df_junction['mode_&_nb_usagers'] .apply(lambda x : dict(x))


#        df_junction['2 roues motorisées'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['2 roues motorisées'] if '2 roues motorisées' in x.keys() else np.nan)
#        df_junction['Trottinettes'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Trottinettes'] if 'Trottinettes' in x.keys() else np.nan)
#        df_junction['Véhicules légers < 3,5t'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Véhicules légers < 3,5t'] if 'Véhicules légers < 3,5t' in x.keys() else np.nan)
#        df_junction['Vélos'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Vélos'] if 'Vélos' in x.keys() else np.nan)
#        df_junction['Autobus et autocars'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Autobus et autocars'] if 'Autobus et autocars' in x.keys() else np.nan)
#        df_junction['Véhicules lourds > 3,5t'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Véhicules lourds > 3,5t'] if 'Véhicules lourds > 3,5t' in x.keys() else np.nan)
#        df_junction['Trottinettes + vélos'] = df_junction['mode_&_nb_usagers'].apply(lambda x : x['Trottinettes + vélos'] if 'Trottinettes + vélos' in x.keys() else np.nan)


#        columns_of_interest = ['label', 'latitude', 'longitude', 't', '2 roues motorisées', 'Trottinettes',
#        'Véhicules légers < 3,5t', 'Vélos', 'Autobus et autocars',
#        'Véhicules lourds > 3,5t', 'Trottinettes + vélos']

#        columns_of_interest = ['label', 'latitude', 'longitude', 't','nb_usagers']
#        df_junction = df_junction[columns_of_interest]

        df_junction.to_parquet(PATH/'{}.parquet'.format(junction))
    





if __name__ == '__main__':
    PATH = Path('data/vehicles_preprocessed')
    print("creating directory structure...")
    (PATH).mkdir(exist_ok=True)
    get_preprocessed_data()



