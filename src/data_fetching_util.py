import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')




class GetDataFromAPI():

    def __init__(self, date, api_key):
        self.date = date
        self.api_key = api_key


    def get_data_from_open_data_paris(self):
        '''
        https://opendata.paris.fr/explore/dataset/comptage-multimodal-comptages/
        '''

        # getting first 5000 rows since its the limit set by API
        url = 'https://opendata.paris.fr/api/records/1.0/search/?dataset=comptage-multimodal-comptages&q=t:[{}T00:00:00Z+TO+{}T23:59:59Z]&rows=10000&facet=label&facet=t&facet=mode&facet=voie&facet=sens&facet=trajectoire'.format(self.date,self.date)
        response = requests.get(url)
        data = json.loads(response.content)['records']


        records = []
        for record in data:
            records.append(record['fields'])

        df = pd.DataFrame(records)
        return df
    

if __name__ == '__main__':
    PATH = Path('data/')
    print("creating directory structure...")
    (PATH).mkdir(exist_ok=True)

    date = '2023-05-17'
    api_key = ''
    get_data_class = GetDataFromAPI(date, api_key)
    data = get_data_class.get_data_from_open_data_paris()
#    data['metric_and_value'] = data['metric_and_value'].astype(str)  ## error in saving to parquet version in the newer version so need to convert this column to str

    data_path = PATH/'vehichle_count_hourly_{}.parquet'.format(date)
    data.to_parquet(data_path)

