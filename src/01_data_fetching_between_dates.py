import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import date, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data_fetching_util import GetDataFromAPI

class DataBetweenDates:

    def __init__(self, dates, api_key) -> list:
        self.dates = dates
        self.api_key = api_key


    def get_data_bw_two_dates(self):



        for date in self.dates:

            path = 'data/vehicles/vehicle_count_hourly_{}.parquet'.format(date)
            if not os.path.exists(path):


                p_vehicle = GetDataFromAPI(date, api_key)


                data_vehicle_hourly = p_vehicle.get_data_from_open_data_paris()


    #            PATH = Path('data/')


                data_vehicle_hourly_path = PATH/'vehicle_count_hourly_{}.parquet'.format(date)
                data_vehicle_hourly.to_parquet(data_vehicle_hourly_path)

if __name__ == '__main__':
    PATH = Path('data/vehicles')
    print("creating directory structure...")
    (PATH).mkdir(exist_ok=True)




    api_key = 'QRhUgmdXxbYTV8KMhgc2IYaKVUpVtJ9lqo2VKWvv'

    today_date = date.today()
    today_date = today_date.strftime("%Y-%m-%d")

    start_date = '2020-01-01'
    end_date =  '2023-05-01'


    dates = pd.date_range( pd.to_datetime(start_date), pd.to_datetime( today_date )-timedelta(days=1), freq='d')
    dates = dates.astype(str)

    p_data = DataBetweenDates(dates, api_key)
    data = p_data.get_data_bw_two_dates()


