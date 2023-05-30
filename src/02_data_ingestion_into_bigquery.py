import pandas as pd
import numpy as np
import requests
import time
import os
import glob
import logging
from pathlib import Path
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'groovy-smithy-386015-6898ff63f9b9.json'


def upload_data_to_bigquery(dataset_id, file_name):
    '''
    this function uploads data into the Google BigQuery
    '''
#    client = bigquery.Client()

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        source_format=bigquery.SourceFormat.PARQUET,
        autodetect=True,
    )  


    table_id = dataset_id + '.' + file_name.split('/')[1]
    print(table_id)

    try :

        with open('data/{}.parquet'.format(file_name), "rb") as source_file:
            job = client.load_table_from_file(source_file, table_id, job_config=job_config)


        while job.state != 'DONE':
            time.sleep(2)
            job.reload()
            print(job.state)

        print(job.result())

        table = client.get_table(table_id)
        print(
            "Loaded {} rows and {} columns to {}".format(
                table.num_rows, len(table.schema), table_id
            )
        )    
    except:
        pass





if __name__ == '__main__':
    dataset_id_vehicle = 'groovy-smithy-386015.vehicles'
#    dataset_id_source = 'groovy-smithy-386015.source'
    file_names = sorted(["/".join(x.split('/')[1:]).split('.')[0] for x in glob.glob('data/vehicles/*.parquet')])
#    file_names_source = sorted(["/".join(x.split('/')[1:]).split('.')[0] for x in glob.glob('data/electricity_sources/*.parquet')])

#    for file_name_demand in file_names_demand:
#        upload_data_to_bigquery(dataset_id_demand, file_name_demand)
    client = bigquery.Client()
    for file_name in file_names:
        table_id = dataset_id_vehicle + '.' + file_name.split('/')[1]
        try:
            client.get_table(table_id)
        except NotFound:
            upload_data_to_bigquery(dataset_id_vehicle, file_name)
