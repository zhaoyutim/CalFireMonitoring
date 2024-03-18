import datetime
import os
import glob
import ee
import pandas as pd
import yaml
from datetime import datetime
from DataPreparation.DatasetPrepareService import DatasetPrepareService
from Evaluation.EvaluationService import EvaluationService
from Preprocessing.PreprocessingService import PreprocessingService
from Preprocessing.Proj1DatesetProcessor import Proj1DatasetProcessor
from Preprocessing.Proj2DatesetProcessor import Proj2DatasetProcessor
from Preprocessing.Proj5DatasetProcessor import Proj5DatasetProcessor

with open("config/rcm_config.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:15236'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:15236'
    ee.Initialize()
    # satellites = ['VIIRS_Day', 'VIIRS_Night']
    satellites = ['Sentinel2']
    # ids = ['slave_lake', 'slave_lake3', 'slave_lake4', 'slave_lake5', 'edmonton', 'rainbow_lake', 'fox_lake', 'donnie_creek']
    ids=['donnie_creek']

    for i, id in enumerate(ids):
        roi = config[id]['roi']
        print(id)
        dataset_pre = DatasetPrepareService(location=id, rectangular_size=1, roi=roi,
                                            start_time=config[id]['start'],
                                            end_time=config[id]['end'])

        # training phase
        # dataset_pre.download_dataset_to_gcloud(satellites, '32618', False)
        dataset_pre.batch_downloading_from_gclound_training(satellites, '2024-02-07')

        