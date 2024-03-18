import datetime
import os

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

dfs = []
for year in ['2023']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

if __name__ == '__main__':
    ee.Initialize()
    ts_length = 6
    satellites = ['VIIRS_Day', 'VIIRS_Night']
    # ids, start_dates, end_dates, lats_min, lats_max, lons_min, lons_max = df['Id'].values.astype(str), df.start_date.values, df.end_date.values, df['lats_min'].values.astype(float), df['lats_max'].values.astype(float),\
    #                                                                       df['lons_min'].values.astype(float), df['lons_max'].values.astype(float)
    # for i, id in enumerate(ids):

    #     start_date, end_date, lat_min, lat_max, lon_min, lon_max = start_dates[i], end_dates[i], lats_min[i], lats_max[i], lons_min[i], lons_max[i]
    #     print(id)
    #     roi = [lon_min, lat_min, lon_max, lat_max]
    #     dataset_pre = DatasetPrepareService(location=id, roi=roi,
    #                                         start_time=datetime.strptime(start_date, '%Y-%m-%d').date(),
    #                                         end_time=datetime.strptime(end_date, '%Y-%m-%d').date())
    #     dataset_pre.batch_downloading_from_gclound_training(satellites, '2024-03-01')
    ids = ['donnie_creek', 'slave_lake']
    proj5_processor = Proj5DatasetProcessor()
    for id in ids:
        # if id =='donnie_creek':
        #     continue
        print(id)
        # proj5_processor.dataset_generator_proj5_images(mode='val', locations=[id], visualize=True,
        #                                                 file_name='proj5_test_'+id+'_imgs.npy',
        #                                                 label_name='proj5_test_'+id+'_labels.npy',
        #                                                 save_path='data_test_proj5/', rs_idx=0.3, cs_idx=0.3,
        #                                                 image_size=(256, 256))
        proj5_processor.dataset_generator_proj5_image_seqtoseq_eva(mode = 'test', locations=[id], visualize=True, file_name='proj5_'+id+'_img_seqtoseql_'+str(ts_length)+'.npy', label_name='proj5_'+id+'_label_seqtoseql_'+str(ts_length)+'.npy',
                                                           save_path='data_test_proj5_v3/', ts_length=ts_length, interval=ts_length, rs_idx=0.3, cs_idx=0.3, image_size=(256, 256))