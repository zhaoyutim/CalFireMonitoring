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

with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
dfs = []
for year in ['2018','2019','2020']:
# for year in ['2021']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

dfs_test = []
for year in ['2021']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df_test = pd.read_csv(filename)
    dfs_test.append(df_test)
df_test = pd.concat(dfs_test, ignore_index=True)

if __name__ == '__main__':
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:15236'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:15236'
    ee.Initialize()
    satellites = ['VIIRS_Day']
    val_ids = ['24462610', '24462788', '24462753']
    test_ids = ['24461623', '24332628']
    skip_ids = ['21890069', '20777160', '20777163', '20777166']
    target_ids = ['21889672', '21889683', '21889697', '21889719', '21889734', '21889754', '21997775']

    df = df.sort_values(by=['Id'])
    df['Id'] = df['Id'].astype(str)
    train_df = df[~df.Id.isin(val_ids + skip_ids + test_ids)]
    val_df = df[df.Id.isin(val_ids)]
    test_df = df[df.Id.isin(test_ids)]
    target_df = df[df.Id.isin(target_ids)]

    train_ids = train_df['Id'].values.astype(str)
    val_ids = val_df['Id'].values.astype(str)

    df_test = df_test.sort_values(by=['Id'])
    test_ids = df_test['Id'].values.astype(str)

    # print(ts_length)
    ts_length = 6
    interval = 3
    proj5_processor = Proj5DatasetProcessor()
    # proj2_processor = Proj2DatasetProcessor()
    # Proj2 used functions
    # proj2_processor.dataset_generator_proj2(locations, window_size=1)
    # proj2_processor.dataset_generator_proj2_image(locations, file_name ='proj3_all_fire_img_v3.npy')

    # Proj5 used functions
    # proj5_processor.dataset_generator_proj5_images(mode='train', locations=train_ids, visualize=True,
    #                                                        file_name='proj5_train_imgs.npy',
    #                                                        label_name='proj5_train_labels.npy',
    #                                                        save_path='data_train_proj5/', rs_idx=0.3, cs_idx=0.3,
    #                                                        image_size=(256, 256))
    # proj5_processor.dataset_generator_proj5_images(mode='val', locations=val_ids, visualize=True,
    #                                                        file_name='proj5_val_imgs.npy',
    #                                                        label_name='proj5_val_labels.npy',
    #                                                        save_path='data_val_proj5/', rs_idx=0.3, cs_idx=0.3,
    #                                                        image_size=(256, 256))
    proj5_processor.dataset_generator_proj5_image_seqtoseq(mode='train', locations=train_ids, visualize=True, file_name='proj5_train_img_seqtoseq_alll_'+str(ts_length)+'.npy', label_name='proj5_train_label_seqtoseq_alll_'+str(ts_length)+'.npy',
                                                        save_path = 'data_train_proj5/', ts_length=ts_length, interval=interval, image_size=(256, 256))
    proj5_processor.dataset_generator_proj5_image_seqtoseq(mode='val', locations=val_ids, visualize=True, file_name='proj5_val_img_seqtoseql_'+str(ts_length)+'.npy', label_name='proj5_val_label_seqtoseql_'+str(ts_length)+'.npy',
                                                        save_path='data_val_proj5/', rs_idx=0.3, cs_idx=0.3, ts_length=ts_length, interval=interval, image_size=(256, 256))
    # for id in test_ids:
    #     print(id)
    #     # proj5_processor.dataset_generator_proj5_images(mode='val', locations=[id], visualize=True,
    #     #                                                 file_name='proj5_test_'+id+'_imgs.npy',
    #     #                                                 label_name='proj5_test_'+id+'_labels.npy',
    #     #                                                 save_path='data_test_proj5/', rs_idx=0.3, cs_idx=0.3,
    #     #                                                 image_size=(256, 256))
    #     proj5_processor.dataset_generator_proj5_image_seqtoseq(mode = 'val', locations=[id], visualize=True, file_name='proj5_'+id+'_img_seqtoseql_'+str(ts_length)+'.npy', label_name='proj5_'+id+'_label_seqtoseql_'+str(ts_length)+'.npy',
    #                                                     save_path='data_test_proj5/', ts_length=ts_length, interval=ts_length, rs_idx=0.3, cs_idx=0.3, image_size=(256, 256))