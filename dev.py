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
for year in ['2018', '2019', '2020']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

if __name__ == '__main__':
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:15236'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:15236'
    ee.Initialize()
    satellites = ['VIIRS_Night']
    val_ids = ['24462610', '24462788', '24462753']
    test_ids = ['24461623', '24332628']
    skip_ids = ['21890069', '20777160', '20777163', '20777166']
    target_ids = ['21889672', '21889683', '21889697', '21889719', '21889734', '21889754', '21997775']
    # satellites = ['GOES']
    locations = ['August_complex', 'LNU_lighting_complex', 'SCU_lighting_complex', 'CZU_lighting_complex', 'North_complex_fire', 'Beachie_wildfire_2', 'Beachie_wildfire', 'Holiday_farm_wildfire', 'creek_fire']
    locations += ['Anonymous_fire1', 'Anonymous_fire2', 'Anonymous_fire3', 'Anonymous_fire4', 'Anonymous_fire5', 'Anonymous_fire6','Anonymous_fire7', 'Anonymous_fire7', 'Anonymous_fire8', 'Anonymous_fire10']

    # locations = ['Doctor_creek_fire']
    df = df.sort_values(by=['Id'])
    train_df = df[~df.Id.isin(val_ids + skip_ids + test_ids)]
    val_df = df[df.Id.isin(val_ids)]
    test_df = df[df.Id.isin(test_ids)]
    target_df = df[df.Id.isin(target_ids)]
    df = train_df
    ids, start_dates, end_dates, lons, lats = df['Id'].values.astype(str), df.start_date.values, df.end_date.values, df['lon'].values.astype(float), df['lat'].values.astype(float)
    train_ids = train_df['Id'].values.astype(str)
    val_ids = val_df['Id'].values.astype(str)
    test_ids = test_df['Id'].values.astype(str)

    for i, id in enumerate(ids):
        # roi = [13.38, 61.55, 15.60, 62.07]
        if id in skip_ids:
            continue
        dataset_pre = DatasetPrepareService(location=id, rectangular_size=1, latitude=lats[i], longitude=lons[i],
                                            start_time=datetime.strptime(start_dates[i], '%Y-%m-%d').date(),
                                            end_time=datetime.strptime(end_dates[i], '%Y-%m-%d').date())
        # print("Current Location:" + id)
        #
        # Visualizing and preparation work
        # map_client = dataset_pre.visualizing_images_per_day(satellites, time_dif=5)
        # dataset_pre.generate_video_for_goes()
        # dataset_pre.tif_to_png_agg(satellites)
        # dataset_pre.generate_custom_gif(satellites)

        # FIRMS progression generation (VIIRS/MODIS)
        # firmsProcessor = FirmsProcessor()
        # firmsProcessor.firms_generation_from_csv_to_tiff(config.get(location).get('start'), config.get(location).get('end'), location, 32610)
        # firmsProcessor.accumulation(location)

        # training phase
        dataset_pre.download_dataset_to_gcloud(satellites, '4326', False)
        # dataset_pre.batch_downloading_from_gclound_training(satellites, '2023-05-12')
        # preprocessing.corp_tiff_to_same_size(location, False)
        # proj1_processor.dataset_generator_proj1(location)

        # inference phase
        # dataset_pre.download_goes_dataset_to_gcloud_every_hour(False, '32610', 'GOES')
        # dataset_pre.batch_downloading_from_gclound_referencing(['GOES'])
        # preprocessing.corp_tiff_to_same_size(location, True)

        # Proj1 used functions for evaluation
        # eval.reconstruct_proj1_output(location)
        # eval.reference_proj1(location, True)
        # eval.evaluate_and_generate_images(location)
        # eval.evaluate_mIoU(location, 'Sentinel2', ['FIRMS','GOES','GOES_FIRE'], s2_date = '2020-09-08')

        # Proj2 used functions
        # proj2_processor.dataset_generator_proj2_image_test(location, file_name ='proj3_' + location + '_img.npy')
        # preprocessing.reconstruct_tif_proj2(location)
    # Proj2 used functions
    # proj2_processor.dataset_generator_proj2(locations, window_size=1)
    # proj2_processor.dataset_generator_proj2_image(locations, file_name ='proj3_all_fire_img_v3.npy')

    # Proj5 used functions
    # proj5_processor.dataset_generator_proj5_image_seqtoseq(train_ids, visualize=False, file_name='proj5_train_img_seqtoseq_alll_'+str(ts_length)+'.npy', label_name='proj5_train_label_seqtoseq_alll_'+str(ts_length)+'.npy',
    #                                                        save_path = 'data_train_proj5/', ts_length=ts_length, interval=3, image_size=(512, 512))
    # proj5_processor.dataset_generator_proj5_image_seqtoseq(val_ids, visualize=True, file_name='proj5_val_img_seqtoseql_'+str(ts_length)+'.npy', label_name='proj5_val_label_seqtoseql_'+str(ts_length)+'.npy',
    #                                                        save_path='data_val_proj5/', rs_idx=0.3, cs_idx=0.3, ts_length=ts_length, interval=3, image_size=(256, 256))
    # proj5_processor.dataset_generator_proj5_image_seqtoseq(test_ids, visualize=True, file_name='proj5_'+test_ids[0]+'_img_seqtoseql_'+str(ts_length)+'.npy', label_name='proj5_'+test_ids[0]+'_label_seqtoseql_'+str(ts_length)+'.npy',
    #                                                        save_path='data_test_proj5/', ts_length=ts_length, interval=ts_length, rs_idx=0.3, cs_idx=0.3, image_size=(256, 256))