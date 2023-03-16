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

year = '2020'
filename = 'roi/us_fire_' + year + '_out.csv'
df = pd.read_csv(filename)

if __name__ == '__main__':
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:15236'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:15236'
    ee.Initialize()
    satellites = ['VIIRS_Day']
    val_ids = ['24462610', '24462788', '24462753']
    # satellites = ['GOES']
    # locations = ['August_complex', 'LNU_lighting_complex', 'SCU_lighting_complex', 'CZU_lighting_complex',
    #              'North_complex_fire', 'Beachie_wildfire_2', 'Beachie_wildfire', 'Holiday_farm_wildfire', 'creek_fire']
    # locations += ['Anonymous_fire1', 'Anonymous_fire2', 'Anonymous_fire3', 'Anonymous_fire4', 'Anonymous_fire5', 'Anonymous_fire6',
    #              'Anonymous_fire7', 'Anonymous_fire7', 'Anonymous_fire8', 'Anonymous_fire10']

    # locations = ['Doctor_creek_fire']
    df = df.sort_values(by=['Id'])
    ids, start_dates, end_dates, lons, lats = df['Id'].values.astype(str), df.start_date.values, df.end_date.values, df['lon'].values.astype(float), df['lat'].values.astype(float)
    train_ids = [id for id in ids if id not in val_ids]
    # locations = ['mosquito_fire'] #32723
    # sydney_fire 32756
    # locations = ['double_creek_fire'] #32723
    # generate_goes = False
    # mode = 'viirs'
    # preprocessing = PreprocessingService()
    # eval = EvaluationService()
    # dataset_proj2 = []
    proj5_processor = Proj5DatasetProcessor()
    # proj2_processor = Proj2DatasetProcessor()
    # proj1_processor = Proj1DatasetProcessor()
    # for i, id in enumerate(ids):
    #     roi = [13.38, 61.55, 15.60, 62.07]
    #     dataset_pre = DatasetPrepareService(location=id, rectangular_size=1, latitude=lats[i], longitude=lons[i],
    #                                         start_time=datetime.strptime(start_dates[i], '%Y-%m-%d').date(),
    #                                         end_time=datetime.strptime(end_dates[i], '%Y-%m-%d').date())
    #     print("Current Location:" + id)

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
        # dataset_pre.download_dataset_to_gcloud(satellites, '4326', False)
        # dataset_pre.batch_downloading_from_gclound_training(satellites, '2023-03-13')
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
    # proj2_processor.dataset_generator_proj2_image(locations, file_name ='proj3_mosquito_fire_img.npy')

    # Proj5 used functions
    # proj5_processor.dataset_generator_proj5_image_seqtoseq(train_ids, visualize=False, file_name='proj5_train_img_seqtoseq.npy', label_name='proj5_train_label_seqtoseq.npy', image_size=(512, 512))
    # proj5_processor.dataset_generator_proj5_image_seqtoseq(val_ids, visualize=False, file_name='proj5_val_img_seqtoseq.npy', label_name='proj5_val_label_seqtoseq.npy', image_size=(512, 512))

    proj5_processor.dataset_generator_proj5_image_seqtoone(train_ids, visualize=False, file_name='proj5_train_img_seqtoone.npy', label_name='proj5_train_label_seqtoone.npy', ts_length=10, interval=3,  image_size=(512, 512))
    proj5_processor.dataset_generator_proj5_image_seqtoone(val_ids, visualize=False, file_name='proj5_val_img_seqtoone.npy', label_name='proj5_val_label_seqtoone.npy', ts_length=10, interval=3, image_size=(512, 512))
