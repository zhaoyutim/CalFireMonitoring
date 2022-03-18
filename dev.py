import ee
import yaml
import numpy as np

from DataPreparation.DatasetPrepareService import DatasetPrepareService
from Evaluation.EvaluationService import EvaluationService
from FirmsProcessor.FirmsProcessor import FirmsProcessor
from LowResSatellitesService.LowResSatellitesService import LowResSatellitesService
from Preprocessing.PreprocessingService import PreprocessingService
from Visualization.VisulizationService import VisualizationService

with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    ee.Initialize()
    satellites = ['VIIRS_Day']
    # satellites = ['GOES']
    # locations = ['August_complex', 'LNU_lighting_complex', 'SCU_lighting_complex', 'CZU_lighting_complex',
    #              'North_complex_fire', 'Doctor_creek_fire', 'Beachie_wildfire', 'Holiday_farm_wildfire']
    # locations += ['Anonymous_fire1', 'Anonymous_fire2', 'Anonymous_fire3', 'Anonymous_fire4', 'Anonymous_fire5', 'Anonymous_fire6',
    #              'Anonymous_fire7', 'Anonymous_fire7', 'Anonymous_fire8', 'Anonymous_fire10']

    # locations = ['SCU_lighting_complex', 'CZU_lighting_complex']
    locations = ['swedish_fire'] #32723
    # locations = ['creek_fire']
    # locations = ['thomas_fire']
    # locations = ['elephant_hill_fire', 'camp_fire', 'fraser_complex', 'chuckegg_creek_fire']
    # locations = ['brazil_1214', 'brazil_668', 'brazil_675', 'brazil_1341', 'brazil_728']
    # locations = ['August_complex']
    # locations_16 = ['camp_fire', 'tubbs_fire', 'carr_fire']
    # locations += ['christie_mountain', 'talbott_creek', 'Doctor_creek_fire']
    # locations += ['R91947', 'R92033']
    # locations += ['R21721', 'R11498']
    # locations += ['magnum_fire', 'bighorn_fire', 'santiam_fire', 'holiday_farm_fire', 'slater_fire']
    generate_goes = False
    mode = 'viirs'
    preprocessing = PreprocessingService()
    visualization = VisualizationService()
    eval = EvaluationService()
    lowres = LowResSatellitesService()
    dataset_proj2 = []
    for location in locations:
        roi = [13.38, 61.55, 15.60, 62.07]
        dataset_pre = DatasetPrepareService(location=location)
        print("Current Location:" + location)

        # Visualizing and preparation work
        # map_client = dataset_pre.visualizing_images_per_day(satellites, time_dif=5)
        # dataset_pre.generate_video_for_goes()
        # dataset_pre.generate_custom_gif(satellites)
        # dataset_pre.evaluate_tiff_to_png(location)
        # dataset_pre.label_tiff_to_png(location)

        # label generation
        # firmsProcessor = FirmsProcessor()
        # firmsProcessor.firms_generation_from_csv_to_tiff(config.get(location).get('start'), config.get(location).get('end'), location, 32610)
        # firmsProcessor.accumulation(location)

        # training phase
        # preprocessing.corp_tiff_to_same_size(location, False)
        # dataset_pre.firms_generation_from_csv_to_tiff(False, mode, '32610')
        # dataset_pre.batch_downloading_from_gclound_training(satellites)
        # dataset_pre.download_dataset_to_gcloud(satellites, '32633', False)
        # preprocessing.dataset_generator_proj1(location)


        # lowres.fetch_imagery_from_sentinel_hub(location, ['S3'])
        # dataset_proj2.append(preprocessing.dataset_generator_proj2(location, satellites))

        # inferencing phase
        # dataset_pre.batch_downloading_from_gclound_referencing(['GOES'])
        # preprocessing.corp_tiff_to_same_size(location, True)
        # eval.evaluate_mIoU(location, 'Sentinel2', ['FIRMS','GOES','GOES_FIRE'], s2_date = '2020-09-08')
        # dataset_pre.download_goes_dataset_to_gcloud_every_hour(False, '32610', 'GOES')
        # eval.reconstruct_trial5(location)
        # eval.reference_trial5(location, True)
        # eval.evaluate_and_generate_images(location)
        preprocessing.dataset_generator_proj3_image_test(location, file_name = 'proj3_'+location+'_img.npy')
        # preprocessing.reconstruct_tif_firmproj3(location)

        # visualization
        # visualization.scatter_plot(location, 'palsar')
    # preprocessing.dataset_generator_proj3(locations, window_size=1)
    # preprocessing.dataset_generator_proj3_image(locations, file_name = 'proj3_test_img.npy')
    # preprocessing.dataset_visualization_proj3(locations)
    # dataset_output = np.concatenate(dataset_proj2, axis=0)
    # np.save('data/train/dataset_proj2.npy', dataset_output)
