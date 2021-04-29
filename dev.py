import ee
import yaml
import numpy as np

from DataPreparation.DatasetPrepareService import DatasetPrepareService
from Evaluation.EvaluationService import EvaluationService
from FirmsProcessor.FirmsProcessor import FirmsProcessor
from LowResSatellitesService.LowResSatellitesService import LowResSatellitesService
from Preprocessing.PreprocessingService import PreprocessingService


with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    ee.Initialize()
    satellites = ['S3']
    # locations = ['August_complex', 'creek_fire','LNU_lighting_complex', 'SCU_lighting_complex', 'CZU_lighting_complex', 'North_complex_fire']
    # locations = ['SCU_lighting_complex', 'CZU_lighting_complex']
    # locations = ['brazil_fire'] #32723
    # locations = ['creek_fire']
    locations=['august']
    # locations = ['brazil_1214', 'brazil_668', 'brazil_675', 'brazil_1341', 'brazil_728']
    # locations = ['August_complex']
    # locations = ['beachie_creek']
    # locations_16 = ['camp_fire', 'tubbs_fire', 'carr_fire']
    # locations += ['christie_mountain', 'talbott_creek', 'Doctor_creek_fire']
    # locations = ['R91947', 'R92033', 'VA1787', 'R12068']
    # locations = ['G41607', 'G80340', 'G82215', 'R21721', 'R11498']
    # locations = ['magnum_fire', 'bighorn_fire', 'santiam_fire', 'holiday_farm_fire', 'slater_fire']
    generate_goes = False
    mode = 'viirs'
    preprocessing = PreprocessingService()
    eval = EvaluationService()
    lowres = LowResSatellitesService()
    dataset_proj2 = []
    for location in locations:
        # dataset_pre = DatasetPrepareService(location=location)
        print("Current Location:" + location)

        # Visualizing and preparation work
        # map_client = dataset_pre.visualizing_images_per_day(satellites)
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
        # dataset_pre.download_dataset_to_gcloud(satellites, '32610', False)
        # preprocessing.dataset_generator_proj1(location)
        lowres.fetch_imagery_from_sentinel_hub(location, satellites)
        # dataset_proj2.append(preprocessing.dataset_generator_proj2(location, satellites))

        # referencing phase
        # dataset_pre.batch_downloading_from_gclound_referencing(['GOES'])
        # preprocessing.corp_tiff_to_same_size(location, True)
        # eval.evaluate_mIoU(location, 'Sentinel2', ['FIRMS','GOES','GOES_FIRE'], s2_date = '2020-09-08')
        # dataset_pre.download_goes_dataset_to_gcloud_every_hour(False, '32610')
        # eval.reconstruct_trial5(location)
        # eval.reference_trial5(location, False)
        # eval.evaluate_and_generate_images(location)

    # dataset_output = np.concatenate(dataset_proj2, axis=0)
    # np.save('data/train/dataset_proj2.npy', dataset_output)
