import ee
from DataPreparation.DatasetPrepareService import DatasetPrepareService
from Evaluation.EvaluationService import EvaluationService
from Preprocessing.PreprocessingService import PreprocessingService

ee.Initialize()

if __name__ == '__main__':
    satellites = ['Sentinel1_asc']
    # locations = ['August_complex', 'creek_fire','LNU_lighting_complex', 'SCU_lighting_complex', 'CZU_lighting_complex']
    # locations = ['SCU_lighting_complex', 'CZU_lighting_complex']
    # locations = ['brazil_fire'] #32723
    locations_16 = ['camp_fire', 'tubbs_fire', 'carr_fire']
    locations_17 = ['magee_fire', 'black_angus_creek_wildfire', 'eagle_bluff_fire', 'tagish_lake_fire', 'richter_creek_fire', 'richter_mountain_fire']
    locations = ['N31179']
    # locations = ['magnum_fire', 'bighorn_fire', 'santiam_fire', 'holiday_farm_fire', 'slater_fire']
    generate_goes = False
    mode = 'viirs'
    preprocessing = PreprocessingService()
    eval = EvaluationService()
    for location in locations:
        dataset_pre = DatasetPrepareService(location=location)
        print("Current Location:" + location)

        # Visualizing and preparation work
        # map_client = dataset_pre.visualizing_images_per_day(satellites)
        # dataset_pre.generate_video_for_goes()
        # dataset_pre.generate_custom_gif(satellites)
        # dataset_pre.evaluate_tiff_to_png(location)
        # dataset_pre.label_tiff_to_png(location)

        # training phase
        # preprocessing.corp_tiff_to_same_size(location, False)
        # dataset_pre.firms_generation_from_csv_to_tiff(False, mode, '32610')
        # dataset_pre.batch_downloading_from_gclound_training(['Sentinel2'])
        # dataset_pre.download_dataset_to_gcloud(satellites, '4326', True)
        # preprocessing.dataset_generator_trial5(location)

        # referencing phase
        # dataset_pre.batch_downloading_from_gclound_referencing(['GOES'])
        # preprocessing.corp_tiff_to_same_size(location, True)
        # eval.evaluate_mIoU(location, 'Sentinel2', ['FIRMS','GOES','GOES_FIRE'])
        # dataset_pre.download_goes_dataset_to_gcloud_every_hour(False, '32610')
        # eval.reconstruct_trial5(location）
        eval.reference_trial5(location, True)