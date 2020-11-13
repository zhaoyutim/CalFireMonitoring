import ee
from DataPreparation.DatasetPrepareService import DatasetPrepareService
from Evaluation.EvaluationService import EvaluationService
from Preprocessing.PreprocessingService import PreprocessingService

ee.Initialize()

if __name__ == '__main__':
    satellites = ['GOES']
    # locations = ['August_complex', 'creek_fire','LNU_lighting_complex', 'SCU_lighting_complex', 'CZU_lighting_complex']
    # locations = ['SCU_lighting_complex', 'CZU_lighting_complex']
    locations = ['creek_fire']
    generate_goes = False
    mode = 'viirs'
    preprocessing = PreprocessingService()
    eval = EvaluationService()
    for location in locations:
        print("Current Location:" + location)
        dataset_pre = DatasetPrepareService(location=location)
        # map_client = dataset_pre.visualizing_images_per_day(satellites)
        # dataset_pre.batch_downloading_from_gclound_training(satellites)
        # dataset_pre.batch_downloading_from_gclound_referencing(satellites)
        # dataset_pre.generate_video_for_goes()
        # dataset_pre.generate_custom_gif(satellites)
        # dataset_pre.firms_generation_from_csv_to_tiff(True, mode)
        # dataset_pre.evaluate_tiff_to_png(location)
        # dataset_pre.label_tiff_to_png(location)
        # preprocessing.dataset_generator_firms_goes(location)
        # preprocessing.dataset_generator_trial5(location)
        eval.evaluate_mIoU(location, 'Sentinel2')
        # dataset_pre.download_dataset_to_gcloud(satellites, False)
        # eval.reconstruct_trial5(location)
        # dataset_pre.download_goes_dataset_to_gcloud_every_hour(False)
        # preprocessing.corp_tiff_to_same_size_training(location)
        # preprocessing.corp_tiff_to_same_size_referencing(location)
        # eval.reference_trial5(location)