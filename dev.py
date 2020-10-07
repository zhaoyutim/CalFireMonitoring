import ee

from DataPreparation.DatasetPrepareService import DatasetPrepareService
from Preprocessing.PreprocessingService import PreprocessingService
ee.Initialize()
if __name__ == '__main__':
    satellites = ['GOES']
    locations = ['LNU_lighting_complex', 'SCU_lighting_complex', 'CZU_lighting_complex', 'North_complex_fire', 'creek_fire', 'North_complex_fire']
    generate_goes = True
    is_downsample = True
    preprocessing = PreprocessingService()
    for location in locations:
        dataset_pre = DatasetPrepareService(location=location)
        # map_client = dataset_pre.visualizing_images_per_day(satellites)
        # dataset_pre.batch_downloading_from_gclound(satellites)
        # dataset_pre.generate_video_for_goes()
        # dataset_pre.generate_custom_gif(satellites)
        dataset_pre.firms_generation_from_csv_to_tiff(generate_goes, is_downsample)
        # dataset_pre.download_dataset_to_gcloud(satellites, False)