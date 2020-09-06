import ee
from data_preparation.DatasetPrepareService import DatasetPrepareService

if __name__ == '__main__':
    ee.Initialize()
    dataset_pre = DatasetPrepareService('LNU_lighting_complex', 'GOES')
    dataset_pre.prepare_dataset(False, True)
    #dataset = dataset_sentinel2.download_from_gcloud_and_parse()
