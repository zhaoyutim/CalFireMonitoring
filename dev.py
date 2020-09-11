import ee
from data_preparation.DatasetPrepareService import DatasetPrepareService
from data_preparation.satellites.Sentinel1 import Sentinel1

if __name__ == '__main__':
    ee.Initialize()
    dataset_pre = DatasetPrepareService('creek_fire', 'GOES')
    dataset_pre.prepare_dataset(False, True)
    #dataset = dataset_sentinel2.download_from_gcloud_and_parse()