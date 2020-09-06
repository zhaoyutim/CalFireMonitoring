import ee
from data_preparation.DatasetSentinel2 import DatasetSentinel2
from data_preparation.DatasetLandsat8 import DatasetLandsat8

ee.Initialize()
dataset_sentinel2 = DatasetLandsat8('LNU_lighting_complex')
dataset_sentinel2.prepare_dataset(False, True)
