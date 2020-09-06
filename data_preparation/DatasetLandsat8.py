from pprint import pprint

import ee
from utils.EarthEngineMapClient import EarthEngineMapClient
import yaml

ee.Initialize()
# Load configuration file
with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class DatasetLandsat8:
    def __init__(self, location):
        self.location = location

    def prepare_dataset(self, enable_downloading, enable_visualization):
        rectangular_size = config.get('rectangular_size')

        latitude = config.get(self.location).get('latitude')
        longitude = config.get(self.location).get('longitude')
        start_time = config.get(self.location).get('start')
        end_time = config.get(self.location).get('end')
        size = config.get('rectangular_size')

        # Raw Dataset import
        sentinel2c = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
        geometry = ee.Geometry.Rectangle([longitude - size, latitude - size, longitude + size, latitude + size])
        # Here needs to filter boundary on the rectangular or polygons, otherwise we would easily reach the limit(5000) of
        # elements.
        landsat_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate('2016-01-01', '2016-12-31').map(self.mask_L8_sr).filterBounds(geometry)

        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
        feature_names = list(bands)

        vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.4}
        map_client = EarthEngineMapClient(latitude, longitude)
        # Download tasks
        if enable_downloading:
            map_client.download_image_collection_to_gcloud(landsat_collection, rectangular_size, "Cal_fire_",
                                                           feature_names)
            # map_client.download_image_to_gcloud(s2c_cloud_masked, rectangular_size, "Cal_fire_", geometry)

        # Visualization
        if enable_visualization:
            pprint({'Image collection info:': landsat_collection.getInfo()})
            img = landsat_collection.mean()
            pprint({'Image info:': img.getInfo()})
            map_client.add_ee_layer(img, vis_params, "landsat8 CloudMasked")
            map_client.initialize_map()

    def mask_L8_sr(self, img):
        cloud_shadow_bit_mask = (1 << 3)
        clouds_bit_mask = (1 << 5)
        qa = img.select('pixel_qa')
        mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
        return img.updateMask(mask)
