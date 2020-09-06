import ee
from utils.EarthEngineMapClient import EarthEngineMapClient
import yaml

ee.Initialize()
# Load configuration file
with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class DatasetSentinel2:
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
        sentinel2c = ee.ImageCollection('COPERNICUS/S2')
        sentinel2a = ee.ImageCollection('COPERNICUS/S2_SR')
        s2_clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        geometry = ee.Geometry.Rectangle([longitude - size, latitude - size, longitude + size, latitude + size])
        # Here needs to filter boundary on the rectangular or polygons, otherwise we would easily reach the limit(5000) of
        # elements.
        s2a_sr = sentinel2a.filterDate(start_time, end_time).filterBounds(geometry).map(self.maskEdges)
        s2c_sr = sentinel2c.filterDate(start_time, end_time).filterBounds(geometry).map(self.maskEdges)
        s2_clouds = s2_clouds.filterDate(start_time, end_time).filterBounds(geometry)

        s2a_sr_with_cloud_mask = ee.Join.saveFirst('cloud_mask') \
            .apply(s2a_sr, s2_clouds, ee.Filter.equals(leftField='system:index', rightField='system:index'))

        s2c_sr_with_cloud_mask = ee.Join.saveFirst('cloud_mask') \
            .apply(s2c_sr, s2_clouds, ee.Filter.equals(leftField='system:index', rightField='system:index'))

        s2a_cloud_masked_collection = ee.ImageCollection(s2a_sr_with_cloud_mask).map(self.maskClouds)
        s2c_cloud_masked_collection = ee.ImageCollection(s2c_sr_with_cloud_mask).map(self.maskClouds)

        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 'QA10', 'QA20',
                 'QA60']
        feature_names = list(bands)

        s2a_cloud_masked = ee.ImageCollection(s2a_sr_with_cloud_mask).map(self.maskClouds).median()
        s2c_cloud_masked = ee.ImageCollection(s2c_sr_with_cloud_mask).map(self.maskClouds).median()

        rgbVis = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}

        map_client = EarthEngineMapClient(latitude, longitude)

        # Download tasks
        if enable_downloading:
            map_client.download_image_collection_to_gcloud(s2a_cloud_masked_collection, rectangular_size, "Cal_fire_",
                                                           feature_names)
            map_client.download_image_collection_to_gcloud(s2c_cloud_masked_collection, rectangular_size, "Cal_fire_",
                                                           feature_names)
            # map_client.download_image_to_gcloud(s2c_cloud_masked, rectangular_size, "Cal_fire_", geometry)

        # Visualization
        if enable_visualization:
            map_client.add_ee_layer(s2a_cloud_masked, rgbVis, "s2-a CloudMasked")
            map_client.add_ee_layer(s2c_cloud_masked, rgbVis, "s2-c CloudMasked")
            map_client.initialize_map()

    def maskClouds(self, img):
        max_cloud_probability = config.get('max_cloud_probability')
        clouds = ee.Image(img.get('cloud_mask')).select('probability');
        isNotCloud = clouds.lt(max_cloud_probability);
        return img.updateMask(isNotCloud)

    def maskEdges(self, s2_img):
        return s2_img.updateMask(
            s2_img.select('B8A')
                .mask()
                .updateMask(s2_img.select('B9')
                            .mask())
        )
