import ee
from eeMapClient import eeMapClient
import yaml

ee.Initialize()

# Load configuration file
with open("configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

rectangular_size = config.get('rectangular_size')
max_cloud_probability = config.get('max_cloud_probability')
latitude = config.get('LNU_lighting_complex').get('latitude')
longitude = config.get('LNU_lighting_complex').get('longitude')
start_time = config.get('LNU_lighting_complex').get('start')
end_time = config.get('LNU_lighting_complex').get('end')
point = ee.Geometry.Point(latitude, longitude)

# Raw Dataset import
sentinel2c = ee.ImageCollection("COPERNICUS/S2")
sentinel2a = ee.ImageCollection("COPERNICUS/S2_SR")
s2_clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')


def maskClouds(img):
    clouds = ee.Image(img.get('cloud_mask')).select('probability');
    isNotCloud = clouds.lt(max_cloud_probability);
    return img.updateMask(isNotCloud)


def maskEdges(s2_img):
    return s2_img.updateMask(
        s2_img.select('B8A')
            .mask()
            .updateMask(s2_img.select('B9')
                        .mask())
    )


s2a_sr = sentinel2a.filterDate(start_time, end_time).filterBounds(point).map(maskEdges)
s2c_sr = sentinel2c.filterDate(start_time, end_time).filterBounds(point).map(maskEdges)
s2_clouds = s2_clouds.filterDate(start_time, end_time).filterBounds(point)

s2a_sr_with_cloud_mask = ee.Join.saveFirst('cloud_mask') \
    .apply(s2a_sr, s2_clouds, ee.Filter.equals(leftField='system:index', rightField='system:index'))

s2c_sr_with_cloud_mask = ee.Join.saveFirst('cloud_mask') \
    .apply(s2c_sr, s2_clouds, ee.Filter.equals(leftField='system:index', rightField='system:index'))

s2a_cloud_masked_collection = ee.ImageCollection(s2a_sr_with_cloud_mask).map(maskClouds)
s2c_cloud_masked_collection = ee.ImageCollection(s2c_sr_with_cloud_mask).map(maskClouds)

bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 'QA10', 'QA20', 'QA60']
label = 'landcover'
feature_names = list(bands)
feature_names.append(label)


s2a_cloud_masked = ee.ImageCollection(s2a_sr_with_cloud_mask).map(maskClouds).median()
s2c_cloud_masked = ee.ImageCollection(s2c_sr_with_cloud_mask).map(maskClouds).median()

rgbVis = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}

map_client = eeMapClient({'longitude': longitude, 'latitude': latitude})
# map_client.download_image_collection_to_gcloud(s2a_cloud_masked_collection, rectangular_size, "Cal_fire_", feature_names)
# map_client.download_image_to_gcloud(s2c_cloud_masked, rectangular_size, "Cal_fire_")

# # Visualization
map_client.add_to_map(s2a_cloud_masked, rgbVis, "s2CloudMasked")

