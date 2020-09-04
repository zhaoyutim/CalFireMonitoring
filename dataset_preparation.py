import ee
from pprint import pprint
import tensorflow as tf
from eeMapClient import eeMapClient

ee.Initialize()

sentinel2c = ee.ImageCollection("COPERNICUS/S2")
sentinel2a = ee.ImageCollection("COPERNICUS/S2_SR")
s2_clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

LNU_lighting_complex = {
    'longitude': -122.237303,
    'latitude': 38.593546,
    'start': '2020-08-17T06:40',
    'end': '2020-09-02T19:54',
}
RECTANGULAR_SIZE = 0.3

MAX_CLOUD_PROBABILITY = 65
region = ee.Geometry.Rectangle(coords=[LNU_lighting_complex.get('longitude') - RECTANGULAR_SIZE,
                                       LNU_lighting_complex.get('latitude') - RECTANGULAR_SIZE,
                                       LNU_lighting_complex.get('longitude') + RECTANGULAR_SIZE,
                                       LNU_lighting_complex.get('latitude') + RECTANGULAR_SIZE],
                               geodesic=False,
                               proj=None)


def maskClouds(img):
    clouds = ee.Image(img.get('cloud_mask')).select('probability');
    isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY);
    return img.updateMask(isNotCloud)


def maskEdges(s2_img):
    return s2_img.updateMask(
        s2_img.select('B8A')
            .mask()
            .updateMask(s2_img.select('B9')
                        .mask()
                        )
    )


s2Sr = sentinel2a.filterDate(LNU_lighting_complex.get('start'), LNU_lighting_complex.get('end')) \
    .filterBounds(region) \
    .map(maskEdges)
s2_clouds = s2_clouds.filterDate(LNU_lighting_complex.get('start'), LNU_lighting_complex.get('end')) \
    .filterBounds(region)

s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask')\
    .apply(s2Sr, s2_clouds, ee.Filter.equals(leftField='system:index', rightField='system:index'))

s2CloudMasked = ee.ImageCollection(
    s2SrWithCloudMask).map(maskClouds).median()
rgbVis = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}

#Visulization
map_client = eeMapClient({'longitude': -122.237303,
                          'latitude': 38.593546})
map_client.add_to_map(s2CloudMasked, rgbVis, "s2CloudMasked")

# USER_NAME = 'test'
# OUTPUT_BUCKET = 'zhaoyutimtest'
# BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
# LABEL = 'landcover'
# N_CLASSES = 3
# FEATURE_NAMES = list(BANDS)
# FEATURE_NAMES.append(LABEL)
# file_extension = '.tfrecord.gz'
# IMAGE_FILE_PREFIX = 'Cal_fire_'
# EXPORT_REGION = ee.Geometry.Rectangle([LNU_lighting_complex.get('longitude') - RECTANGULAR_SIZE,
#                                        LNU_lighting_complex.get('latitude') - RECTANGULAR_SIZE,
#                                        LNU_lighting_complex.get('longitude') + RECTANGULAR_SIZE,
#                                        LNU_lighting_complex.get('latitude') + RECTANGULAR_SIZE])
#
# pprint({'training': s2CloudMasked.getInfo()})
# print('Found Cloud Storage bucket.' if tf.io.gfile.exists('gs://' + OUTPUT_BUCKET)
#     else 'Can not find output Cloud Storage bucket.')
#
# # Setup the task.
# image_task = ee.batch.Export.image.toCloudStorage(
#   image=s2CloudMasked,
#   description='Image Export',
#   fileNamePrefix=IMAGE_FILE_PREFIX,
#   bucket=OUTPUT_BUCKET,
#   scale=30,
#   fileFormat='GeoTIFF',
#   region=EXPORT_REGION.toGeoJSON()['coordinates'],
# )
#
# image_task.start()
#
# import time
# while image_task.active():
#   print('Polling for task (id: {}).'.format(image_task.id))
#   time.sleep(30)
# print('Done with image export.')