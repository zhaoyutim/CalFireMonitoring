import datetime
import urllib
from pprint import pprint

import ee
import tensorflow as tf
import yaml

from data_preparation.satellites.GOES import GOES
from data_preparation.satellites.Landsat8 import Landsat8
from data_preparation.satellites.MODIS import MODIS
from data_preparation.satellites.Sentinel1 import Sentinel1
from data_preparation.satellites.Sentinel2 import Sentinel2
from data_preparation.satellites.VIIRS import VIIRS
from data_preparation.utils.EarthEngineMapClient import EarthEngineMapClient

# Load configuration file
with open("data_preparation/config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class DatasetPrepareService:
    def __init__(self, location):
        self.location = location
        self.rectangular_size = config.get('rectangular_size')
        self.latitude = config.get(self.location).get('latitude')
        self.longitude = config.get(self.location).get('longitude')

        self.rectangular_size = config.get('rectangular_size')
        self.geometry = ee.Geometry.Rectangle(
            [self.longitude - self.rectangular_size, self.latitude - self.rectangular_size,
             self.longitude + self.rectangular_size, self.latitude + self.rectangular_size])

    def cast_to_uint8(self, image):
        return image.multiply(512).uint8()

    def get_satellite_client(self, satellite):
        if satellite == 'Sentinel2':
            satellite_client = Sentinel2()
        elif satellite == 'MODIS':
            satellite_client = MODIS()
        elif satellite == 'GOES':
            satellite_client = GOES()
        elif satellite == 'Sentinel1_asc':
            satellite_client = Sentinel1("asc")
        elif satellite == 'Sentinel1_dsc':
            satellite_client = Sentinel1("dsc")
        elif satellite == 'VIIRS':
            satellite_client = VIIRS()
        else:
            satellite_client = Landsat8()
        return satellite_client

    def prepare_daily_image(self, enable_image_downloading, satellite, date_of_interest):
        satellite_client = self.get_satellite_client(satellite)
        img_collection = satellite_client.collection_of_interest(date_of_interest + 'T00:00',
                                                                 date_of_interest + 'T23:59',
                                                                 self.geometry)
        vis_params = satellite_client.get_visualization_parameter()
        img_collection_as_gif = img_collection.select(vis_params.get('bands')).map(self.cast_to_uint8)
        if enable_image_downloading and len(img_collection.max().getInfo().get('bands')) != 0:
            vis_params['format'] = 'jpg'
            vis_params['dimensions'] = 768
            url = img_collection.max().clip(self.geometry).getThumbURL(vis_params)
            print(url)
            urllib.request.urlretrieve(url, 'images_for_gif/' + self.location + '/' + satellite + str(date_of_interest) + '.jpg')
        return img_collection, img_collection_as_gif

    def visualize_in_openstreetmap(self, img, map_client, satellite, date_of_interest):
        satellite_client = self.get_satellite_client(satellite)
        vis_params = satellite_client.get_visualization_parameter()
        if len(img.getInfo().get('bands')) != 0:
            map_client.add_ee_layer(img.clip(self.geometry), vis_params, satellite + date_of_interest)
        return map_client


    def download_from_gcloud_and_parse(self):
        train_file_path = 'gs://' + config.get(
            'output_bucket') + '/' + "Cal_fire_" + self.location + 's2-a' + '.tfrecord.gz'
        print('Found training file.' if tf.io.gfile.exists(train_file_path)
              else FileNotFoundError('No training file found.'))
        dataset = tf.data.TFRecordDataset(train_file_path, compression_type='GZIP')
        # List of fixed-length features, all of which are float32.
        columns = [
            tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in self.feature_names
        ]

        # Dictionary with names as keys, features as values.
        features_dict = dict(zip(self.feature_names, columns))
        pprint(features_dict)

        # Map the function over the dataset.
        parsed_dataset = tf.io.parse_single_example(dataset, features_dict)

        # Print the first parsed record to check.
        pprint(iter(parsed_dataset).next())

        return parsed_dataset

    def download_image_to_gcloud(self, img, satellite, index):
        '''
        Export images to google cloud, the output image is a rectangular with the center at given latitude and longitude
        :param img: Image in GEE
        :return: None
        '''

        # Setup the task.
        image_task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description='Image Export',
            fileNamePrefix="Cal_fire_" + self.location + satellite + '-' + index,
            bucket=config.get('output_bucket') + self.location + satellite + '/',
            scale=30,
            maxPixels=1e9,
            fileDimensions=256,
            # fileFormat='TFRecord',
            region=self.geometry.toGeoJSON()['coordinates'],
        )

        image_task.start()

        print('Start with image task (id: {}).'.format(image_task.id))

    def download_collection_as_video(self, img_as_gif_collection, satellite, date):

        video_task = ee.batch.Export.video.toCloudStorage(
            collection=img_as_gif_collection,
            description='Image Export',
            fileNamePrefix="Cal_fire_" + self.location + satellite + '-' + str(date),
            bucket=config.get('output_bucket'),
            maxPixels=1e9,
            dimensions=256,
            region=self.geometry.toGeoJSON()['coordinates'],
        )

        video_task.start()

        print('Start with video task (id: {}).'.format(video_task.id))

