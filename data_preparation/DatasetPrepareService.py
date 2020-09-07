import datetime
import time
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
from utils.EarthEngineMapClient import EarthEngineMapClient

ee.Initialize()
# Load configuration file
with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class DatasetPrepareService:
    def __init__(self, location, satellite):
        self.location = location
        self.satellite = satellite
        self.rectangular_size = config.get('rectangular_size')
        self.latitude = config.get(self.location).get('latitude')
        self.longitude = config.get(self.location).get('longitude')
        self.start_time = config.get(self.location).get('start')
        self.end_time = config.get(self.location).get('end')
        self.rectangular_size = config.get('rectangular_size')
        self.geometry = ee.Geometry.Rectangle(
            [self.longitude - self.rectangular_size, self.latitude - self.rectangular_size,
             self.longitude + self.rectangular_size, self.latitude + self.rectangular_size])

    def prepare_dataset(self, enable_downloading, enable_visualization):
        if self.satellite == 'Sentinel2':
            satellite_client = Sentinel2()
        elif self.satellite == 'MODIS':
            satellite_client = MODIS()
        elif self.satellite == 'GOES':
            satellite_client = GOES(self.geometry)
        elif self.satellite == 'Sentinel1':
            satellite_client = Sentinel1()
        elif self.satellite == 'VIIRS':
            satellite_client = VIIRS()
        else:
            satellite_client = Landsat8()
        time_dif = self.end_time-self.start_time
        for i in range(time_dif.days):
            date_of_interest = str(self.start_time + datetime.timedelta(days=i))
            img_collection, vis_params = satellite_client.collection_of_interest(date_of_interest + 'T00:00',
                                                                                 date_of_interest + 'T23:59',
                                                                                 self.geometry)
            # pprint(img_collection.getInfo())
            img = img_collection.median()
            system_id = str(img.get('system:index').getInfo())
            # Download tasks
            if enable_downloading:
                self.download_image_to_gcloud(img, system_id)
            # Visualization
            if enable_visualization:
                map_client = EarthEngineMapClient(self.latitude, self.longitude)
                pprint({'Image info:': img.getInfo()})
                map_client.add_ee_layer(img.clip(self.geometry), vis_params, self.satellite + system_id)
        if enable_visualization:
            map_client.initialize_map()

    def download_from_gcloud_and_parse(self):
        train_file_path = 'gs://' + config.get('output_bucket') + '/' + "Cal_fire_" + self.location + 's2-a' + '.tfrecord.gz'
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

    def download_image_to_gcloud(self, img, index):
        '''
        Export images to google cloud, the output image is a rectangular with the center at given latitude and longitude
        :param img: Image in GEE
        :return: None
        '''
        img = ee.Image(img).toFloat()

        pprint({'Image info:': img.getInfo()})
        print('Found Cloud Storage bucket.' if tf.io.gfile.exists('gs://' + config.get('output_bucket'))
              else 'Can not find output Cloud Storage bucket.')

        # Setup the task.
        image_task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description='Image Export',
            fileNamePrefix="Cal_fire_" + self.location + self.satellite + '-' + index,
            bucket=config.get('output_bucket'),
            scale=30,
            # fileFormat='TFRecord',
            region=self.geometry.toGeoJSON()['coordinates'],
        )

        image_task.start()

        while image_task.active():
            print('Polling for task (id: {}).'.format(image_task.id))
            time.sleep(30)
        print('Done with image export.')