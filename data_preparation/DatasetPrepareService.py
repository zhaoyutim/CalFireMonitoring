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
        # self.end_time = config.get(self.location).get('end')
        self.end_time = datetime.date.today()
        self.rectangular_size = config.get('rectangular_size')
        self.geometry = ee.Geometry.Rectangle(
            [self.longitude - self.rectangular_size, self.latitude - self.rectangular_size,
             self.longitude + self.rectangular_size, self.latitude + self.rectangular_size])

    def prepare_dataset(self, enable_downloading, enable_visualization, generate_gif):
        map_client = EarthEngineMapClient(self.latitude, self.longitude)
        if self.satellite == 'Sentinel2':
            satellite_client = Sentinel2()
        elif self.satellite == 'MODIS':
            satellite_client = MODIS()
        elif self.satellite == 'GOES':
            satellite_client = GOES(self.geometry)
        elif self.satellite == 'Sentinel1_asc':
            satellite_client = Sentinel1("asc")
        elif self.satellite == 'Sentinel1_dsc':
            satellite_client = Sentinel1("dsc")
        elif self.satellite == 'VIIRS':
            satellite_client = VIIRS()
        else:
            satellite_client = Landsat8()
        time_dif = self.end_time - self.start_time
        img_as_gif = []
        vis_params = satellite_client.get_visualization_parameter()
        for i in range(time_dif.days):
            date_of_interest = str(self.start_time + datetime.timedelta(days=i))
            img_collection = satellite_client.collection_of_interest(date_of_interest + 'T00:00',
                                                                    date_of_interest + 'T23:59',
                                                                    self.geometry)
            img = img_collection.max()
            # Download tasks
            if enable_downloading:
                self.download_image_to_gcloud(img, date_of_interest)
            # Visualization
            if enable_visualization:
                if len(img.getInfo().get('bands')) != 0:
                    map_client.add_ee_layer(img.clip(self.geometry), vis_params, self.satellite + date_of_interest)
                    img_as_gif.append(img)
            if generate_gif and self.satellite == 'GOES':
                self.download_collection_as_video(img_collection.select(vis_params.get('bands')), vis_params)

        if enable_visualization:
            map_client.initialize_map()

        if generate_gif and self.satellite != 'GOES':
            img_as_gif = ee.ImageCollection(img_as_gif).select(vis_params.get('bands'))
            self.download_collection_as_video(img_as_gif, vis_params)

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
            maxPixels=1e6,
            # fileFormat='TFRecord',
            region=self.geometry.toGeoJSON()['coordinates'],
        )

        image_task.start()

        print('Start with task (id: {}).'.format(image_task.id))

    def download_collection_as_video(self, img_as_gif_collection, vis_params):
        new_collection = []
        collection_size = img_as_gif_collection.size().getInfo()
        collection_list = img_as_gif_collection.toList(collection_size)
        videoArgs = {
            'dimensions': 256,
            'region': self.geometry,
            'framesPerSecond': 7,
            'min': vis_params.get('min'),
            'max': vis_params.get('max'),
        }
        increment = 50
        for offset in range (0, collection_size, increment):
            for index in range(offset, min(increment + offset, collection_size)):
                new_image = ee.Image(collection_list.get(index))
                new_collection.append(new_image)

            print(ee.ImageCollection(new_collection).getVideoThumbURL(videoArgs))