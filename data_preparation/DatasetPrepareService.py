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
            satellite_client = Sentinel2(False)
        elif satellite == 'MODIS':
            satellite_client = MODIS()
        elif satellite == 'GOES':
            satellite_client = GOES()
        elif satellite == 'Sentinel1_asc':
            satellite_client = Sentinel1("asc", self.location)
        elif satellite == 'Sentinel1_dsc':
            satellite_client = Sentinel1("dsc", self.location)
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

    def prepare_image_patch(self, satellite, date_of_interest):
        satellite_client = self.get_satellite_client(satellite)
        img_collection = satellite_client.collection_of_interest(date_of_interest + 'T00:00',
                                                                 date_of_interest + 'T23:59',
                                                                 self.geometry)
        featureStack = img_collection.max().float()

        list = ee.List.repeat(1, 256)
        lists = ee.List.repeat(list, 256)
        kernel = ee.Kernel.fixed(256, 256, lists)

        arrays = featureStack.neighborhoodToArray(kernel)
        sample = arrays.sample(
            region=self.geometry,
            scale=30,
            numPixels=10,
            seed=1,
            tileScale=8
        )
        task = ee.batch.Export.table.toCloudStorage(
            collection=sample,
            description='batch_export_test',
            bucket=config.get('output_bucket'),
            fileNamePrefix= self.location + satellite + '/' + 'Cal_fire',
            fileFormat='GeoJSON'
        )
        task.start()
        print('Start with image task (id: {}).'.format(task.id))
        return sample

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
            fileNamePrefix=self.location + satellite + '/' + "Cal_fire_" + self.location + satellite + '-' + index,
            bucket=config.get('output_bucket'),
            scale=30,
            maxPixels=1e9,
            fileDimensions=256,
            fileFormat='TFRecord',
            region=self.geometry.toGeoJSON()['coordinates'],
        )

        image_task.start()

        print('Start with image task (id: {}).'.format(image_task.id))

    def download_collection_as_video(self, img_as_gif_collection, satellite, date):

        video_task = ee.batch.Export.video.toCloudStorage(
            collection=img_as_gif_collection,
            description='Image Export',
            fileNamePrefix=self.location + satellite + '/' + "Cal_fire_" + self.location + satellite + '-' + str(date),
            bucket=config.get('output_bucket'),
            maxPixels=1e9,
            dimensions=768,
            region=self.geometry.toGeoJSON()['coordinates'],
        )

        video_task.start()

        print('Start with video task (id: {}).'.format(video_task.id))

    # def parse_tfrecord(self, example_proto):
    #   """The parsing function.
    #   Read a serialized example into the structure defined by FEATURES_DICT.
    #   Args:
    #     example_proto: a serialized Example.
    #   Returns:
    #     A dictionary of tensors, keyed by feature name.
    #   """
    #   return tf.io.parse_single_example(example_proto, FEATURES_DICT)
    #
    #
    # def to_tuple(self, inputs):
    #   """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    #   Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    #   Args:
    #     inputs: A dictionary of tensors, keyed by feature name.
    #   Returns:
    #     A tuple of (inputs, outputs).
    #   """
    #   inputsList = [inputs.get(key) for key in FEATURES]
    #   stacked = tf.stack(inputsList, axis=0)
    #   # Convert from CHW to HWC
    #   stacked = tf.transpose(stacked, [1, 2, 0])
    #   return stacked[:,:,:len(BANDS)], stacked[:,:,len(BANDS):]
    #
    #
    # def get_dataset(self, pattern):
    #   """Function to read, parse and format to tuple a set of input tfrecord files.
    #   Get all the files matching the pattern, parse and convert to tuple.
    #   Args:
    #     pattern: A file pattern to match in a Cloud Storage bucket.
    #   Returns:
    #     A tf.data.Dataset
    #   """
    #   glob = tf.io.gfile.glob(pattern)
    #   dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
    #   dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=5)
    #   dataset = dataset.map(self.to_tuple, num_parallel_calls=5)
    #   return dataset
    # def get_training_dataset():
    #     """Get the preprocessed training dataset
    #   Returns:
    #     A tf.data.Dataset of training data.
    #   """
    #     glob = 'gs://' + BUCKET + '/' + FOLDER + '/' + TRAINING_BASE + '*'
    #     dataset = get_dataset(glob)
    #   shuffle will generate a buffer and everytime the records entered batch from buffer, shuffle function will fetch]
    #   a new records
    #     dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    #     return dataset
