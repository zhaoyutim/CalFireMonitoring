import datetime
import os
import urllib
from glob import glob
from pprint import pprint

import cv2
import ee
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from google.cloud import storage
from matplotlib import pyplot as plt

from DataPreparation.satellites.FIRM import FIRMS
from DataPreparation.satellites.FirePred import FirePred
from DataPreparation.satellites.GOES import GOES
from DataPreparation.satellites.GOES_FIRE import GOES_FIRE
from DataPreparation.satellites.Landsat8 import Landsat8
from DataPreparation.satellites.MODIS import MODIS
from DataPreparation.satellites.Sentinel1 import Sentinel1
from DataPreparation.satellites.Sentinel2 import Sentinel2
from DataPreparation.satellites.VIIRS import VIIRS
from DataPreparation.satellites.VIIRS_Day import VIIRS_Day
from DataPreparation.utils.EarthEngineMapClient import EarthEngineMapClient
from Preprocessing.PreprocessingService import PreprocessingService

# Load configuration file

with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

year = '2020'
filename = 'roi/us_fire_' + year + '_out.csv'
df = pd.read_csv(filename)
df = df.sort_values(by=['Id'])
ids, start_dates, end_dates, lons, lats = df['Id'].values.astype(str), df['start_date'].values.astype(str), df[
    'end_date'].values.astype(str), df['lon'].values.astype(float), df['lat'].values.astype(float)
class DatasetPrepareService:
    def __init__(self, location, start_time, end_time, roi=None, rectangular_size=None, latitude=None, longitude=None):
        self.location = location
        # self.rectangular_size = config.get('rectangular_size')
        # self.latitude = config.get(self.location).get('latitude')
        # self.longitude = config.get(self.location).get('longitude')
        # self.start_time = config.get(location).get('start')
        # self.end_time = config.get(location).get('end')

        self.start_time = start_time
        self.end_time = end_time
        # self.start_time = datetime.date(2018, 7, 14)
        # self.end_time = datetime.date(2018, 7, 18)
        # self.end_time = self.start_time + datetime.timedelta(days=10)
        # self.end_time = datetime.date.today()
        # self.rectangular_size = config.get('rectangular_size')
        if roi == None:
            self.rectangular_size = rectangular_size
            self.latitude = latitude
            self.longitude = longitude
            self.geometry = ee.Geometry.Rectangle(
                [self.longitude - self.rectangular_size, self.latitude - self.rectangular_size,
                 self.longitude + self.rectangular_size, self.latitude + self.rectangular_size])
        else:
            self.geometry = ee.Geometry.Rectangle(roi)

        self.scale_dict = {"VIIRS_Day":375, "GOES": 375, "GOES_FIRE": 2000, "FIRMS": 500, "Sentinel2": 20, "VIIRS": 375, "MODIS": 500, "Sentinel1_asc": 20, "Sentinel1_dsc":20, "FirePred":375}

    def cast_to_uint8(self, image):
        return image.multiply(512).uint8()

    def convert_int_to_timestamp(self, number, period):
        mins = int(number % 100)
        hours = int(number / 100 % 100)
        start = "{:02d}:{:02d}".format(max(hours - period, 0), mins)
        end = "{:02d}:{:02d}".format(min(hours, 23), mins)
        return start, end

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
        elif satellite == 'FIRMS':
            satellite_client = FIRMS()
        elif satellite == 'GOES_FIRE':
            satellite_client = GOES_FIRE()
        elif satellite == 'VIIRS_Day':
            satellite_client = VIIRS_Day()
        elif satellite == 'FirePred':
            satellite_client = FirePred()
        else:
            satellite_client = Landsat8()
        return satellite_client

    def prepare_daily_image(self, enable_image_downloading, satellite, date_of_interest, time_stamp_start="00:00", time_stamp_end="23:59"):
        satellite_client = self.get_satellite_client(satellite)
        img_collection = satellite_client.collection_of_interest(date_of_interest+'T'+time_stamp_start,
                                                                 date_of_interest+'T'+time_stamp_end,
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

    def visualizing_images_per_day(self, satellites, time_dif):
        map_client = EarthEngineMapClient(self.location)

        dataset_pre = DatasetPrepareService(location=self.location)
        # time_dif = self.end_time - self.start_time

        for i in range(time_dif):
            date_of_interest = str(self.start_time + datetime.timedelta(days=i))
            for satellite in satellites:
                img_collection, img_collection_as_gif = dataset_pre.prepare_daily_image(False, satellite=satellite,
                                                                                        date_of_interest=date_of_interest)
                map_client = dataset_pre.visualize_in_openstreetmap(img_collection.max(), map_client, satellite,
                                                                    date_of_interest)
        map_client.initialize_map()
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

    def download_image_to_gcloud(self, img_coll, satellite, index, utm_zone):
        '''
        Export images to google cloud, the output image is a rectangular with the center at given latitude and longitude
        :param img: Image in GEE
        :return: None
        '''

        # Setup the task.
        # size = img_coll.size().getInfo()
        # img_coll = img_coll.toList(size)
        # for i in range(size):
            # img = ee.Image(img_coll.get(i))
        if satellite != 'GOES_every':
            img = img_coll.max().toFloat()
            image_task = ee.batch.Export.image.toCloudStorage(
                image=img,
                description='Image Export',
                fileNamePrefix=self.location +'/' + satellite + '/' + index,
                bucket=config.get('output_bucket'),
                scale=self.scale_dict.get(satellite),
                crs='EPSG:' + utm_zone,
                maxPixels=1e13,
                # fileDimensions=256,
                # fileFormat='TFRecord',
                region=self.geometry.toGeoJSON()['coordinates'],
            )
            image_task.start()
            print('Start with image task (id: {}).'.format(image_task.id))
        else:
            size = img_coll.size().getInfo()
            img_list = img_coll.toList(size)

            n = 0
            while n < size:
                img = ee.Image(img_list.get(n))

                image_task = ee.batch.Export.image.toCloudStorage(image=img.toFloat(),
                                                                  description='Image Export',
                                                                  fileNamePrefix=self.location + satellite + '/' + "Cal_fire_" + self.location + satellite + '-' + index + '-' + str(n),
                                                                  bucket=config.get('output_bucket'),
                                                                  scale=self.scale_dict.get(satellite),
                                                                  crs='EPSG:32610',
                                                                  maxPixels=1e13,
                                                                  # fileDimensions=256,
                                                                  # fileFormat='TFRecord',
                                                                  region=self.geometry.toGeoJSON()['coordinates']
                                                                  )
                image_task.start()
                print('Start with image task (id: {}).'.format(image_task.id)+index)
                n += 1

    def download_collection_as_video(self, img_as_gif_collection, satellite, date):

        video_task = ee.batch.Export.video.toCloudStorage(
            collection=img_as_gif_collection,
            description='Image Export',
            fileNamePrefix=self.location + satellite + '/' + "Cal_fire_" + self.location + satellite + '-' + str(date),
            bucket=config.get('output_bucket'),
            maxPixels=1e9,
            crs='EPSG:32610',
            scale=self.scale_dict.get(satellite),
            region=self.geometry.toGeoJSON()['coordinates'],
        )

        video_task.start()

        print('Start with video task (id: {}).'.format(video_task.id))

    def download_dataset_to_gcloud(self, satellites, utm_zone, download_images_as_jpeg_locally):
        filenames = []
        time_dif = self.end_time - self.start_time
        for i in range(time_dif.days):
            date_of_interest = str(self.start_time + datetime.timedelta(days=i))
            print(date_of_interest)
            for satellite in satellites:
                img_collection, img_collection_as_gif = self.prepare_daily_image(download_images_as_jpeg_locally,
                                                                                        satellite=satellite,
                                                                                        date_of_interest=date_of_interest)
                max_img = img_collection.median()
                if len(max_img.getInfo().get('bands')) != 0:
                    self.download_image_to_gcloud(img_collection, satellite, date_of_interest, utm_zone)
        if download_images_as_jpeg_locally:
            images = []
            for filename in filenames:
                images.append(imageio.imread('images_for_gif/' + self.location + '/' + filename + '.jpg'))
            imageio.mimsave('images_for_gif/' + self.location + '.gif', images, format='GIF', fps=1)

    def download_goes_dataset_to_gcloud_every_hour(self, download_images_as_jpeg_locally, utm_zone, satellite='GOES'):
        filenames = []
        time_dif = self.end_time - self.start_time

        for i in range(3):
            date_of_interest = str(self.start_time + datetime.timedelta(days=i))
            for start_hour in range(0, 24):
                for minute in range(0, 60, 15):
                    end_minute = minute + 14
                    img_collection, img_collection_as_gif = self.prepare_daily_image(download_images_as_jpeg_locally,
                                                                                            satellite=satellite,
                                                                                            date_of_interest=date_of_interest,
                                                                                            time_stamp_start="{:02d}:{:02d}".format(start_hour, minute),
                                                                                            time_stamp_end="{:02d}:{:02d}".format(start_hour, end_minute)
                                                                                            )
                    max_img = img_collection.max()
                    if len(max_img.getInfo().get('bands')) != 0:
                        self.download_image_to_gcloud(img_collection, satellite, date_of_interest + "{:02d}:{:02d}".format(start_hour, minute), utm_zone)
        if download_images_as_jpeg_locally:
            images = []
            for filename in filenames:
                images.append(imageio.imread('images_for_gif/' + self.location + '/' + filename + '.jpg'))
            imageio.mimsave('images_for_gif/' + self.location + '.gif', images, format='GIF', fps=1)

    def download_blob(self, bucket_name, prefix, destination_file_name, create_time):
        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.time_created.date() < datetime.datetime.strptime(create_time, '%Y-%m-%d').date():
                continue
            filename = blob.name.split('/')[2].replace('.tif', '')+'_'+blob.name.split('/')[1]+'.tif'
            blob.download_to_filename(destination_file_name + filename)
            print(
                "Blob {} downloaded to {}.".format(
                    filename, destination_file_name
                )
            )

    def batch_downloading_from_gclound_training(self, satellites, create_time):
        for satellite in satellites:
            blob_name = self.location + '/'+ satellite + '/'
            destination_name = 'data/' + self.location + '/' + satellite + '/'
            dir_name = os.path.dirname(destination_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            self.download_blob(config.get('output_bucket'), blob_name, destination_name, create_time)

    def batch_downloading_from_gclound_referencing(self, satellites):
        for satellite in satellites:
            blob_name = self.location + '/' + satellite + '/'
            destination_name = 'data/evaluate/' + self.location + '/reference/'
            dir_name = os.path.dirname(destination_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            self.download_blob(config.get('output_bucket'), blob_name, destination_name)

    def generate_video_for_goes(self):

        time_dif = self.end_time - self.start_time

        for i in range(time_dif.days):
            date_of_interest = str(self.start_time + datetime.timedelta(days=i))
            satellite = 'GOES'
            _, img_collection_as_gif = self.prepare_daily_image(False,
                                                                       satellite=satellite,
                                                                       date_of_interest=date_of_interest)
            self.download_collection_as_video(img_collection_as_gif, satellite, date_of_interest)

    def generate_custom_gif(self, satellites):
        path = os.path.join('images_for_gif', self.location)
        file_list = glob(path+'/*.png')
        file_list.sort()
        images=[]
        for filename in file_list:
            img = imageio.imread(filename)
            goes_resized = cv2.resize(img, (450, 500), interpolation=cv2.INTER_LINEAR)
            images.append(goes_resized)
        for fps in [5, 10]:
            imageio.mimsave('images_after_processing/' + self.location + 'fps'+str(fps)+'.gif', images, format='GIF', fps=fps)

    def tif_to_png(self, satellites):
        start_time = config.get(self.location).get('start')
        end_time = config.get(self.location).get('end')
        for satellite in satellites:
            # path = os.path.join('data/', self.location, satellite)
            path = os.path.join('data/evaluate/',self.location, 'reference')
            file_list = glob(os.path.join(path, '*.tif'))
            file_list.sort()
            preprocessing = PreprocessingService()
            for i, file in enumerate(file_list):
                date_of_interest = file.split('/')[-1].split('_')[0]
                array, _ = preprocessing.read_tiff(file)
                img = np.zeros((array.shape[1], array.shape[2], 3))
                for j in range(3):
                    img[:,:,2-j] = (array[j,:,:]-array[j,:,:].min())/(array[j,:,:].max()-array[j,:,:].min())*255.0
                cv2.putText(img, self.location + '-' + satellite + '-' + str(date_of_interest), (array.shape[2]//2-220, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite('images_for_gif/' + self.location +'/'+date_of_interest+ '.png', img)

    def tif_to_png_agg(self, satellites):
        start_time = config.get(self.location).get('start')
        end_time = config.get(self.location).get('end')
        for satellite in satellites:
            for time in range(3):
                date_of_interest = str(self.start_time + datetime.timedelta(days=time))
                # path = os.path.join('data/', self.location, satellite)
                # path = os.path.join('data/evaluate/',self.location, 'reference')
                path = os.path.join('data/',self.location, satellite, str(date_of_interest))
                file_list = glob(os.path.join(path, '*.tif'))
                file_list.sort()
                preprocessing = PreprocessingService()
                array = []
                for i, file in enumerate(file_list):
                    date_of_interest = file.split('/')[-1].split('_')[0]
                    array_i, _ = preprocessing.read_tiff(file)
                    array.append(array_i)
                array=np.stack(array, axis=0)
                array = np.mean(array, axis=0)
                img = np.zeros((array.shape[1], array.shape[2], 3))
                for j in range(3):
                    img[:,:,2-j] = (array[j,:,:]-array[j,:,:].min())/(array[j,:,:].max()-array[j,:,:].min())*255.0
                cv2.putText(img, self.location + '-' + satellite + '-' + str(date_of_interest), (array.shape[2]//2-220, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite('images_for_gif/' + self.location +'/'+date_of_interest+ '.png', img)