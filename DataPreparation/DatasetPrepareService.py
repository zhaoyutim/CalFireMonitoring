import datetime
import os
import urllib
from glob import glob
from pathlib import Path
from pprint import pprint

from array2gif import write_gif
from geetools import batch, tools, utils

import cv2
import ee
import imageio
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import yaml
from geetools.utils import makeName
from google.cloud import storage
from osgeo import gdal
from osgeo import osr
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from DataPreparation.satellites.FIRM import FIRMS
from DataPreparation.satellites.GOES import GOES
from DataPreparation.satellites.GOES_FIRE import GOES_FIRE
from DataPreparation.satellites.Landsat8 import Landsat8
from DataPreparation.satellites.MODIS import MODIS
from DataPreparation.satellites.Sentinel1 import Sentinel1
from DataPreparation.satellites.Sentinel2 import Sentinel2
from DataPreparation.satellites.VIIRS import VIIRS
from DataPreparation.utils.EarthEngineMapClient import EarthEngineMapClient
# Load configuration file
from Preprocessing.PreprocessingService import PreprocessingService

with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class DatasetPrepareService:
    def __init__(self, location):
        self.location = location
        self.rectangular_size = config.get('rectangular_size')
        self.latitude = config.get(self.location).get('latitude')
        self.longitude = config.get(self.location).get('longitude')
        self.start_time = config.get(location).get('start')
        self.end_time = config.get(location).get('end')
        # self.start_time = datetime.date(2020, 9, 10)
        # self.end_time = self.start_time + datetime.timedelta(days=2)
        # self.end_time = datetime.date.today()

        self.rectangular_size = config.get('rectangular_size')
        self.geometry = ee.Geometry.Rectangle(
            [self.longitude - self.rectangular_size, self.latitude - self.rectangular_size,
             self.longitude + self.rectangular_size, self.latitude + self.rectangular_size])
        self.scale_dict = {"GOES": 375, "GOES_FIRE": 375, "FIRMS": 1000, "Sentinel2": 375, "VIIRS": 375, "MODIS": 500, "Sentinel1_asc": 20, "Sentinel1_dsc":20}

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

    def visualizing_images_per_day(self, satellites):
        map_client = EarthEngineMapClient(self.location)

        dataset_pre = DatasetPrepareService(location=self.location)
        time_dif = self.end_time - self.start_time

        for i in range(time_dif.days):
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
        if satellite != 'GOES_every':
            img = img_coll.max().toFloat()
            image_task = ee.batch.Export.image.toCloudStorage(
                image=img,
                description='Image Export',
                fileNamePrefix=self.location + satellite + '/' + "Cal_fire_" + self.location + satellite + '-' + index,
                bucket=config.get('output_bucket'),
                scale=self.scale_dict.get(satellite),
                crs='EPSG:' + utm_zone,
                maxPixels=1e9,
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
                                                                  maxPixels=1e9,
                                                                  # fileDimensions=256,
                                                                  # fileFormat='TFRecord',
                                                                  region=self.geometry.toGeoJSON()['coordinates']
                                                                  )
                image_task.start()
                print('Start with image task (id: {}).'.format(image_task.id))
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
            for satellite in satellites:
                img_collection, img_collection_as_gif = self.prepare_daily_image(download_images_as_jpeg_locally,
                                                                                        satellite=satellite,
                                                                                        date_of_interest=date_of_interest)
                max_img = img_collection.max()
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

        for i in range(time_dif.days):
            date_of_interest = str(self.start_time + datetime.timedelta(days=i))
            for start_hour in range(0, 24):
                img_collection, img_collection_as_gif = self.prepare_daily_image(download_images_as_jpeg_locally,
                                                                                        satellite=satellite,
                                                                                        date_of_interest=date_of_interest,
                                                                                        time_stamp_start="{:02d}:00".format(start_hour),
                                                                                        time_stamp_end="{:02d}:59".format(start_hour)
                                                                                        )
                max_img = img_collection.max()
                if len(max_img.getInfo().get('bands')) != 0:
                    self.download_image_to_gcloud(img_collection, satellite, date_of_interest + "{:02d}:00".format(start_hour), utm_zone)
        if download_images_as_jpeg_locally:
            images = []
            for filename in filenames:
                images.append(imageio.imread('images_for_gif/' + self.location + '/' + filename + '.jpg'))
            imageio.mimsave('images_for_gif/' + self.location + '.gif', images, format='GIF', fps=1)

    def download_blob(self, bucket_name, prefix, destination_file_name):
        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.time_created.date() < datetime.date.today()-datetime.timedelta(days=1):
                continue
            filename = blob.name.replace('/', '_')
            blob.download_to_filename(destination_file_name + filename)
            print(
                "Blob {} downloaded to {}.".format(
                    filename, destination_file_name
                )
            )

    def batch_downloading_from_gclound_training(self, satellites):
        for satellite in satellites:
            blob_name = self.location + satellite + '/'
            destination_name = 'data/' + self.location + satellite + '/'
            dir_name = os.path.dirname(destination_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            self.download_blob(config.get('output_bucket'), blob_name, destination_name)

    def batch_downloading_from_gclound_referencing(self, satellites):
        for satellite in satellites:
            blob_name = self.location + satellite + '/'
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
        start_time = config.get(self.location).get('start')
        # end_time = config.get(location).get('end')
        end_time = datetime.date.today()
        filenames = []
        time_dif = end_time - start_time

        for i in range(time_dif.days):
            date_of_interest = str(start_time + datetime.timedelta(days=i))
            for satellite in satellites:
                if os.path.isfile('images_for_gif/' + self.location + '/' + satellite + str(date_of_interest) + '.jpg'):
                    filenames.append(satellite + str(date_of_interest))

                if os.path.isfile('images_for_gif/' + self.location + '/' + satellite + str(date_of_interest) + '.jpg'):
                    bk_img = cv2.imread(
                        'images_for_gif/' + self.location + '/' + satellite + str(date_of_interest) + '.jpg')
                    cv2.putText(bk_img, self.location + '-' + satellite + '-' + str(date_of_interest), (150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imwrite('images_after_processing/' + self.location + '/' + satellite + str(
                        date_of_interest) + '.jpg', bk_img)
                    filenames.append(satellite + str(date_of_interest))
        images = []
        for filename in filenames:
            images.append(imageio.imread('images_after_processing/' + self.location + '/' + filename + '.jpg'))
        imageio.mimsave('images_after_processing/gif/' + '/' + self.location + '.gif', images, format='GIF', fps=1)

    def firms_generation_from_csv_to_tiff(self, generate_goes, is_downsample, utm_zone, satellite='GOES'):
        preprocessing = PreprocessingService()
        time_dif = self.end_time - self.start_time
        dataset_pre = DatasetPrepareService(location=self.location)
        label_path = 'data/label/' + self.location + ' label' + '/'
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        for i in range(time_dif.days):
            date_of_interest = str(self.start_time + datetime.timedelta(days=i))
            path = Path('data/evaluate/'+self.location+'/reference')
            file_list = glob(str(path / "*.tif"))
            _, profile = preprocessing.read_tiff(file_list[0])
            bbox = [profile.data.get('transform').column_vectors[2][0],
                    profile.data.get('transform').column_vectors[2][0] +
                    profile.data.get('transform').column_vectors[0][0] * profile.data.get('width'),
                    profile.data.get('transform').column_vectors[2][1] +
                    profile.data.get('transform').column_vectors[1][1] * profile.data.get('height'),
                    profile.data.get('transform').column_vectors[2][1]]
            transformer = Transformer.from_crs(int(utm_zone), 4326)
            bot_left = transformer.transform(bbox[0], bbox[2])
            top_right = transformer.transform(bbox[1], bbox[3])
            lon = [bbox[0], bbox[1]]
            lat = [bbox[2], bbox[3]]
            if is_downsample == 'modis':
                res = 1000
                all_location = pd.read_csv('data/FIRMS/fire_nrt_M6_156697.csv')
            elif is_downsample == 'viirs':
                res = 375
                # all_location = pd.read_csv('data/FIRMS/fire_nrt_J1V-C2_156698.csv')
                # all_location = pd.read_csv('data/FIRMS/fire_archive_mid-2018-mid-2020.csv')

                all_location = pd.read_csv('data/FIRMS/fire_nrt_V1_mid2020-end-2020.csv')
            else:
                res = 2000
                all_location = pd.read_csv('data/FIRMS/fire_archive_V1_166189.csv')
            xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
            nx = int((xmax - xmin) // res)
            ny = int((ymax - ymin) // res)

            fire_data_filter_on_date_and_bbox = all_location[all_location.acq_date.eq(date_of_interest)
                                                             & all_location.latitude.gt(bot_left[0])
                                                             & all_location.latitude.lt(top_right[0])
                                                             & all_location.longitude.gt(bot_left[1])
                                                             & all_location.longitude.lt(top_right[1])]
            day_pixel = fire_data_filter_on_date_and_bbox[fire_data_filter_on_date_and_bbox.daynight.eq('D')]
            night_pixel = fire_data_filter_on_date_and_bbox[fire_data_filter_on_date_and_bbox.daynight.eq('N')]
            daynight = fire_data_filter_on_date_and_bbox.daynight.unique()
            transformer2 = Transformer.from_crs(4326, int(utm_zone))
            # for l in range(daynight.shape[0]):
            for time in range(len(fire_data_filter_on_date_and_bbox.acq_time.unique())):
                timestamp_per_day = fire_data_filter_on_date_and_bbox.acq_time.unique()[time]
                # if daynight[l] == 'D':
                #     timestamp_per_day = day_pixel.acq_time.unique()[time]
                # else:
                #     timestamp_per_day = night_pixel.acq_time.unique()[time]
                if generate_goes:
                    time_stamp_start, time_stamp_end = self.convert_int_to_timestamp(timestamp_per_day, 3)
                    img_collection, img_collection_as_gif = dataset_pre.prepare_daily_image(False, satellite=satellite,
                                                                                            date_of_interest=date_of_interest,
                                                                                            time_stamp_start=time_stamp_start,
                                                                                            time_stamp_end=time_stamp_end)
                    dataset_pre.download_image_to_gcloud(img_collection, satellite,
                                                         date_of_interest + '{:04d}'.format(timestamp_per_day), utm_zone)
                # fire_data_filter_on_timestamp = np.array(fire_data_filter_on_date_and_bbox[
                #                                              fire_data_filter_on_date_and_bbox.daynight.eq(
                #                                                  daynight[l])])
                fire_data_filter_on_timestamp = np.array(fire_data_filter_on_date_and_bbox[fire_data_filter_on_date_and_bbox.acq_time.eq(timestamp_per_day)])
                image_size = (ny, nx)
                #  Create Each Channel
                b1_pixels = np.zeros((image_size), dtype=np.float)
                b2_pixels = np.zeros((image_size), dtype=np.float)
                b3_pixels = np.zeros((image_size), dtype=np.float)
                b4_pixels = np.zeros((image_size), dtype=np.float)

                # coordination in images = (coord_in_crs - min_bbox) / resolution

                for k in range(1, fire_data_filter_on_timestamp.shape[0]):
                    record = fire_data_filter_on_timestamp[k]
                    lon_point = transformer2.transform(record[0], record[1])[0]
                    lat_point = transformer2.transform(record[0], record[1])[1]
                    cord_x = int((lon_point - xmin) // res)
                    cord_y = int((lat_point - ymin) // res)
                    if cord_x >= nx or cord_y >= ny:
                        continue
                    b1_pixels[cord_y, cord_x] = max(b1_pixels[cord_y, cord_x], record[2])
                    if record[8] == 'n':
                        b2_pixels[cord_y, cord_x] = max(b2_pixels[cord_y, cord_x], 1)
                    elif record[8] == 'h':
                        b2_pixels[cord_y, cord_x] = max(b2_pixels[cord_y, cord_x], 1)
                    else:
                        b2_pixels[cord_y, cord_x] = max(b2_pixels[cord_y, cord_x], 1)
                    b3_pixels[cord_y, cord_x] = max(b3_pixels[cord_y, cord_x], record[11])
                    b4_pixels[cord_y, cord_x] = max(b4_pixels[cord_y, cord_x], record[12])

                # Geotransform matrix: (top_left_lon, resolution_x, spin, top_left_lat, resolution_y, spin)
                geotransform = (xmin, res, 0, ymin, 0, res)

                # create the n-band raster file
                if is_downsample:
                    dst_ds = gdal.GetDriverByName('GTiff').Create(
                        'data/label/' + self.location + ' label' + '/' + "Cal_fire_" + self.location + 'FIRMS' + '-' + str(
                            date_of_interest) + '{:04d}'.format(timestamp_per_day) + '.tif', image_size[1],
                        image_size[0], 4, gdal.GDT_Float64)
                else:
                    dst_ds = gdal.GetDriverByName('GTiff').Create(
                        'data/label/' + self.location + ' label' + '/' + "Cal_fire_" + self.location + 'FIRMS' + '-' + str(
                            date_of_interest) + '{:04d}'.format(timestamp_per_day) + '.tif', image_size[1], image_size[0], 4,
                        gdal.GDT_Float64)

                # b1_pixels = gaussian_filter(b1_pixels, sigma=1, order=0)
                # b2_pixels = gaussian_filter(b2_pixels, sigma=1, order=0)
                # b3_pixels = gaussian_filter(b3_pixels, sigma=1, order=0)
                # b4_pixels = gaussian_filter(b4_pixels, sigma=1, order=0)
                dst_ds.SetGeoTransform(geotransform)  # specify coords
                srs = osr.SpatialReference()  # establish encoding
                srs.ImportFromEPSG(int(utm_zone))  # WGS84 lat/long
                dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
                dst_ds.GetRasterBand(1).WriteArray(b1_pixels)  # write r-band to the raster
                dst_ds.GetRasterBand(2).WriteArray(b2_pixels)  # write g-band to the raster
                dst_ds.GetRasterBand(3).WriteArray(b3_pixels)  # write b-band to the raster
                dst_ds.GetRasterBand(4).WriteArray(b4_pixels)  # write b-band to the raster
                dst_ds.FlushCache()  # write to disk
                dst_ds = None

    def label_tiff_to_png(self, location):
        data_path = 'data/progression/'+location
        data_path = Path(data_path)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        data_file_list = glob(str(data_path / "*.tif"))
        data_file_list.sort()
        count=0
        # background_path = 'data/png/'+location+'_sentinel2.png'
        # with rasterio.open(background_path, 'r') as reader:
        #     background_as_array = reader.read()
        for file in data_file_list:
            time = file[-14:-4]
            with rasterio.open(file, 'r') as reader:
                tif_as_array = reader.read()
            output_image = np.rot90(tif_as_array[0, :, :], 2)
            output_image = np.flip(output_image, axis=1)
            # output_image = cv2.resize(output_image, (background_as_array.shape[2], background_as_array.shape[1]), interpolation=cv2.INTER_CUBIC)
            output_image = ((output_image - output_image.min()) * (1 / (output_image.max() - output_image.min()) * 255)).astype(
                'uint8')
            # output_three_channel = np.zeros((output_image.shape[0], output_image.shape[1], 4))
            # output_three_channel[:, :, 0] = output_image
            # output_three_channel[:, :, 1] = np.zeros((output_image.shape[0], output_image.shape[1]))
            # output_three_channel[:, :, 2] = np.zeros((output_image.shape[0], output_image.shape[1]))
            #
            # # ret,alpha = cv2.threshold(output_image,0,255,cv2.THRESH_BINARY)
            # output_three_channel[:, :, 3] = output_image

            output_three_channel = np.zeros((output_image.shape[0], output_image.shape[1]))
            output_three_channel[:, :] = output_image

            # cv2.putText(output_three_channel, location+' ' + file.replace('data/label/blue_ridge_fire label/Cal_fire_blue_ridge_fireGOES-','').replace('3_downsampled.tif','')[:-2]+':'+file.replace('data/label/blue_ridge_fire label/Cal_fire_blue_ridge_fireGOES-','').replace('3_downsampled.tif','')[-2:] , (400, 70),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (255, 255, 255, 255), 2, cv2.LINE_AA)
            imageio.imsave('data/png/'+location+time+'.png', output_three_channel)
            count += 1


    def evaluate_tiff_to_png(self, location):
        data_path = 'data/evaluate/'+location+'/recon'
        data_path = Path(data_path)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        data_file_list = glob(str(data_path / "*.tif"))
        data_file_list.sort()

        background_path = 'data/png/'+location+'_sentinel2.png'
        with rasterio.open(background_path, 'r') as reader:
            background_as_array = reader.read()
        count=0
        output_gif = []
        for file in data_file_list:
            with rasterio.open(file, 'r') as reader:
                tif_as_array = reader.read()
            output_image = np.rot90(tif_as_array[0, :, :], 2)
            output_image = np.flip(output_image, axis=1)
            output_image = cv2.resize(output_image, (background_as_array.shape[2], background_as_array.shape[1]), interpolation=cv2.INTER_CUBIC)
            output_image = ((output_image - output_image.min()) * (1 / (output_image.max() - output_image.min()) * 255)).astype(
                'uint8')
            output_three_channel = np.zeros((output_image.shape[0], output_image.shape[1], 4))
            output_three_channel[:, :, 0] = output_image
            output_three_channel[:, :, 1] = np.zeros((output_image.shape[0], output_image.shape[1]))
            output_three_channel[:, :, 2] = np.zeros((output_image.shape[0], output_image.shape[1]))

            # ret,alpha = cv2.threshold(output_image,0,255,cv2.THRESH_BINARY)
            output_three_channel[:, :, 3] = output_image

            cv2.putText(output_three_channel, location+' ' + str(self.start_time + datetime.timedelta(days=count / 24)) + " {:02d}:00".format(count % 24), (400, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255, 255), 2, cv2.LINE_AA)
            imageio.imsave('data/png/'+location+str(count)+'.png', output_three_channel)
            count += 1
            output_gif.append(output_three_channel)
        # write_gif(output_gif, 'data/png/'+location+'.gif', fps=1)
        # imageio.mimsave('data/png/'+location+'.gif', output_gif, format='GIF', fps=1)