import copy
import os
from datetime import timedelta
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from google.cloud import storage
from osgeo import gdal
from rasterio._io import Affine
with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class PreprocessingService:


    def padding(self, coarse_arr, array_to_be_downsampled):
        array_to_be_downsampled = np.pad(array_to_be_downsampled, ((0, 0), (0, coarse_arr.shape[1] * 2 - array_to_be_downsampled.shape[1]), (0, coarse_arr.shape[2] * 2 - array_to_be_downsampled.shape[2])), 'constant', constant_values = (0, 0))
        return array_to_be_downsampled

    def down_sampling(self, input_arr):
        return np.mean(input_arr)

    def read_tiff(self, file_path):
        with rasterio.open(file_path, 'r') as reader:
            profile = reader.profile
            tif_as_array = reader.read()
        return tif_as_array, profile

    def write_tiff(self, file_path, arr, profile):
        with rasterio.Env():
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(arr.astype(rasterio.float32))

    def upload_to_gcloud(self, file):
        print('Upload to gcloud')
        file_name = file.split('/')[-1]
        storage_client = storage.Client()
        bucket = storage_client.bucket('ai4wildfire')
        if not storage.Blob(bucket=bucket, name='dataset/'+ file_name).exists(storage_client):
            upload_cmd = 'gsutil cp ' + file + ' gs://ai4wildfire/' + 'dataset/' + file_name
            print(upload_cmd)
            os.system(upload_cmd)
            print('finish uploading' + file)
        else:
            print('file exist already')

    def dataset_generator_proj1(self, location):
        location_and_satellite = location + 'GOES'
        data_path = 'data/label/' + location + ' label'
        data_path = Path(data_path)
        save_path = 'data/train/' + location + 'GOES'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        data_file_list = glob(str(data_path / "*.tif"))
        dataset = []
        data_file_list.sort()
        for file in data_file_list:
            goes_file = file.replace('_downsampled', '').replace(
                'data/label/' + location + ' label/Cal_fire_' + location + 'FIRMS-',
                'data/' + location_and_satellite + '/' + location_and_satellite + '_Cal_fire_' + location_and_satellite + '-')

            with rasterio.open(file, 'r') as reader:
                firms_arr_source = reader.read()  # read all raster values
            firms_composition = firms_arr_source[1, :, :]
            firms_after_processing = firms_composition
            firms_after_processing = (firms_after_processing- firms_after_processing.min()) / (firms_after_processing.max() - firms_after_processing.min())
            # plt.imshow(firms_after_processing)
            # plt.show()
            x_size = firms_after_processing.shape[0]
            y_size = firms_after_processing.shape[1]
            vectorized_label = firms_after_processing[x_size - 6:4:-1, 5:y_size - 5].flatten()

            with rasterio.open(goes_file, 'r') as reader:
                goes_arr = reader.read()  # read all raster values
            goes_vis_params = {"red_band": 10, "green_band": 22, "blue_band": 24}
            nan_value = 0
            if np.isnan(nan_value):
                continue
            goes_composition = np.nan_to_num(goes_arr[0, :, :], nan=nan_value)
            goes_after_processing = goes_composition[:, :]
            goes_resized = cv2.resize(goes_after_processing, (y_size, x_size), interpolation=cv2.INTER_LINEAR)
            goes_resized = (goes_resized-goes_resized.min())/(goes_resized.max()-goes_resized.min())

            vectorized_feature = np.zeros(((x_size - 10) * (y_size - 10), 121))
            for i in range(5, x_size - 5):
                for j in range(5, y_size - 5):
                    index = (i - 5) * (y_size - 10) + j - 5
                    vectorized_feature[index, :] = goes_resized[i - 5:i + 6, j - 5:j + 6].flatten()
            dataset.append(np.concatenate((vectorized_feature, vectorized_label[:, np.newaxis]), axis=1))
        dataset_output = np.stack(dataset, axis=0)
        np.save(save_path + '/' + location + ' dataset_trial5.npy', dataset_output.astype(np.float32))

    def dataset_generator_proj2(self, location, satellites):
        modis_path = Path('data/' + location + '/' + 'MODIS')
        s3_path = 'data/' + location + '/' + 'S3'
        label_path = 'data/' + location + '/' + 'FIRMS'
        modis_file_list = glob(str(modis_path / "*.tif"))
        modis_file_list.sort()
        template_arr, _ = self.read_tiff(modis_file_list[0])
        s3_bands_1k = ["S7", "S8", "F1", "F2"]
        s3_bands_500 = ["S6", "S3", "S1"]
        size_dataset = len(modis_file_list)
        arr_500 = np.zeros((size_dataset, 11, 256, 256))
        arr_acc = np.zeros((7,256,256))
        for i in range(size_dataset):
            current_date = modis_file_list[i][-20:-10]
            label_file = label_path + '/' + str(current_date)+'_FIRMS.tif'

            for j in range(len(s3_bands_500)):
                s3_file_500 = s3_path + '/' + str(current_date)+'T17_S3.band' + s3_bands_500[j] + '.tif'
                s3_arr, s3_profile = self.read_tiff(s3_file_500.replace('-', ''))

                s3_arr = cv2.resize(s3_arr[0], (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
                s3_arr = np.nan_to_num(s3_arr)
                s3_arr = np.ma.masked_equal(s3_arr, 0)
                s3_arr = (s3_arr - s3_arr.min()) / (s3_arr.max() - s3_arr.min())
                s3_arr = np.ma.filled(s3_arr, 0)
                arr_500[i,j,:,:] = s3_arr
                arr_acc[j,:,:] = np.maximum(arr_acc[j,:,:], s3_arr)
            for j in range(len(s3_bands_1k)):
                s3_file_1k = s3_path + '/' + str(current_date)+'T17_S3.band' + s3_bands_1k[j] + '.tif'
                s3_arr, s3_profile = self.read_tiff(s3_file_1k.replace('-', ''))

                # s3_arr = (s3_arr-200)/(s3_arr.max()-200)


                s3_arr = cv2.resize(s3_arr[0], (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
                # ret, s3_arr = cv2.threshold(s3_arr, 200, 1000, cv2.THRESH_TOZERO)
                s3_arr = np.nan_to_num(s3_arr)
                s3_arr = np.ma.masked_equal(s3_arr, 0)
                s3_arr = (s3_arr - s3_arr.min()) / (s3_arr.max() - s3_arr.min())
                s3_arr = np.ma.filled(s3_arr, 0)

                arr_500[i,j+3,:,:] = s3_arr
                arr_acc[j+3,:,:] = np.maximum(arr_acc[j+1,:,:], s3_arr)

            new_label_arr, label_profile = self.read_tiff(label_file)
            new_label_arr = np.nan_to_num(new_label_arr)
            new_label_arr = cv2.resize(new_label_arr[0], (256, 256),
                                interpolation=cv2.INTER_LINEAR)
            ret, new_label_arr = cv2.threshold(new_label_arr,0,1,cv2.THRESH_BINARY)
            if i == 0:
                label_arr = new_label_arr
            else:
                label_arr = np.logical_or(label_arr,new_label_arr)


            modis_arr, modis_profile = self.read_tiff(modis_file_list[i])
            modis_arr = np.nan_to_num(modis_arr)
            for k in range(3):
                arr_500[i, k+7, :, :] = cv2.resize(modis_arr[k,:,:], (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
            arr_500[i, 10, :, :] = label_arr
        plt.imshow(arr_acc[5,:,:])
        plt.show()

            # fig, ax = plt.subplots(1, 4)
            #
            # img = arr_500[i, 0:3, :, :].transpose((1,2,0))
            # img = (img - img.min()) / (img.max() - img.min())
            # ax[0].imshow(img[:, :])
            #
            # img = arr_500[i, 3:6, :, :].transpose((1,2,0))
            # img = (img - img.min()) / (img.max() - img.min())
            # ax[1].imshow(img[:, :])
            #
            # img = arr_500[i, 7:10, :, :].transpose((1,2,0))
            # img = (img - img.min()) / (img.max() - img.min())
            # ax[2].imshow(img[:, :])
            #
            # img = arr_500[i, 10, :, :]
            # img = (img - img.min()) / (img.max() - img.min())
            # ax[3].imshow(img[:, :])
            # plt.savefig("dataset_inspection/proj2" +location+ str(i) + ".png", dpi=150)
            # plt.show()

        return arr_500

    def corp_tiff_to_same_size(self, location, reference_mode):
        s2_path = 'data/' + location + 'Sentinel2'
        if reference_mode:
            goes_path = 'data/evaluate/' + location + '/reference'
            goes_fire_path = 'data/evaluate/' + location + '/goes_fire'
        else:
            goes_path = 'data/' + location +'GOES'
            goes_fire_path = 'data/' + location + 'GOES_FIRE'
        s2_path = Path(s2_path)
        goes_path = Path(goes_path)
        goes_file_list = glob(str(goes_path / "*.tif"))
        s2_file_list = glob(str(s2_path / "*.tif"))
        goes_fire_path = Path(goes_fire_path)
        goes_fire_file_list = glob(str(goes_fire_path / "*.tif"))
        _, goes_profile = self.read_tiff(goes_file_list[0])
        _, s2_profile = self.read_tiff(s2_file_list[0])
        goes_bbox = [goes_profile.data.get('transform').column_vectors[2][0],
                    goes_profile.data.get('transform').column_vectors[2][0] +
                    goes_profile.data.get('transform').column_vectors[0][0] * goes_profile.data.get('width'),
                    goes_profile.data.get('transform').column_vectors[2][1] +
                    goes_profile.data.get('transform').column_vectors[1][1] * goes_profile.data.get('height'),
                    goes_profile.data.get('transform').column_vectors[2][1]]
        s2_bbox = [s2_profile.data.get('transform').column_vectors[2][0],
                    s2_profile.data.get('transform').column_vectors[2][0] +
                    s2_profile.data.get('transform').column_vectors[0][0] * s2_profile.data.get('width'),
                    s2_profile.data.get('transform').column_vectors[2][1] +
                    s2_profile.data.get('transform').column_vectors[1][1] * s2_profile.data.get('height'),
                    s2_profile.data.get('transform').column_vectors[2][1]]
        lon_low = max(s2_bbox[0], goes_bbox[0])
        lon_high = min(s2_bbox[1], goes_bbox[1])
        lat_low = max(s2_bbox[2], goes_bbox[2])
        lat_high = min(s2_bbox[3], goes_bbox[3])
        if lon_low % 6000 != 0:
            lon_low = (lon_low // 6000 + 1) * 6000
        if lon_high % 6000 != 0:
            lon_high = (lon_high // 6000) * 6000
        if lat_low % 6000 != 0:
            lat_low = (lat_low // 6000 + 1) * 6000
        if lat_high % 6000 != 0:
            lat_high = (lat_high // 6000) * 6000

        for file in s2_file_list:
            gdal.Warp(file, file, outputBounds=(lon_low,lat_low,lon_high,lat_high))
        for goes_file in goes_file_list:
            gdal.Warp(goes_file, goes_file, outputBounds=(lon_low, lat_low, lon_high, lat_high))
        for goes_fire_file in goes_fire_file_list:
            gdal.Warp(goes_fire_file, goes_fire_file, outputBounds=(lon_low, lat_low, lon_high, lat_high))

    def corp_tiff_to_same_size_referencing(self, location):
        s2_path = 'data/' + location + 'Sentinel2'
        goes_path = 'data/evaluate/'+location+'/reference'
        goes_fire_path = 'data/evaluate/'+location+'/goes_fire'
        s2_path = Path(s2_path)
        goes_path = Path(goes_path)
        goes_file_list = glob(str(goes_path / "*.tif"))
        goes_fire_path = Path(goes_fire_path)
        goes_fire_file_list = glob(str(goes_fire_path / "*.tif"))
        s2_file_list = glob(str(s2_path / "*.tif"))
        _, goes_profile = self.read_tiff(goes_file_list[0])
        _, s2_profile = self.read_tiff(s2_file_list[0])
        goes_bbox = [goes_profile.data.get('transform').column_vectors[2][0],
                    goes_profile.data.get('transform').column_vectors[2][0] +
                    goes_profile.data.get('transform').column_vectors[0][0] * goes_profile.data.get('width'),
                    goes_profile.data.get('transform').column_vectors[2][1] +
                    goes_profile.data.get('transform').column_vectors[1][1] * goes_profile.data.get('height'),
                    goes_profile.data.get('transform').column_vectors[2][1]]
        s2_bbox = [s2_profile.data.get('transform').column_vectors[2][0],
                    s2_profile.data.get('transform').column_vectors[2][0] +
                    s2_profile.data.get('transform').column_vectors[0][0] * s2_profile.data.get('width'),
                    s2_profile.data.get('transform').column_vectors[2][1] +
                    s2_profile.data.get('transform').column_vectors[1][1] * s2_profile.data.get('height'),
                    s2_profile.data.get('transform').column_vectors[2][1]]
        lon_low = max(s2_bbox[0], goes_bbox[0])
        lon_high = min(s2_bbox[1], goes_bbox[1])
        lat_low = max(s2_bbox[2], goes_bbox[2])
        lat_high = min(s2_bbox[3], goes_bbox[3])
        if lon_low % 6000 != 0:
            lon_low = (lon_low // 6000 + 1) * 6000
        if lon_high % 6000 != 0:
            lon_high = (lon_high // 6000) * 6000
        if lat_low % 6000 != 0:
            lat_low = (lat_low // 6000 + 1) * 6000
        if lat_high % 6000 != 0:
            lat_high = (lat_high // 6000) * 6000

        for file in s2_file_list:
            gdal.Warp(file, file, outputBounds=(lon_low,lat_low,lon_high,lat_high))
        for goes_file in goes_file_list:
            gdal.Warp(goes_file, goes_file, outputBounds=(lon_low, lat_low, lon_high, lat_high))
        for goes_fire_file in goes_fire_file_list:
            gdal.Warp(goes_fire_file, goes_fire_file, outputBounds=(lon_low, lat_low, lon_high, lat_high))

    def normalization(self, array):
        n_channels = array.shape[0]-2
        for i in range(n_channels):
            nanmean = np.nanmean(array[i, :, :])
            array[i, :, :] = np.nan_to_num(array[i, :, :], nan=nanmean)
            array[i,:,:] = (array[i,:,:]-array[i,:,:].mean())/array[i,:,:].std()
        return np.nan_to_num(array)

    def dataset_generator_proj3(self, locations, window_size):
        satellite = 'VIIRS_Day'
        ts_length = 10
        stack_over_location=[]
        save_path = 'data_train_proj3/'
        n_channels = 5
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = 'data/' + location + '/' + satellite + '/'
            file_list = glob(data_path+'/*.tif')
            file_list.sort()
            if len(file_list) % ts_length != 0:
                num_sequence = len(file_list)//ts_length+1
            else:
                num_sequence = len(file_list) // ts_length
            preprocessing = PreprocessingService()
            array, _ = preprocessing.read_tiff(file_list[0])
            padding = window_size // 2
            array_stack = []
            for j in range(num_sequence):
                output_array = np.zeros(
                    (10, (array.shape[1] - padding * 2) * (array.shape[2] - padding * 2), pow(window_size, 2) * n_channels + 1))
                if j == num_sequence-1 and j != 0:
                    file_list_size = len(file_list) % ts_length
                else:
                    file_list_size=ts_length
                for i in range(file_list_size):
                    file = file_list[i+j*10]
                    array, _ = preprocessing.read_tiff(file)
                    # pick up channels here
                    # array = array[3:, :, :]
                    array = self.normalization(array)

                    for ix in range(padding, array.shape[1]-padding):
                        for iy in range(padding, array.shape[2]-padding):
                            idx = (ix-padding)*(array.shape[2]-padding*2)+iy-padding
                            window = array[:n_channels, ix-(window_size-1)//2:ix+(window_size-1)//2+1, iy-(window_size-1)//2:iy+(window_size-1)//2+1].flatten()
                            label = array[n_channels+1, ix, iy]
                            # mask = array[n_channels+1, ix, iy]
                            output_array[i, idx, :pow(window_size, 2) * n_channels] = window
                            output_array[i, idx, pow(window_size, 2) * n_channels] = label > 0
                            # output_array[i, idx, pow(window_size, 2) * n_channels+1] = mask > 0
                    time_stamp = i
                    shape = (array.shape[1]-padding*2, array.shape[2]-padding*2)
                    plt.subplot(211)
                    plt.imshow((output_array[time_stamp, :, 45].reshape(shape)-output_array[time_stamp, :, 45].reshape(shape).min())-(output_array[time_stamp, :, 45].reshape(shape).max()-output_array[time_stamp, :, 45].reshape(shape).min()))
                    plt.subplot(212)
                    plt.imshow((output_array[time_stamp, :, 30].reshape(shape)-output_array[time_stamp, :, 30].reshape(shape).min())-(output_array[time_stamp, :, 30].reshape(shape).max()-output_array[time_stamp, :, 30].reshape(shape).min()))
                    plt.savefig('plt/'+location+str(i)+'.png')
                    plt.show()
                array_stack.append(output_array)
            output_array_stacked = np.concatenate(array_stack, axis=1)
            stack_over_location.append(output_array_stacked)
        output_array_stacked_over_location = np.concatenate(stack_over_location, axis=1)
        print(output_array_stacked_over_location.shape)
        # output_array_stacked_over_location = self.normalization(output_array_stacked_over_location, n_channels)
        file_name = 'proj3_test_5_channel.npy'
        np.save(save_path+file_name, output_array_stacked_over_location.astype(np.float))
        self.upload_to_gcloud(save_path+file_name)

    def dataset_generator_proj3_image(self, locations, file_name, image_size=(224, 224)):
        satellite = 'VIIRS_Day'
        window_size = 3
        ts_length = 10
        stack_over_location=[]
        save_path = 'data_train_proj3/'
        n_channels = 5
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = 'data/' + location + '/' + satellite + '/'
            file_list = glob(data_path+'/*.tif')
            file_list.sort()
            if len(file_list) % ts_length != 0:
                num_sequence = len(file_list)//ts_length+1
            else:
                num_sequence = len(file_list) // ts_length
            preprocessing = PreprocessingService()
            array, _ = preprocessing.read_tiff(file_list[0])
            padding = window_size // 2
            array_stack = []
            # th = [(350,335), (350,335), (350,335), (350,335), (350,335), (350,335), (335,335), (335,335), (340,335), (344,335)]
            # th = [(340,330), (337, 335), (335, 335), (335, 330), (335, 330), (330, 330), (330, 330), (330, 330), (330, 330), (330, 330)]
            for j in range(num_sequence):
                output_array = np.zeros((ts_length, n_channels+2, image_size[0], image_size[1]))
                if j == num_sequence-1 and j != 0:
                    file_list_size = len(file_list) % ts_length
                else:
                    file_list_size=ts_length
                for i in range(file_list_size):
                    file = file_list[i+j*10]
                    array, _ = preprocessing.read_tiff(file)
                    # pick up channels here
                    # array = array[3:, :, :]
                    th_i = th[i]
                    plt.subplot(231)
                    plt.imshow(array[3, :, :])
                    af = np.zeros(array[3,:,:].shape)
                    # Avoid direct modifing original array
                    af[:,:] = np.logical_or(array[3,:,:] > th_i[0], array[4,:,:] > th_i[1])
                    # af[np.logical_not(af[:,:])] = np.nan

                    plt.imshow(af, cmap='Reds', alpha=1)
                    plt.subplot(232)
                    plt.imshow(array[3, :, :])
                    plt.imshow(array[6, :, :], cmap='Reds', alpha=1)
                    plt.subplot(233)
                    plt.imshow(array[3, :, :])
                    plt.subplot(234)
                    plt.imshow(af)
                    plt.subplot(235)
                    plt.imshow(array[6, :, :])
                    plt.show()

                    array = self.normalization(array)
                    row_start=int(array.shape[1]*0.1)
                    col_start=int(array.shape[2]*0)
                    array = np.concatenate((array, af[np.newaxis,:,:]))
                    array = array[:, row_start:row_start+image_size[0], col_start:col_start+image_size[1]]
                    output_array[i,:n_channels,:array.shape[1],:array.shape[2]] = array[:n_channels, :, :]
                    output_array[i,n_channels:n_channels+2,:array.shape[1],:array.shape[2]] = np.nan_to_num(array[n_channels+1:n_channels+3, :, :])
                    plt.figure(figsize= (12, 4), dpi=80)
                    plt.subplot(121)
                    plt.imshow((output_array[i,3,:,:]-output_array[i,3,:,:].min())/(output_array[i,3,:,:].max()-output_array[i,3,:,:].min()))
                    masked_img = np.ma.masked_where(output_array[i,6,:,:] == 0, output_array[i,6,:,:])
                    plt.imshow(masked_img, interpolation='nearest')
                    plt.subplot(122)
                    plt.imshow((output_array[i, 3, :, :] - output_array[i, 3, :, :].min()) / (
                                output_array[i, 3, :, :].max() - output_array[i, 3, :, :].min()))
                    img = np.zeros((224, 224))
                    img[:, :] = output_array[i, 5, :, :] > 0
                    img[img == False] = np.nan
                    plt.imshow(img , interpolation='nearest')
                    # plt.savefig('plt_proj3_img/'+location+str(i)+'.png')
                    plt.show()
                array_stack.append(output_array)
            output_array_stacked = np.stack(array_stack, axis=0)
            stack_over_location.append(output_array_stacked)
        output_array_stacked_over_location = np.concatenate(stack_over_location, axis=0)
        print(output_array_stacked_over_location.shape)
        # output_array_stacked_over_location = self.normalization(output_array_stacked_over_location, n_channels)

        np.save(save_path+file_name, output_array_stacked_over_location.astype(np.float))
        # self.upload_to_gcloud(save_path+file_name)

    def dataset_generator_proj3_image_test(self, location, file_name, image_size=(224, 224)):
        satellite = 'VIIRS_Day'
        window_size = 3
        ts_length = 10
        stack_over_location=[]
        save_path = 'data_train_proj3/'
        n_channels = 5
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print(location)
        data_path = 'data/' + location + '/' + satellite + '/'
        file_list = glob(data_path+'/*.tif')
        file_list.sort()
        if len(file_list) % ts_length != 0:
            num_sequence = len(file_list)//ts_length+1
        else:
            num_sequence = len(file_list) // ts_length
        preprocessing = PreprocessingService()
        array, _ = preprocessing.read_tiff(file_list[0])
        padding = window_size // 2
        array_stack = []
        # th = [(350,335), (350,335), (350,335), (350,335), (350,335), (350,335), (335,335), (335,335), (340,335), (344,335)]
        # th = [(350,335), (350,335), (350,335), (350,335), (350,335), (350,335), (335,335), (335,335), (340,335), (344,335)]

        # th = [(350,335), (345,335), (336,335), (331,335), (334,335), (332,335), (332,335), (330,330), (330,335), (329,340)] # elephant_hill
        # th = [(340,330), (339, 337), (328,335), (333, 330), (335, 330), (330, 330), (330, 330), (335, 330), (337, 330), (340, 330)] #sparkslake
        # th = [(330,330), (325, 337), (335,335), (340, 330), (335, 330), (330, 330), (330, 330), (335, 330), (337, 330), (340, 330)] # blue_ridge_fire
        # th = [(335,330), (335, 337), (333,335), (335, 330), (335, 330), (330, 330), (330, 330), (335, 330), (337, 330), (340, 330)] # eagle_bluff_fire
        # th = [(335,330), (330, 337), (315,335), (330, 330), (330, 330), (330, 330), (325, 330), (335, 330), (337, 330), (340, 330)] # thomas_fire
        # th = [(325,330), (330, 337), (335,325), (330, 330), (330, 330), (320, 330), (325, 330), (335, 330), (330, 330), (340, 330)] # sydney_fire
        th = [(325,330), (330, 337), (330,325), (320, 330), (330, 330), (320, 330), (325, 330), (335, 330), (330, 330), (330, 330)] # swedish_fire
        # th = [(330,330), (320, 337), (335,335), (330, 330), (330, 330), (330, 330), (320, 330), (330, 330), (337, 330), (340, 330)] # kincade_fire
        # th = [(330,330), (320, 337), (335,335), (330, 330), (330, 330), (330, 330), (320, 330), (330, 330), (337, 330), (340, 330)] # kincade_fire
        # th = [(330,330), (320, 337), (335,335), (330, 330), (330, 330), (330, 330), (320, 330), (330, 330), (337, 330), (340, 330)] # walker_fire
        # th = [(333,330), (339, 337), (343,335), (343, 330), (337, 330), (328, 330), (330, 330), (335, 330), (337, 330), (340, 330)] # carr_fire
        # th = [(333,330), (339, 337), (343,335), (343, 330), (337, 330), (328, 330), (330, 330), (335, 330), (337, 330), (340, 330)] # walker_fire
        # th = [(333,330), (329, 337), (340,335), (343, 330), (325, 330), (328, 330), (325, 330), (335, 330), (337, 330), (330, 330)] # hanceville_fire
        # th = [(340,330), (320, 337), (320,310), (310, 330), (310, 330), (293, 330), (310, 330), (315, 330), (310, 330), (310, 330)] # elephant_hill_fire
        # th = [(340,330), (320, 337), (320,310), (310, 330), (310, 330), (293, 330), (310, 330), (315, 330), (310, 330), (310, 330)] # camp_fire
        # th = [(340,330), (320, 337), (325,330), (330, 330), (330, 330), (340, 330), (330, 330), (325, 330), (325, 330), (338, 330)] # chuckegg_creek_fire
        # th = [(340,330), (320, 337), (330,330), (320, 330), (320, 330), (325, 330), (330, 330), (330, 330), (330, 330), (330, 330)] # tubbs_fire
        for k in range(2):
            for j in range(num_sequence):
                output_array = np.zeros((ts_length, n_channels+2, image_size[0], image_size[1]))
                if j == num_sequence-1 and j != 0:
                    file_list_size = len(file_list) % ts_length
                else:
                    file_list_size=ts_length
                for i in range(file_list_size):
                    file = file_list[i+j*10]
                    array, profile = preprocessing.read_tiff(file)
                    # pick up channels here
                    # array = array[3:, :, :]
                    th_i = th[i]
                    plt.subplot(131)
                    plt.imshow(array[3, :, :])
                    af = np.zeros(array[3,:,:].shape)
                    # Avoid direct modifing original array
                    af[:,:] = np.logical_or(array[3,:,:] > th_i[0], array[4,:,:] > th_i[1])
                    af_img = af
                    af_img[np.logical_not(af_img[:,:])] = np.nan
                    plt.title('Manual label')
                    plt.imshow(af_img, cmap='hsv', interpolation='nearest')
                    plt.subplot(132)
                    plt.title('VIIRS AF product')
                    plt.imshow(array[3, :, :])
                    array[6, :, :][np.where(~np.isnan(array[6, :, :]))]=1
                    plt.imshow(array[6, :, :], cmap='hsv', interpolation='nearest')
                    plt.subplot(133)
                    plt.title('original image')
                    plt.imshow(array[3, :, :])
                    plt.show()

                    array = self.normalization(array)
                    if k == 0:
                        col_start=int(array.shape[2]*0)
                    else:
                        col_start=int(array.shape[2]-224)
                    row_start=int(array.shape[1]*0.3)
                    array = np.concatenate((array, af[np.newaxis,:,:]))
                    array = array[:, row_start:row_start+image_size[0], col_start:col_start+image_size[1]]
                    output_array[i,:n_channels,:array.shape[1],:array.shape[2]] = array[:n_channels, :, :]
                    output_array[i,n_channels:n_channels+2,:array.shape[1],:array.shape[2]] = np.nan_to_num(array[n_channels+1:n_channels+3, :, :])
                    # plt.figure(figsize= (12, 4), dpi=80)
                    # plt.subplot(121)
                    # plt.imshow((output_array[i,3,:,:]-output_array[i,3,:,:].min())/(output_array[i,3,:,:].max()-output_array[i,3,:,:].min()))
                    # masked_img = np.ma.masked_where(output_array[i,6,:,:] == 0, output_array[i,6,:,:])
                    # plt.imshow(masked_img, interpolation='nearest')
                    # plt.subplot(122)
                    # plt.imshow((output_array[i, 3, :, :] - output_array[i, 3, :, :].min()) / (
                    #         output_array[i, 3, :, :].max() - output_array[i, 3, :, :].min()))
                    # img = np.zeros((224, 224))
                    # img[:, :] = output_array[i, 5, :, :] > 0
                    # img[img == False] = np.nan
                    # plt.imshow(img , interpolation='nearest')
                    # # plt.savefig('plt_proj3_img/'+location+str(i)+'.png')
                    # plt.show()
            array_stack.append(output_array)
        output_array_stacked = np.stack(array_stack, axis=0)

        np.save(save_path+file_name, output_array_stacked.astype(np.float))

    def reconstruct_tif_proj3(self, location, satellite='VIIRS_Day', image_size=(224, 224)):
        data_path = 'data/' + location + '/' + satellite + '/'
        file_list = glob(data_path+'/*.tif')
        file_list.sort()
        array, profile = self.read_tiff(file_list[0])
        row_start=int(array.shape[1]*0.1)
        col_start=int(array.shape[2]*0)

        save_path = 'data_result_project3/' + location + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        start_date = config.get(location).get('start')
        output_array = np.load('data_result_project3/'+ location + '.npy')

        duration = output_array.shape[0]
        for i in range(duration):
            output_array_t = copy.deepcopy(output_array[i])
            current_date = start_date+timedelta(i)
            assert output_array_t.shape[0] == image_size[0]
            assert output_array_t.shape[1] == image_size[1]

            new_profile = copy.deepcopy(profile)
            new_profile.data['width'] = image_size[0]
            new_profile.data['height'] = image_size[1]

            new_transform = Affine(375.0, 0, profile.data['transform'].xoff+375*col_start, 0, -375, profile.data['transform'].yoff-(375.0*row_start))
            new_profile.data['transform'] = new_transform
            new_profile.data['count']=1
            plt.imshow(output_array_t)
            plt.show()
            # output_array_t[np.where(output_array_t==0)] = np.nan
            print('save images to'+save_path+location+'_'+str(current_date)+'.tif')
            self.write_tiff(save_path+location+'_'+str(current_date)+'.tif', output_array_t[np.newaxis,:,:], new_profile)




