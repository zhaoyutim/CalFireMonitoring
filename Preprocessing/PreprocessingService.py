import datetime
import os
from glob import glob
from pathlib import Path

from osgeo import gdal
import matplotlib.pyplot as plt
from array2gif import write_gif
from imageio import imread, imsave
from model import resolve_single
from model.srgan import generator, discriminator
import cv2
import numpy as np
import rasterio
import yaml

with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class PreprocessingService:

    def get_composition(self, input_arr, vis_params):
        if len(vis_params) == 1:
            return input_arr
        else:
            composition = np.zeros((3, input_arr.shape[1], input_arr.shape[2]))
            red = vis_params.get("red_band")
            green = vis_params.get("green_band")
            blue = vis_params.get("blue_band")
            composition[0, :, :] = input_arr[red, :, :]
            composition[1, :, :] = input_arr[green, :, :]
            composition[2, :, :] = input_arr[blue, :, :]
            return composition

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
        viirs_path = Path('data/' + location + 'VIIRS')
        goes_path = 'data/' + location + 'GOES'
        label_path = 'data/progression/'+ location
        viirs_file_list = glob(str(viirs_path / "*.tif"))
        viirs_file_list.sort()
        template_arr, _ = self.read_tiff(viirs_file_list[0])
        dataset_proj2 = np.zeros((len(viirs_file_list), 7, 256, 160))
        goes_acc = np.zeros(template_arr.shape)
        for i in range(len(viirs_file_list)):
            current_date = viirs_file_list[i][-14:-4]
            goes_file = goes_path + '/' + location + 'GOES_Cal_fire_' + location+'GOES-'+str(current_date)+'.tif'
            label_file = label_path + '/' + location + str(current_date)+'.tif'
            goes_arr, goes_profile = self.read_tiff(goes_file)
            goes_arr = np.maximum(goes_arr, goes_acc)
            goes_acc = goes_arr
            if not os.path.isfile(label_file):
                dataset_proj2 = dataset_proj2[:i,:,:,:]
                break
            label_arr, label_profile = self.read_tiff(label_file)
            label_arr = np.nan_to_num(np.flip(label_arr, axis=0))
            viirs_arr, viirs_profile = self.read_tiff(viirs_file_list[i])
            viirs_arr = np.nan_to_num(viirs_arr)
            goes_arr = np.nan_to_num(goes_arr)
            dataset_proj2[i, :3, :, :] = goes_arr[:, round(goes_arr.shape[1]*0.03):round(goes_arr.shape[1]*0.03)+256, :160]
            dataset_proj2[i, 3:6, :, :] = viirs_arr[:, round(viirs_arr.shape[1]*0.03):round(viirs_arr.shape[1]*0.03)+256, :160]
            dataset_proj2[i, 6, :, :] = label_arr[:, round(label_arr.shape[1]*0.03):round(label_arr.shape[1]*0.03)+256, :160]

        return dataset_proj2

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






