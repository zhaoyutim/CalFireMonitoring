import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import rasterio
from osgeo import gdal

from Preprocessing.PreprocessingService import PreprocessingService


class Proj1DatasetProcessor(PreprocessingService):
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
            firms_after_processing = (firms_after_processing - firms_after_processing.min()) / (
                        firms_after_processing.max() - firms_after_processing.min())
            x_size = firms_after_processing.shape[0]
            y_size = firms_after_processing.shape[1]
            vectorized_label = firms_after_processing[x_size - 6:4:-1, 5:y_size - 5].flatten()

            with rasterio.open(goes_file, 'r') as reader:
                goes_arr = reader.read()  # read all raster values
            nan_value = 0
            if np.isnan(nan_value):
                continue
            goes_composition = np.nan_to_num(goes_arr[0, :, :], nan=nan_value)
            goes_after_processing = goes_composition[:, :]
            goes_resized = cv2.resize(goes_after_processing, (y_size, x_size), interpolation=cv2.INTER_LINEAR)
            goes_resized = (goes_resized - goes_resized.min()) / (goes_resized.max() - goes_resized.min())

            vectorized_feature = np.zeros(((x_size - 10) * (y_size - 10), 121))
            for i in range(5, x_size - 5):
                for j in range(5, y_size - 5):
                    index = (i - 5) * (y_size - 10) + j - 5
                    vectorized_feature[index, :] = goes_resized[i - 5:i + 6, j - 5:j + 6].flatten()
            dataset.append(np.concatenate((vectorized_feature, vectorized_label[:, np.newaxis]), axis=1))
        dataset_output = np.stack(dataset, axis=0)
        np.save(save_path + '/' + location + ' dataset_trial5.npy', dataset_output.astype(np.float32))


    def corp_tiff_to_same_size(self, location, reference_mode):
        s2_path = 'data/' + location + 'Sentinel2'
        if reference_mode:
            goes_path = 'data/evaluate/' + location + '/reference'
            goes_fire_path = 'data/evaluate/' + location + '/goes_fire'
        else:
            goes_path = 'data/' + location + 'GOES'
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
            gdal.Warp(file, file, outputBounds=(lon_low, lat_low, lon_high, lat_high))
        for goes_file in goes_file_list:
            gdal.Warp(goes_file, goes_file, outputBounds=(lon_low, lat_low, lon_high, lat_high))
        for goes_fire_file in goes_fire_file_list:
            gdal.Warp(goes_fire_file, goes_fire_file, outputBounds=(lon_low, lat_low, lon_high, lat_high))


    def corp_tiff_to_same_size_referencing(self, location):
        s2_path = 'data/' + location + 'Sentinel2'
        goes_path = 'data/evaluate/' + location + '/reference'
        goes_fire_path = 'data/evaluate/' + location + '/goes_fire'
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
            gdal.Warp(file, file, outputBounds=(lon_low, lat_low, lon_high, lat_high))
        for goes_file in goes_file_list:
            gdal.Warp(goes_file, goes_file, outputBounds=(lon_low, lat_low, lon_high, lat_high))
        for goes_fire_file in goes_fire_file_list:
            gdal.Warp(goes_fire_file, goes_fire_file, outputBounds=(lon_low, lat_low, lon_high, lat_high))

