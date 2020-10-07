import os
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
import yaml

with open("DataPreparation/config/configuration.yml", "r", encoding="utf8") as f:
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

    def get_feature_per_pixel(self, array, index_x, index_y):
        res = np.zeros(27)
        return np.concatenate((array[index_x - 1, index_y - 1, :], array[index_x, index_y - 1, :],
                               array[index_x + 1, index_y - 1, :],
                               array[index_x - 1, index_y, :], array[index_x, index_y, :],
                               array[index_x + 1, index_y, :],
                               array[index_x - 1, index_y + 1, :], array[index_x, index_y + 1, :],
                               array[index_x + 1, index_y + 1, :]))

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

    def dataset_generator(self, location, satellites):
        data_path = 'data/August_complexGOES'
        data_path = Path(data_path)
        save_path = 'train/August_complexGOES'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        data_file_list = glob(str(data_path / "*.tif"))
        dataset = []
        for file in data_file_list:
            label_file = file.replace('data/August_complexGOES/August_complexGOES_Cal_fire_August_complexGOES-',
                                      'label/August_complex label/Cal_fire_August_complexFIRMS-').replace('.tif',
                                                                                                          '_downsampled.tif')
            with rasterio.open(label_file, 'r') as reader:
                firms_profile = reader.profile
                firms_arr_source = reader.read()  # read all raster values
            firms_vis_params = {"confidence": 1}

            with rasterio.open(file, 'r') as reader:
                goes_profile = reader.profile
                goes_arr = reader.read()  # read all raster values
            goes_vis_params = {"red_band": 10, "green_band": 22, "blue_band": 24}

            goes_composition = self.get_composition(goes_arr, goes_vis_params)
            goes_composition = goes_composition.transpose((1, 2, 0))
            goes_after_processing = goes_composition / goes_composition.max(axis=2, keepdims=True)
            x_size = goes_composition.shape[0]
            y_size = goes_composition.shape[1]

            firms_composition = firms_arr_source[1, :, :]
            if np.max(firms_composition) != 0:
                firms_after_processing = firms_composition / np.max(firms_composition)
            else:
                firms_after_processing = firms_composition


            vectorized_feature = np.zeros(((x_size - 2) * (y_size - 2), 27))
            for i in range(1, x_size - 1):
                for j in range(1, y_size - 1):
                    index = i - 1 + (j - 1) * (x_size - 2)
                    vectorized_feature[index, :] = self.get_feature_per_pixel(goes_composition, i, j)
            vectorized_label = np.zeros(((x_size - 2) * (y_size - 2)))
            vectorized_label = firms_after_processing[1:x_size - 1, 1:y_size - 1].flatten()

            vectorized_set = np.concatenate((vectorized_feature, vectorized_label[:, np.newaxis]), axis=1)
            dataset.append(vectorized_set)

        dataset_output = np.stack(dataset, axis=0)
        np.save(save_path + "/dataset.npy", dataset_output.transpose((1, 0, 2)))









