import copy
import os
from datetime import timedelta
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import yaml
from rasterio._io import Affine

from Preprocessing.PreprocessingService import PreprocessingService

with open("config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class Proj2DatasetProcessor(PreprocessingService):

    def reconstruct_tif_proj2(self, location, satellite='VIIRS_Day', image_size=(224, 224)):
        data_path = 'data/' + location + '/' + satellite + '/'
        file_list = glob(data_path + '/*.tif')
        file_list.sort()
        array, profile = self.read_tiff(file_list[0])
        row_start = int(array.shape[1] * 0.1)
        col_start = int(array.shape[2] * 0)

        save_path = 'data_result_project3/' + location + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        start_date = config.get(location).get('start')
        output_array = np.load('data_result_project3/' + location + '.npy')

        duration = output_array.shape[0]
        for i in range(duration):
            output_array_t = copy.deepcopy(output_array[i])
            current_date = start_date + timedelta(i)
            assert output_array_t.shape[0] == image_size[0]
            assert output_array_t.shape[1] == image_size[1]

            new_profile = copy.deepcopy(profile)
            new_profile.data['width'] = image_size[0]
            new_profile.data['height'] = image_size[1]

            new_transform = Affine(375.0, 0, profile.data['transform'].xoff + 375 * col_start, 0, -375,
                                   profile.data['transform'].yoff - (375.0 * row_start))
            new_profile.data['transform'] = new_transform
            new_profile.data['count'] = 1
            plt.imshow(output_array_t)
            plt.show()
            # output_array_t[np.where(output_array_t==0)] = np.nan
            print('save images to' + save_path + location + '_' + str(current_date) + '.tif')
            self.write_tiff(save_path + location + '_' + str(current_date) + '.tif',
                            output_array_t[np.newaxis, :, :], new_profile)


    def dataset_generator_proj2(self, locations, window_size):
        satellite = 'VIIRS_Day'
        ts_length = 10
        stack_over_location = []
        save_path = 'data_train_proj3/'
        n_channels = 5
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = 'data/' + location + '/' + satellite + '/'
            file_list = glob(data_path + '/*.tif')
            file_list.sort()
            if len(file_list) % ts_length != 0:
                num_sequence = len(file_list) // ts_length + 1
            else:
                num_sequence = len(file_list) // ts_length
            preprocessing = PreprocessingService()
            array, _ = preprocessing.read_tiff(file_list[0])
            padding = window_size // 2
            array_stack = []
            for j in range(num_sequence):
                output_array = np.zeros(
                    (10, (array.shape[1] - padding * 2) * (array.shape[2] - padding * 2),
                     pow(window_size, 2) * n_channels + 1))
                if j == num_sequence - 1 and j != 0:
                    file_list_size = len(file_list) % ts_length
                else:
                    file_list_size = ts_length
                for i in range(file_list_size):
                    file = file_list[i + j * 10]
                    array, _ = preprocessing.read_tiff(file)
                    # pick up channels here
                    # array = array[3:, :, :]
                    array = self.normalization(array)

                    for ix in range(padding, array.shape[1] - padding):
                        for iy in range(padding, array.shape[2] - padding):
                            idx = (ix - padding) * (array.shape[2] - padding * 2) + iy - padding
                            window = array[:n_channels, ix - (window_size - 1) // 2:ix + (window_size - 1) // 2 + 1,
                                     iy - (window_size - 1) // 2:iy + (window_size - 1) // 2 + 1].flatten()
                            label = array[n_channels + 1, ix, iy]
                            # mask = array[n_channels+1, ix, iy]
                            output_array[i, idx, :pow(window_size, 2) * n_channels] = window
                            output_array[i, idx, pow(window_size, 2) * n_channels] = label > 0
                            # output_array[i, idx, pow(window_size, 2) * n_channels+1] = mask > 0
                    time_stamp = i
                    shape = (array.shape[1] - padding * 2, array.shape[2] - padding * 2)
                    plt.subplot(211)
                    plt.imshow((output_array[time_stamp, :, 45].reshape(shape) - output_array[time_stamp, :,
                                                                                 45].reshape(shape).min()) - (
                                       output_array[time_stamp, :, 45].reshape(shape).max() - output_array[
                                                                                              time_stamp, :,
                                                                                              45].reshape(
                                   shape).min()))
                    plt.subplot(212)
                    plt.imshow((output_array[time_stamp, :, 30].reshape(shape) - output_array[time_stamp, :,
                                                                                 30].reshape(shape).min()) - (
                                       output_array[time_stamp, :, 30].reshape(shape).max() - output_array[
                                                                                              time_stamp, :,
                                                                                              30].reshape(
                                   shape).min()))
                    plt.savefig('plt/' + location + str(i) + '.png')
                    plt.show()
                array_stack.append(output_array)
            output_array_stacked = np.concatenate(array_stack, axis=1)
            stack_over_location.append(output_array_stacked)
        output_array_stacked_over_location = np.concatenate(stack_over_location, axis=1)
        print(output_array_stacked_over_location.shape)
        # output_array_stacked_over_location = self.normalization(output_array_stacked_over_location, n_channels)
        file_name = 'proj3_test_5_channel.npy'
        np.save(save_path + file_name, output_array_stacked_over_location.astype(np.float))

    def dataset_generator_proj2_image(self, locations, file_name, image_size=(224, 224)):
        satellite = 'VIIRS_Day'
        window_size = 3
        ts_length = 10
        stack_over_location = []
        save_path = 'data_train_proj2/'
        n_channels = 5
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = 'data/' + location + '/' + satellite + '/'
            file_list = glob(data_path + '/*.tif')
            file_list.sort()
            if len(file_list) % ts_length != 0:
                num_sequence = len(file_list) // ts_length + 1
            else:
                num_sequence = len(file_list) // ts_length
            preprocessing = PreprocessingService()
            array, _ = preprocessing.read_tiff(file_list[0])
            padding = window_size // 2
            array_stack = []
            th = [(350,335), (350,335), (350,335), (350,335), (350,335), (350,335), (335,335), (335,335), (340,335), (344,335)]
            # th = [(340,330), (337, 335), (335, 335), (335, 330), (335, 330), (330, 330), (330, 330), (330, 330), (330, 330), (330, 330)]
            for j in range(num_sequence):
                output_array = np.zeros((ts_length, n_channels + 2, image_size[0], image_size[1]))
                if j == num_sequence - 1 and j != 0:
                    file_list_size = len(file_list) % ts_length
                else:
                    file_list_size = ts_length
                for i in range(file_list_size):
                    file = file_list[i + j * 10]
                    array, _ = preprocessing.read_tiff(file)
                    # pick up channels here
                    # array = array[3:, :, :]
                    th_i = th[i]
                    plt.subplot(231)
                    plt.imshow(array[3, :, :])
                    af = np.zeros(array[3, :, :].shape)
                    # Avoid direct modifing original array
                    af[:, :] = np.logical_or(array[3, :, :] > th_i[0], array[4, :, :] > th_i[1])
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
                    row_start = int(array.shape[1] * 0.1)
                    col_start = int(array.shape[2] * 0)
                    array = np.concatenate((array, af[np.newaxis, :, :]))
                    array = array[:, row_start:row_start + image_size[0], col_start:col_start + image_size[1]]
                    output_array[i, :n_channels, :array.shape[1], :array.shape[2]] = array[:n_channels, :, :]
                    output_array[i, n_channels:n_channels + 2, :array.shape[1], :array.shape[2]] = np.nan_to_num(
                        array[n_channels + 1:n_channels + 3, :, :])
                    plt.figure(figsize=(12, 4), dpi=80)
                    plt.subplot(121)
                    plt.imshow((output_array[i, 3, :, :] - output_array[i, 3, :, :].min()) / (
                            output_array[i, 3, :, :].max() - output_array[i, 3, :, :].min()))
                    masked_img = np.ma.masked_where(output_array[i, 6, :, :] == 0, output_array[i, 6, :, :])
                    plt.imshow(masked_img, interpolation='nearest')
                    plt.subplot(122)
                    plt.imshow((output_array[i, 3, :, :] - output_array[i, 3, :, :].min()) / (
                            output_array[i, 3, :, :].max() - output_array[i, 3, :, :].min()))
                    img = np.zeros((224, 224))
                    img[:, :] = output_array[i, 5, :, :] > 0
                    img[img == False] = np.nan
                    plt.imshow(img, interpolation='nearest')
                    # plt.savefig('plt_proj2_img/'+location+str(i)+'.png')
                    plt.show()
                array_stack.append(output_array)
            output_array_stacked = np.stack(array_stack, axis=0)
            stack_over_location.append(output_array_stacked)
        output_array_stacked_over_location = np.concatenate(stack_over_location, axis=0)
        print(output_array_stacked_over_location.shape)
        # output_array_stacked_over_location = self.normalization(output_array_stacked_over_location, n_channels)

        np.save(save_path + file_name, output_array_stacked_over_location.astype(np.float))
        # self.upload_to_gcloud(save_path+file_name)

    def dataset_generator_proj2_image_test(self, location, file_name, image_size=(224, 224)):
        satellite = 'VIIRS_Day'
        window_size = 3
        ts_length = 10
        stack_over_location = []
        save_path = 'data_train_proj2/'
        n_channels = 5
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print(location)
        data_path = 'data/' + location + '/' + satellite + '/'
        file_list = glob(data_path + '/*.tif')
        file_list.sort()
        if len(file_list) % ts_length != 0:
            num_sequence = len(file_list) // ts_length + 1
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
        th = [(325, 330), (330, 337), (330, 325), (320, 330), (330, 330), (320, 330), (325, 330), (335, 330),
              (330, 330), (330, 330)]  # swedish_fire
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
                output_array = np.zeros((ts_length, n_channels + 2, image_size[0], image_size[1]))
                if j == num_sequence - 1 and j != 0:
                    file_list_size = len(file_list) % ts_length
                else:
                    file_list_size = ts_length
                for i in range(file_list_size):
                    file = file_list[i + j * 10]
                    array, profile = preprocessing.read_tiff(file)
                    # pick up channels here
                    # array = array[3:, :, :]
                    th_i = th[i]
                    plt.subplot(131)
                    plt.imshow(array[3, :, :])
                    af = np.zeros(array[3, :, :].shape)
                    # Avoid direct modifing original array
                    af[:, :] = np.logical_or(array[3, :, :] > th_i[0], array[4, :, :] > th_i[1])
                    af_img = af
                    af_img[np.logical_not(af_img[:, :])] = np.nan
                    plt.title('Manual label')
                    plt.imshow(af_img, cmap='hsv', interpolation='nearest')
                    plt.subplot(132)
                    plt.title('VIIRS AF product')
                    plt.imshow(array[3, :, :])
                    array[6, :, :][np.where(~np.isnan(array[6, :, :]))] = 1
                    plt.imshow(array[6, :, :], cmap='hsv', interpolation='nearest')
                    plt.subplot(133)
                    plt.title('original image')
                    plt.imshow(array[3, :, :])
                    plt.show()

                    array = self.normalization(array)
                    if k == 0:
                        col_start = int(array.shape[2] * 0)
                    else:
                        col_start = int(array.shape[2] - 224)
                    row_start = int(array.shape[1] * 0.3)
                    array = np.concatenate((array, af[np.newaxis, :, :]))
                    array = array[:, row_start:row_start + image_size[0], col_start:col_start + image_size[1]]
                    output_array[i, :n_channels, :array.shape[1], :array.shape[2]] = array[:n_channels, :, :]
                    output_array[i, n_channels:n_channels + 2, :array.shape[1], :array.shape[2]] = np.nan_to_num(
                        array[n_channels + 1:n_channels + 3, :, :])
            array_stack.append(output_array)
        output_array_stacked = np.stack(array_stack, axis=0)

        np.save(save_path + file_name, output_array_stacked.astype(np.float))
