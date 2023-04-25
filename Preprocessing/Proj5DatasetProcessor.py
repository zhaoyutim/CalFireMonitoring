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

class Proj5DatasetProcessor(PreprocessingService):

    def reconstruct_tif_proj5(self, location, satellite='VIIRS_Day', image_size=(224, 224)):
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

    def dataset_generator_proj5_image_seqtoseq(self, locations, file_name, label_name, save_path, visualize=True, ts_length=10, interval=3, image_size=(224, 224)):
        satellite = 'VIIRS_Day'
        window_size = 1
        stack_over_location = []
        stack_label_over_locations = []
        n_channels = 6
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = 'data/' + location + '/' + satellite + '/'
            file_list = glob(data_path + '/*.tif')
            file_list.sort()
            if len(file_list) == 0:
                continue
            preprocessing = PreprocessingService()
            array, _ = preprocessing.read_tiff(file_list[0])
            array_stack = []
            label_stack = []
            ba_label = np.zeros((image_size[0], image_size[1]))
            af_label = np.zeros((image_size[0], image_size[1]))
            output_array = np.zeros((ts_length, n_channels, image_size[0], image_size[1]), dtype=np.float32)
            output_label = np.zeros((ts_length, 2, image_size[0], image_size[1]), dtype=np.float32)
            file_list_size = len(file_list)
            max_img = np.zeros((n_channels, image_size[0], image_size[1]), dtype=np.float32)
            for i in range(0, file_list_size, interval):
                for j in range(ts_length):
                    if i + j>=file_list_size:
                        break
                    file = file_list[j + i]
                    array, _ = preprocessing.read_tiff(file)
                    if array.shape[0]!=8:
                        print(file, 'band incomplete')
                        continue
                    img = array[:6,:,:]
                    row_start = int(img.shape[1] * 0)
                    col_start = int(img.shape[2] * 0)

                    img = np.nan_to_num(img[:, row_start:row_start + image_size[0], col_start:col_start + image_size[1]])
                    max_img = np.maximum(img, max_img)
                    img = max_img.copy()
                    ba_img = np.concatenate([img[[3],:,:], img[[5],:,:], img[[2],:,:]], axis=0)

                    label = array[7, :, :]
                    af= array[6, :, :]

                    img = self.standardization(img)
                    ba_img = self.normalization(ba_img)
                    label = np.nan_to_num(label[row_start:row_start + image_size[0], col_start:col_start + image_size[1]])
                    af = np.nan_to_num(af[row_start:row_start + image_size[0], col_start:col_start + image_size[1]])
                    ba_label = np.logical_or(label, ba_label)
                    af_label = np.logical_or(af, af_label)
                    ba_img = np.nan_to_num(ba_img[:, row_start:row_start + image_size[0], col_start:col_start + image_size[1]])
                    output_array[j, :6, :, :] = img
                    output_label[j, 0, :, :] = ba_label
                    output_label[j, 1, :, :] = af_label
                    if visualize:
                        plt.figure(figsize=(12, 4), dpi=80)
                        plt.subplot(131)
                        plt.imshow(self.normalization(ba_img).transpose((1,2,0)))
                        plt.imshow(np.where(ba_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        plt.subplot(132)
                        plt.imshow(self.normalization(ba_img).transpose((1,2,0)))
                        plt.imshow(np.where(af_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        plt.subplot(133)
                        plt.imshow(self.normalization(ba_img).transpose((1,2,0)))
                        plt.savefig(save_path+location+'_sequence_'+str(j)+'_time_'+str(i)+'.png')
                        plt.show()
                array_stack.append(output_array)
                label_stack.append(output_label)
            output_array_stacked = np.stack(array_stack, axis=0)
            output_label_stacked = np.stack(label_stack, axis=0)
            stack_over_location.append(output_array_stacked)
            stack_label_over_locations.append(output_label_stacked)
        output_array_stacked_over_location = np.concatenate(stack_over_location, axis=0)
        output_label_stacked_over_location = np.concatenate(stack_label_over_locations, axis=0)
        del stack_over_location
        del stack_label_over_locations
        print(output_array_stacked_over_location.shape)
        print(output_label_stacked_over_location.shape)

        # np.save(save_path + file_name, output_array_stacked_over_location.astype(np.float32))
        # np.save(save_path + label_name, output_label_stacked_over_location.astype(np.float32))

    def dataset_generator_proj5_image_seqtoone(self, locations, file_name, label_name, visualize=True, ts_length=10, interval=3, image_size=(224, 224)):
        satellite = 'VIIRS_Day'
        window_size = 1
        stack_over_location = []
        stack_label_over_locations = []
        save_path = 'data_train_proj5/'
        n_channels = 6
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = 'data/' + location + '/' + satellite + '/'
            file_list = glob(data_path + '/*.tif')
            file_list.sort()
            preprocessing = PreprocessingService()
            array, _ = preprocessing.read_tiff(file_list[0])
            array_stack = []
            label_stack = []
            ba_label = np.zeros((image_size[0], image_size[1]))
            af_label = np.zeros((image_size[0], image_size[1]))
            output_array = np.zeros((ts_length, n_channels, image_size[0], image_size[1]), dtype=np.float32)
            output_label = np.zeros((2, image_size[0], image_size[1]), dtype=np.float32)
            file_list_size = len(file_list)
            for i in range(0, file_list_size, interval):
                for j in range(ts_length):
                    if i + j>=file_list_size:
                        break
                    file = file_list[j + i]
                    array, _ = preprocessing.read_tiff(file)
                    if array.shape[0]!=9:
                        print(file, 'band incomplete')
                        continue
                    img = np.concatenate([array[:5,:,:], array[[6],:,:]], axis=0)
                    ba_img = np.concatenate([array[[6],:,:], array[[1],:,:], array[[0],:,:]], axis=0)
                    label = array[8, :, :]
                    af= array[7,:,:]

                    img = self.standardization(img)
                    row_start = int(img.shape[1] * 0)
                    col_start = int(img.shape[2] * 0)

                    img = img[:, row_start:row_start + image_size[0], col_start:col_start + image_size[1]]
                    label = np.nan_to_num(label[row_start:row_start + image_size[0], col_start:col_start + image_size[1]])
                    af = np.nan_to_num(af[row_start:row_start + image_size[0], col_start:col_start + image_size[1]])
                    ba_label = np.logical_or(label, ba_label)
                    af_label = np.logical_or(af, af_label)
                    af = af[row_start:row_start + image_size[0], col_start:col_start + image_size[1]]
                    ba_img = np.nan_to_num(ba_img[:, row_start:row_start + image_size[0], col_start:col_start + image_size[1]])
                    output_array[j, :6, :, :] = img[:, :, :]
                    if visualize:
                        plt.figure(figsize=(12, 4), dpi=80)
                        plt.subplot(131)
                        plt.imshow(self.normalization(ba_img).transpose((1,2,0)))
                        plt.imshow(np.where(ba_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        plt.subplot(132)
                        plt.imshow(self.normalization(ba_img).transpose((1,2,0)))
                        plt.imshow(np.where(af_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        plt.subplot(133)
                        plt.imshow(self.normalization(ba_img).transpose((1,2,0)))
                        plt.savefig('img_train_proj5/'+location+'_sequence_'+str(j)+'_time_'+str(i)+'.png')
                        plt.show()
                output_label[0, :, :] = ba_label
                output_label[1, :, :] = af_label
                array_stack.append(output_array)
                label_stack.append(output_label)
            output_array_stacked = np.stack(array_stack, axis=0)
            output_label_stacked = np.stack(label_stack, axis=0)
            stack_over_location.append(output_array_stacked)
            stack_label_over_locations.append(output_label_stacked)
        output_array_stacked_over_location = np.concatenate(stack_over_location, axis=0)
        output_label_stacked_over_location = np.concatenate(stack_label_over_locations, axis=0)
        del stack_over_location
        del stack_label_over_locations
        print(output_array_stacked_over_location.shape)
        print(output_label_stacked_over_location.shape)

        np.save(save_path + file_name, output_array_stacked_over_location.astype(np.float32))
        np.save(save_path + label_name, output_label_stacked_over_location.astype(np.float32))
