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

    def dataset_generator_proj5_images(self, mode, locations, file_name, label_name, save_path, rs_idx=0,
                                               cs_idx=0,
                                               visualize=True, image_size=(224, 224)):
        satellite_day = 'VIIRS_Day'
        satellite_night = 'VIIRS_Night'
        window_size = 1
        stack_over_location = []
        stack_label_over_locations = []
        n_channels = 8
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = 'data/' + location + '/' + satellite_day + '/'
            file_list = glob(data_path + '/*.tif')
            file_list.sort()
            if len(file_list) == 0:
                print('empty file list')
                continue
            preprocessing = PreprocessingService()
            array_day, _ = preprocessing.read_tiff(file_list[0])
            array_stack = []
            label_stack = []

            if mode == 'train' or mode == 'val':
                output_shape_x = 256
                output_shape_y = 256
                offset = 128
            else:
                output_shape_x = array_day.shape[1] // 2
                output_shape_y = array_day.shape[2] // 2
                offset = 256

            original_shape_x = array_day.shape[1]
            original_shape_y = array_day.shape[2]

            ba_label = np.zeros((output_shape_x, output_shape_y))
            af_label = np.zeros((output_shape_x, output_shape_y))
            file_list_size = len(file_list)
            max_img = np.zeros((n_channels, output_shape_x, output_shape_y), dtype=np.float32)
            acc_label = np.zeros((3, output_shape_x, output_shape_y), dtype=np.float32)
            for i in range(file_list_size):
                output_array = np.zeros((n_channels, output_shape_x, output_shape_y), dtype=np.float32)
                output_label = np.zeros((3, output_shape_x, output_shape_y), dtype=np.float32)
                file = file_list[i]
                array_day, _ = preprocessing.read_tiff(file)
                if os.path.exists(file.replace('VIIRS_Day', 'VIIRS_Night')):
                    array_night, _ = preprocessing.read_tiff(file.replace('VIIRS_Day', 'VIIRS_Night'))
                    if array_night.shape[0] == 5:
                        print('Day_night miss align')
                        array_night = array_night[3:, :, :]
                    if array_night.shape[0] < 2:
                        print(file.replace('VIIRS_Day', 'VIIRS_Night'), 'band incomplete')
                        continue
                    if array_night.shape[1] != array_day.shape[1] or array_night.shape[2] != array_day.shape[2]:
                        print('Day Night not match')
                        print(file)
                else:
                    array_night = np.zeros((2, original_shape_x, original_shape_y))
                print(file)

                img = np.concatenate((array_day[:6, offset:output_shape_x + offset, offset:output_shape_y + offset],
                                      array_night[:, offset:output_shape_x + offset,
                                      offset:output_shape_y + offset]), axis=0)

                img = np.nan_to_num(img[:, :output_shape_x, :output_shape_y])
                max_img = np.maximum(img, max_img)
                img = max_img.copy()
                ba_img = np.concatenate([img[[2], :, :], img[[1], :, :], img[[0], :, :]], axis=0)

                label = array_day[7, :, :]
                af = array_day[6, :, :]

                # img = self.standardization(img)
                ba_img = self.normalization(ba_img)
                label = np.nan_to_num(label[offset:output_shape_x + offset, offset:output_shape_y + offset])
                af = np.nan_to_num(af[offset:output_shape_x + offset, offset:output_shape_y + offset])
                ba_label = np.logical_or(label, ba_label)
                af_label = np.logical_or(af, af_label)
                output_array[:n_channels, :, :] = img
                output_label[0, :, :] = ba_label
                output_label[1, :, :] = af_label
                output_label[2, :, :] = np.logical_or(af_label, ba_label)
                acc_label = np.logical_or(acc_label, output_label)
                if visualize:
                    plt.figure(figsize=(12, 4), dpi=80)
                    plt.subplot(131)
                    plt.imshow(ba_img.transpose((1, 2, 0)))
                    plt.imshow(np.where(ba_label == 0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                    plt.subplot(132)
                    plt.imshow(ba_img.transpose((1, 2, 0)))
                    plt.imshow(np.where(af_label == 0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                    plt.subplot(133)
                    plt.imshow(ba_img.transpose((1, 2, 0)))
                    plt.savefig(save_path + location +'_img_' + str(i)  + '.png')
                    plt.show()
                array_stack.append(output_array)
                label_stack.append(acc_label)
            if len(array_stack) == 0:
                print('No enough TS')
                continue
            output_array_stacked = np.stack(array_stack, axis=0)
            output_label_stacked = np.stack(label_stack, axis=0)
            stack_over_location.append(output_array_stacked)
            stack_label_over_locations.append(output_label_stacked)
        if len(stack_over_location) == 0:
            print('No array to stack')
            return
        dataset_stacked_over_locations = np.concatenate(stack_over_location, axis=0)
        labels_stacked_over_locations = np.concatenate(stack_label_over_locations, axis=0)
        del stack_over_location
        del stack_label_over_locations
        for i in range(8):
            print(np.nanmean(dataset_stacked_over_locations[:, i, :, :]))
            print(np.nanstd(dataset_stacked_over_locations[:, i, :, :]))
        np.save(save_path + file_name, dataset_stacked_over_locations.astype(np.float32))
        np.save(save_path + label_name, labels_stacked_over_locations.astype(np.float32))

    def dataset_generator_proj5_image_seqtoseq(self, mode, locations, file_name, label_name, save_path, rs_idx=0, cs_idx=0,
                                               visualize=True, ts_length=10, interval=3, image_size=(224, 224)):
        satellite_day = 'VIIRS_Day'
        satellite_night = 'VIIRS_Night'
        window_size = 1
        stack_over_location = []
        stack_label_over_locations = []
        n_channels = 8
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = 'data/' + location + '/' + satellite_day + '/'
            file_list = glob(data_path + '/*.tif')
            file_list.sort()
            if len(file_list) == 0:
                print('empty file list')
                continue
            preprocessing = PreprocessingService()
            array_day, _ = preprocessing.read_tiff(file_list[0])
            array_stack = []
            label_stack = []

            if mode == 'train' or mode == 'val':
                output_shape_x = 256
                output_shape_y = 256
                offset=128
            else:
                output_shape_x = array_day.shape[1]//2
                output_shape_y = array_day.shape[2]//2
                offset=256
            
            original_shape_x = array_day.shape[1]
            original_shape_y = array_day.shape[2]

            ba_label = np.zeros((output_shape_x, output_shape_y))
            af_acc_label = np.zeros((output_shape_x, output_shape_y))
            file_list_size = len(file_list)
            max_img = np.zeros((n_channels, output_shape_x, output_shape_y), dtype=np.float32)
            for i in range(0, file_list_size, interval):
                if i + ts_length >= file_list_size:
                    print('drop the tail')
                    break
                output_array = np.zeros((ts_length, n_channels, output_shape_x, output_shape_y), dtype=np.float32)
                output_label = np.zeros((ts_length, 3, output_shape_x, output_shape_y), dtype=np.float32)
                for j in range(ts_length):
                    file = file_list[j + i]
                    array_day, _ = preprocessing.read_tiff(file)
                    if os.path.exists(file.replace('VIIRS_Day', 'VIIRS_Night')):
                        array_night, _ = preprocessing.read_tiff(file.replace('VIIRS_Day', 'VIIRS_Night'))
                        if array_night.shape[0] == 5:
                            print('Day_night miss align')
                            array_night = array_night[3:, :, :]
                        if array_night.shape[0] < 2:
                            print(file.replace('VIIRS_Day', 'VIIRS_Night'), 'band incomplete')
                            continue
                        if array_night.shape[1] != array_day.shape[1] or array_night.shape[2] != array_day.shape[2]:
                            print('Day Night not match')
                            print(file)
                    else:
                        array_night = np.zeros((2, original_shape_x, original_shape_y))
                    print(file)
                    
                    img = np.concatenate((array_day[:6, offset:output_shape_x+offset, offset:output_shape_y+offset], array_night[:, offset:output_shape_x+offset, offset:output_shape_y+offset]), axis=0)

                    img = np.nan_to_num(img[:,:output_shape_x, :output_shape_y])
                    max_img = np.maximum(img, max_img)
                    # img = max_img.copy()
                    # todo change this?
                    img = np.concatenate((img[:3,...],max_img[3:5,...],img[[5],...],max_img[6:8,...]))
                    # ba_img = np.concatenate([img[[2],:,:], img[[1],:,:], img[[0],:,:]], axis=0)
                    ba_img = np.concatenate(([img[[5],:,:], img[[1],:,:], img[[0],:,:]]))
                    label = array_day[7, :, :]
                    af= array_day[6, :, :]

                    # img = self.standardization(img)
                    ba_img = self.normalization(ba_img)
                    label = np.nan_to_num(label[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    af = np.nan_to_num(af[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    ba_label = np.logical_or(label, ba_label)
                    af_acc_label = np.logical_or(af, af_acc_label)
                    output_array[j, :n_channels, :, :] = img
                    output_label[j, 0, :, :] = np.logical_or(af_acc_label, ba_label)
                    output_label[j, 1, :, :] = af_acc_label
                    output_label[j, 2, :, :] = af
                    if visualize:
                        # plt.figure(figsize=(12, 4), dpi=80)
                        # plt.subplot(131)
                        # plt.imshow(ba_img.transpose((1,2,0)))
                        # plt.imshow(np.where(ba_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        # plt.subplot(132)
                        # plt.imshow(ba_img.transpose((1,2,0)))
                        # plt.imshow(np.where(af_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        # plt.subplot(133)
                        # plt.imshow(ba_img.transpose((1,2,0)))
                        # plt.savefig(save_path+location+'_sequence_'+str(i)+'_time_'+str(j)+'.png')
                        # plt.show()
                        plt.figure(figsize=(12, 4), dpi=80)
                        plt.subplot(131)
                        plt.imshow(ba_img.transpose((1,2,0)))
                        plt.imshow(np.where(af_acc_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        plt.axis('off')
                        plt.title('AF ACC')
                        plt.subplot(132)
                        plt.imshow(ba_img.transpose((1,2,0)))
                        plt.imshow(np.where(af==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        plt.axis('off')
                        plt.title('AF')
                        plt.subplot(133)
                        plt.imshow(ba_img.transpose((1,2,0)))
                        plt.imshow(np.where(ba_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        plt.axis('off')
                        plt.title('BA')
                        plt.savefig(save_path+location+'_sequence_'+str(i)+'_time_'+str(j)+'_ts_'+str(ts_length)+'_comb.png', bbox_inches='tight')
                array_stack.append(output_array)
                label_stack.append(output_label)
            if len(array_stack)==0:
                print('No enough TS')
                continue
            output_array_stacked = np.stack(array_stack, axis=0)
            output_label_stacked = np.stack(label_stack, axis=0)
            stack_over_location.append(output_array_stacked)
            stack_label_over_locations.append(output_label_stacked)
        dataset_stacked_over_locations = np.concatenate(stack_over_location, axis=0).transpose((0,2,1,3,4))
        labels_stacked_over_locations = np.concatenate(stack_label_over_locations, axis=0).transpose((0,2,1,3,4))
        del stack_over_location
        del stack_label_over_locations
        for i in range(8):
            print(np.nanmean(dataset_stacked_over_locations[:,i,:,:,:]))
            print(np.nanstd(dataset_stacked_over_locations[:,i,:,:,:]))
        np.save(save_path + file_name, dataset_stacked_over_locations.astype(np.float32))
        np.save(save_path + label_name, labels_stacked_over_locations.astype(np.float32))


    def dataset_generator_proj5_image_seqtoseq_eva(self, mode, locations, file_name, label_name, save_path, rs_idx=0, cs_idx=0,
                                               visualize=True, ts_length=10, interval=3, image_size=(224, 224)):
        satellite_day = 'VIIRS_Day'
        satellite_night = 'VIIRS_Night'
        window_size = 1
        stack_over_location = []
        stack_label_over_locations = []
        n_channels = 8
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for location in locations:
            print(location)
            data_path = 'data/' + location + '/' + satellite_day + '/'
            file_list = glob(data_path + '/*.tif')
            file_list.sort()
            if len(file_list) == 0:
                print('empty file list')
                continue
            preprocessing = PreprocessingService()
            array_day, _ = preprocessing.read_tiff(file_list[0])
            array_stack = []
            label_stack = []

            if mode == 'train' or mode == 'val':
                output_shape_x = 256
                output_shape_y = 256
                offset=128
            else:
                output_shape_x = array_day.shape[1]
                output_shape_y = array_day.shape[2]
                offset=0
            
            original_shape_x = array_day.shape[1]
            original_shape_y = array_day.shape[2]

            ba_label = np.zeros((output_shape_x, output_shape_y))
            af_acc_label = np.zeros((output_shape_x, output_shape_y))
            file_list_size = len(file_list)
            max_img = np.zeros((n_channels, output_shape_x, output_shape_y), dtype=np.float32)
            for i in range(0, file_list_size, interval):
                if i + ts_length > file_list_size:
                    print('drop the tail')
                    break
                output_array = np.zeros((ts_length, n_channels, output_shape_x, output_shape_y), dtype=np.float32)
                output_label = np.zeros((ts_length, 3, output_shape_x, output_shape_y), dtype=np.float32)
                for j in range(ts_length):
                    file = file_list[j + i]
                    array_day, _ = preprocessing.read_tiff(file)
                    if os.path.exists(file.replace('VIIRS_Day', 'VIIRS_Night')):
                        array_night, _ = preprocessing.read_tiff(file.replace('VIIRS_Day', 'VIIRS_Night'))
                        if array_night.shape[0] == 5:
                            print('Day_night miss align')
                            array_night = array_night[3:, :, :]
                        if array_night.shape[0] < 2:
                            print(file.replace('VIIRS_Day', 'VIIRS_Night'), 'band incomplete')
                            continue
                        if array_night.shape[1] != array_day.shape[1] or array_night.shape[2] != array_day.shape[2]:
                            print('Day Night not match')
                            print(file)
                    else:
                        array_night = np.zeros((2, original_shape_x, original_shape_y))
                    print(file)
                    
                    img = np.concatenate((array_day[:6, offset:output_shape_x+offset, offset:output_shape_y+offset], array_night[:, offset:output_shape_x+offset, offset:output_shape_y+offset]), axis=0)
                    thermal_day = img[[3],...]
                    thermal_night = img[[6],...]
                    img = np.nan_to_num(img[:,:output_shape_x, :output_shape_y])
                    max_img = np.maximum(img, max_img)
                    print(max_img.shape, img.shape)
                    img = np.concatenate((img[:3,...],max_img[3:5,...],img[[5],...],max_img[6:8,...]))
                    # img = max_img.copy()
                    ba_img = np.concatenate(([img[[5],:,:], img[[1],:,:], img[[0],:,:]]))
                    if array_day.shape[0]==8:
                        label = np.nan_to_num(array_day[7, :, :])
                    else:
                        label = np.zeros((output_shape_x, output_shape_y))
                    af= array_day[6, :, :]

                    # img = self.standardization(img)
                    # ba_img = self.normalization(ba_img)
                    ba_img = ba_img/40
                    label = np.nan_to_num(label[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    af = np.nan_to_num(af[offset:output_shape_x+offset, offset:output_shape_y+offset])
                    ba_label = np.logical_or(label, ba_label)
                    af_acc_label = np.logical_or(af, af_acc_label)
                    output_array[j, :n_channels, :, :] = img
                    output_label[j, 0, :, :] = ba_label
                    output_label[j, 1, :, :] = af_acc_label
                    output_label[j, 2, :, :] = af
                    if visualize:
                        
                        # plt.imshow(ba_img.transpose((1,2,0)))
                        plt.imshow(np.where(af_acc_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        plt.axis('off')
                        plt.savefig(save_path+location+'_sequence_'+str(i)+'_time_'+str(j)+'_ts_'+str(ts_length)+'overlay.png', bbox_inches='tight')
                        plt.show()
                        # plt.imshow(ba_img.transpose((1,2,0)))
                        # plt.axis('off')
                        # plt.savefig(save_path+location+'_sequence_'+str(i)+'_time_'+str(j)+'_ts_'+str(ts_length)+'.png', bbox_inches='tight')
                        # plt.show()
                        # plt.imshow(self.normalization(thermal_day).transpose((1,2,0)))
                        # plt.axis('off')
                        # plt.savefig(save_path+location+'_sequence_'+str(i)+'_time_'+str(j)+'_ts_'+str(ts_length)+'_thermal_day.png', bbox_inches='tight')
                        # plt.show()

                        # plt.imshow(self.normalization(thermal_night).transpose((1,2,0)))
                        # plt.axis('off')
                        # plt.savefig(save_path+location+'_sequence_'+str(i)+'_time_'+str(j)+'_ts_'+str(ts_length)+'_thermal_night.png', bbox_inches='tight')
                        # plt.show()

                        # plt.figure(figsize=(12, 4), dpi=80)
                        # plt.subplot(131)
                        # plt.imshow(ba_img.transpose((1,2,0)))
                        # plt.imshow(np.where(af_acc_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        # plt.axis('off')
                        # plt.title('AF ACC')
                        # plt.subplot(132)
                        # plt.imshow(ba_img.transpose((1,2,0)))
                        # plt.imshow(np.where(af==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        # plt.axis('off')
                        # plt.title('AF')
                        # plt.subplot(133)
                        # plt.imshow(ba_img.transpose((1,2,0)))
                        # plt.imshow(np.where(ba_label==0, np.nan, 1), cmap='hsv', interpolation='nearest', alpha=1)
                        # plt.axis('off')
                        # plt.title('BA')
                        # plt.savefig(save_path+location+'_sequence_'+str(i)+'_time_'+str(j)+'_ts_'+str(ts_length)+'_comb.png', bbox_inches='tight')

                array_stack.append(output_array)
                label_stack.append(output_label)
            if len(array_stack)==0:
                print('No enough TS')
                continue
            output_array_stacked = np.stack(array_stack, axis=0)
            output_label_stacked = np.stack(label_stack, axis=0)
            stack_over_location.append(output_array_stacked)
            stack_label_over_locations.append(output_label_stacked)
        dataset_stacked_over_locations = np.concatenate(stack_over_location, axis=0).transpose((0,2,1,3,4))
        labels_stacked_over_locations = np.concatenate(stack_label_over_locations, axis=0).transpose((0,2,1,3,4))
        del stack_over_location
        del stack_label_over_locations
        for i in range(8):
            print(np.nanmean(dataset_stacked_over_locations[:,i,:,:,:]))
            print(np.nanstd(dataset_stacked_over_locations[:,i,:,:,:]))
        np.save(save_path + file_name, dataset_stacked_over_locations.astype(np.float32))
        np.save(save_path + label_name, labels_stacked_over_locations.astype(np.float32))