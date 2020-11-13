import datetime
import os
import numpy.ma as npm
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import cv2
import numpy as np
import rasterio
import yaml
from numpy.ma.core import MaskedConstant

from Preprocessing.PreprocessingService import PreprocessingService

with open("DataPreparation/config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class EvaluationService:
    # def __init__(self):

    def postprocessing(self, array):
        ret, lstm_conf = cv2.threshold(array,2,15,cv2.THRESH_BINARY)
        return lstm_conf

    def reconstruct_trial3(self, location):
        preprocessing = PreprocessingService()
        output = np.load('data/evaluate/'+location + '/output_' + location + '.npy')
        save_path = 'data/evaluate/'+location
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        data_path = 'data/label/'+location + ' label'
        data_path = Path(data_path)
        data_file_list = glob(str(data_path / "*.tif"))
        data_file_list.sort()
        index = 0
        for file in data_file_list:
            filename = file.replace(str(data_path) + '/','').replace('FIRMS', 'Recon')
            output_perday = output[:,:,index]
            index += 1
            firms_arr, firms_profile = preprocessing.read_tiff(file)
            firms_profile.update(count=1, dtype='float32')
            x_size = firms_arr.shape[1]
            y_size = firms_arr.shape[2]
            output_recons = np.zeros((x_size, y_size))
            output_recons[x_size - 3:1:-1, 2:y_size - 2] = output_perday
            preprocessing.write_tiff(save_path + filename, output_recons[np.newaxis, :, :], firms_profile)

    def reconstruct_trial5(self, location):
        preprocessing = PreprocessingService()

        output = np.load('data/evaluate/'+location + '/output_' + location + '.npy')
        save_path = 'data/evaluate/'+location +'/recon/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        data_path = 'data/label/' + location + ' label'
        data_path = Path(data_path)
        data_file_list = glob(str(data_path / "*.tif"))
        data_file_list.sort()
        file = data_file_list[0]
        firms_arr, firms_profile = preprocessing.read_tiff(file)
        firms_profile.update(count=1, dtype='float32')
        for index in range(output.shape[2]):
            output_perday = output[:,:,index]

            x_size = firms_arr.shape[1]
            y_size = firms_arr.shape[2]
            output_recons = np.zeros((x_size, y_size))
            output_recons[x_size - 6:4:-1, 5:y_size - 5] = output_perday
            preprocessing.write_tiff(save_path + location+'_recon_' + "{:02d}:00".format(index) + '.tif', output_recons[np.newaxis, :, :], firms_profile)

    def reference_trial3(self, location):
        data_path = 'data/evaluate/' + location + '/reference'
        data_path = Path(data_path)
        save_path = 'data/evaluate/' + location + '/reference_dataset'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        data_file_list = glob(str(data_path / "*.tif"))
        dataset = []
        data_file_list.sort()
        for file in data_file_list:
            with rasterio.open(file, 'r') as reader:
                goes_arr = reader.read()  # read all raster values
            if np.isnan(np.nanmean(goes_arr[0, :, :])):
                continue
            goes_composition = np.nan_to_num(goes_arr[0, :, :], nan=np.nanmean(goes_arr[0, :, :]))
            goes_after_processing = goes_composition[:, :]
            x_size = 314
            y_size = 256
            goes_resized = cv2.resize(goes_after_processing, (y_size, x_size), interpolation=cv2.INTER_CUBIC)

            vectorized_feature = np.zeros(((x_size - 4) * (y_size - 4), 25))
            for i in range(2, x_size - 2):
                for j in range(2, y_size - 2):
                    index = (i - 2) * (y_size - 4) + j - 2
                    vectorized_feature[index, :] = goes_resized[i - 2:i + 3, j - 2:j + 3].flatten()
            dataset.append(vectorized_feature)
        dataset_output = np.stack(dataset, axis=0)
        np.save(save_path + '/' + location + ' dataset_ref.npy', dataset_output)

    def reference_trial5(self, location):
        data_path = 'data/evaluate/' + location + '/reference'
        data_path = Path(data_path)
        save_path = 'data/evaluate/' + location + '/reference_dataset'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        data_file_list = glob(str(data_path / "*.tif"))
        dataset = []
        maxed_dataset = []
        data_file_list.sort()
        label_path = 'data/label/'+location + ' label'
        label_path = Path(label_path)
        label_file_list = glob(str(label_path / "*.tif"))
        if len(label_file_list) == 0:
            print('Please generate label first')
            return
        with rasterio.open(label_file_list[0], 'r') as reader:
            label_arr = reader.read()  # read all raster values
        x_size = label_arr.shape[1]
        y_size = label_arr.shape[2]
        for file in data_file_list:
            with rasterio.open(file, 'r') as reader:
                goes_arr = reader.read()  # read all raster values
            nan_value = 0
            if np.isnan(nan_value):
                continue
            goes_composition = np.nan_to_num(goes_arr[0, :, :], nan=nan_value)
            # plt.imshow(goes_composition)
            # plt.show()
            goes_after_processing = goes_composition[:, :]
            goes_resized = cv2.resize(goes_after_processing, (y_size, x_size), interpolation=cv2.INTER_LINEAR)

            vectorized_feature = np.zeros(((x_size - 10) * (y_size - 10), 121))
            for i in range(5, x_size - 5):
                for j in range(5, y_size - 5):
                    index = (i - 5) * (y_size - 10) + j - 5
                    vectorized_feature[index, :] = goes_resized[i - 5:i + 6, j - 5:j + 6].flatten()

            dataset.append(vectorized_feature)
        # for i in range(2, len(dataset)):
        #     maxed_dataset.append(np.amax(np.stack(dataset[i-2:i+1], axis=0), axis=0))

        dataset_output = np.stack(dataset, axis=0)
        np.save(save_path + '/' + location + ' dataset_ref_trial5'+str(x_size)+'*'+str(y_size)+'.npy', dataset_output.astype(np.float32))

    def evaluate_mIoU(self, location, reference_satellite):
        preprocessing = PreprocessingService()
        pre_fire_path = 'data/' + location + reference_satellite + '/' + location + reference_satellite + '_Cal_fire_' + location + reference_satellite + '-prefire.tif'
        s2_prefire, s2_prefire_profile = preprocessing.read_tiff(pre_fire_path)
        path = 'data/' + location + reference_satellite + '/'
        data_file_list = glob(str(Path(path) / "*.tif"))
        data_file_list.sort()
        eval_path = 'data/label/' + location + ' label/'
        eval_file_list = glob(str(Path(eval_path) / "*.tif"))
        eval_file_list.sort()
        independent_ec = 0
        independent_eo = 0
        tp = 0
        tn = 0
        for file in data_file_list:
            if 'prefire' in file:
                continue
            s2_afterfire, _ = preprocessing.read_tiff(file)
            d_index = np.nan_to_num(s2_afterfire[3]) - np.nan_to_num(s2_prefire[3])
            # mask = d_index>-0.15
            # img = np.ones((d_index.shape))
            # img = npm.array(img, mask = mask)*255
            ret, img = cv2.threshold(d_index, -2, -0.15, cv2.THRESH_TOZERO)
            img = img < -0.15
            s2_date = datetime.datetime.strptime(file.replace('data/'+location+'Sentinel2/'+location+'Sentinel2_Cal_fire_'+location+'Sentinel2-', '').replace('.tif', ''), '%Y-%m-%d')
            acc_img = []
            for eval_file in eval_file_list:
                label_img, _ = preprocessing.read_tiff(eval_file)
                label_img = np.flip(label_img[0,:,:], axis=0)
                eval_date = datetime.datetime.strptime(eval_file.replace('data/label/'+location+' label/Cal_fire_'+location+'FIRMS-', '').replace('.tif', '')[:-17], '%Y-%m-%d')
                if s2_date < eval_date:
                    break
                acc_img.append(label_img)
            eval_img = np.stack(acc_img, axis=2).sum(axis=2)
            # eval_img = np.load('data/evaluate/creek_fire/output_creek_fire_acc305_248.npy')
            plt.imshow(eval_img)
            plt.show()
            plt.imshow(img)
            plt.show()
            mapping_factor = (img.shape[0]/eval_img.shape[0], img.shape[1]/eval_img.shape[1])
            for i in range(75,225):
                for j in range(50,175):
                    s2_patch = img[round(i*mapping_factor[0]):round((i+1)*mapping_factor[0]), round(j*mapping_factor[1]):round((j+1)*mapping_factor[1])]
                    if (s2_patch).max()==False and eval_img[i, j] != 0:
                        independent_ec += 1
                    elif (s2_patch).max()!=False and eval_img[i, j] == 0:
                        independent_eo += 1
                    elif (s2_patch).max()!=False and eval_img[i, j] != 0:
                        tp += 1
                    elif (s2_patch).max()==False and eval_img[i, j] == 0:
                        tn += 1

        print('Error of Commission:{}'.format(independent_ec/(tp+independent_ec)))
        print('Error of Ommission:{}'.format(independent_eo/(tp+independent_eo)))
        print(tp, tn)
