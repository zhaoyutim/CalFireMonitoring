import datetime
import fnmatch
import os
from glob import glob
from pathlib import Path

import cv2
import imageio
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as npm
import rasterio
import tensorflow as tf
import yaml
from imageio import imsave

from Preprocessing.PreprocessingService import PreprocessingService

with open("config/configuration.yml", "r", encoding="utf8") as f:
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
        output = output.transpose((1,2,0))
        start_date = config.get(location).get('start')
        for index in range(output.shape[2]):
            output_perday = output[:,:,index]

            x_size = firms_arr.shape[1]
            y_size = firms_arr.shape[2]
            output_recons = np.zeros((x_size, y_size))
            output_recons[x_size - 6:4:-1, 5:y_size - 5] = output_perday
            current_date = start_date + datetime.timedelta(index // 24)
            current_time = index % 24
            preprocessing.write_tiff(save_path + location + '_output_' + str(current_date) + 'T' + str(current_time) + '.tif', output_recons[np.newaxis, :, :], firms_profile)
        imageio.mimsave(save_path + location + '.gif', output.transpose((2,0,1)), format='GIF', fps=10)

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

    def reference_trial5(self, location, custom_size):
        data_path = 'data/evaluate/' + location + '/reference'
        data_path = Path(data_path)
        save_path = 'data/evaluate/' + location + '/reference_dataset'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        data_file_list = glob(str(data_path / "*.tif"))
        dataset = []
        maxed_dataset = []
        data_file_list.sort()
        if not custom_size:
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
        else:
            x_size = 300
            y_size = 250
        for file in data_file_list:
            with rasterio.open(file, 'r') as reader:
                goes_arr = reader.read()  # read all raster values
            nan_value = 0
            if np.isnan(nan_value):
                continue
            goes_composition = np.nan_to_num(goes_arr[0, :, :], nan=nan_value)
            plt.imshow(goes_composition)
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

    def evaluate_mIoU(self, location, reference_satellite, satellites, s2_date):

        preprocessing = PreprocessingService()
        pre_fire_path = 'data/' + location + reference_satellite + '/' + location + reference_satellite + '_Cal_fire_' + location + reference_satellite + '-prefire.tif'
        # s2_prefire, s2_prefire_profile = preprocessing.read_tiff(pre_fire_path)
        path = 'data/' + location + reference_satellite + '/'
        data_file_list = glob(str(Path(path) / "*.tif"))
        data_file_list.sort()

        file = 'data/'+location+'Sentinel2/'+location+'Sentinel2_Cal_fire_'+location+'Sentinel2-' + s2_date + '.tif'
        s2_afterfire, _ = preprocessing.read_tiff(file)
        img = s2_afterfire[0, :, :]
        img = np.nan_to_num(img)
        # bbox_x=(0.47, 0.60)
        # bbox_y=(0.45, 0.60)
        bbox_x=(0.4,0.7)
        bbox_y = (0.1, 0.6)

        # bbox_x=(0.2,0.8)
        # bbox_y = (0.2, 0.9)

        img[:round(bbox_x[0]*img.shape[0]), :] = 0
        img[:, :round(bbox_y[0]*img.shape[1])] = 0
        img[round(bbox_x[1] * img.shape[0]):, :] = 0
        img[:,round(bbox_y[1] * img.shape[1]):] = 0
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
        plt.imshow(img, cmap='Greys')
        plt.axis('off')
        plt.show()
        save_path = 'results/'+location
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        imageio.imsave(save_path + 'S2.png', img)
        s2_date = datetime.datetime.strptime(
            file.replace('data/' + location + 'Sentinel2/' + location + 'Sentinel2_Cal_fire_' + location + 'Sentinel2-',
                         '').replace('.tif', ''), '%Y-%m-%d')

        for satellite in satellites:
            if satellite == 'FIRMS':
                eval_path = 'data/label/' + location + ' label/'
                eval_file_list = glob(str(Path(eval_path) / "*.tif"))
                eval_file_list.sort()
            else:
                eval_path = 'data/' + location + satellite + '/'
                eval_file_list = glob(str(Path(eval_path) / "*.tif"))
                eval_file_list.sort()
            independent_ec = 0
            independent_eo = 0
            tp = 0
            tn = 0
            if satellite != 'GOES':
                acc_img = []
                for eval_file in eval_file_list:
                    label_img, _ = preprocessing.read_tiff(eval_file)
                    label_img = np.flip(label_img[0,:,:], axis=0)
                    if satellite == 'FIRMS':
                        eval_date = datetime.datetime.strptime(eval_file.replace('data/label/'+location+' label/Cal_fire_'+location+'FIRMS-', '').replace('.tif', '')[:-4], '%Y-%m-%d')

                    else:
                        eval_date = datetime.datetime.strptime(
                            eval_file.replace('data/' + location + satellite+'/'+location+satellite+'_Cal_fire_' + location + satellite + '-',
                                              '').replace('.tif', ''), '%Y-%m-%d')
                    if s2_date < eval_date:
                        break
                    acc_img.append(label_img)
                eval_img = np.nan_to_num(np.stack(acc_img, axis=2).sum(axis=2))
                if eval_img.shape != img.shape:
                    eval_img = cv2.resize(eval_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                eval_img = np.load('data/evaluate/' + location + '/output_' + location + '_278_214rf.npy')

                eval_img_resize = np.zeros((eval_img.shape[0]+10, eval_img.shape[1]+10))
                eval_img_resize[5:-5, 5:-5] = eval_img
                eval_img = eval_img_resize
            ret, eval_img = cv2.threshold(eval_img, 0, 1, cv2.THRESH_BINARY)
            imsa
            plt.imshow(eval_img, cmap='Greys')
            plt.axis('off')
            plt.show()
            imageio.imsave(save_path + satellite + '.png', eval_img)
            for i in range(eval_img.shape[0]):
                for j in range(eval_img.shape[1]):
                    s2_patch = img[i,j]
                    if s2_patch == 0 and eval_img[i, j] == 1:
                        independent_ec += 1
                    elif s2_patch == 1 and eval_img[i, j] == 0:
                        independent_eo += 1
                    elif s2_patch == 1 and eval_img[i, j] == 1:
                        tp += 1
                    elif s2_patch == 0 and eval_img[i, j] == 0:
                        tn += 1

            print('Error of Commission:{}'.format(independent_ec/(tp+independent_ec)))
            print('Error of Ommission:{}'.format(independent_eo/(tp+independent_eo)))
            print('F1 score:{}'.format(tp/(tp+0.5*(independent_ec+independent_eo))))
            print('mIoU score:{}'.format(self.mIou(img, eval_img)))
            print(tp, tn)

    def mIou(self, input1, input2):
        component1 = input1
        component2 = input2

        overlap = component1 * component2  # Logical AND
        union = component1 + component2  # Logical OR
        IOU = np.count_nonzero(overlap) / float(np.count_nonzero(union))
        print(overlap.sum(), float(union.sum()))
        return IOU

    def evaluate_and_generate_images(self, location):
        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

        lstm_model = tf.keras.models.load_model('model/lstm_model5', custom_objects={'f1_m': f1_m})
        path = '/Users/zhaoyu/PycharmProjects/CalFireMonitoring/data/evaluate/'+location+'/reference_dataset'
        path = self.find('*.npy', path)[0]
        predict_dataset = np.load(path)
        predict_dataset = predict_dataset.transpose((1, 0, 2))

        predict_dataset = predict_dataset[:, :, :121]
        for i in range(predict_dataset.shape[1]):
            ret, predict_dataset[:, i, :] = cv2.threshold(predict_dataset[:, i, :], 2, 100, cv2.THRESH_TOZERO)
            predict_dataset_mean = predict_dataset[:, i, :].mean()
            predict_dataset_std = predict_dataset[:, i, :].std()
            if predict_dataset_std == 0 and predict_dataset_mean == 0:
                continue
            predict_dataset[:, i, :] = (predict_dataset[:, i, :] - predict_dataset_mean) / predict_dataset_std

        oupput_lstm = lstm_model.predict(predict_dataset)

        x_size = int(path[-11:-8]) - 10
        y_size = int(path[-7:-4]) - 10

        output = np.zeros((x_size, y_size, oupput_lstm.shape[1]))
        output_path = 'data/evaluate/'+location+'/output'

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for j in range(predict_dataset.shape[1]):
            index_day = j
            lstm_conf = oupput_lstm[:, index_day, 0].reshape((x_size, y_size))
            ret, lstm_conf = cv2.threshold(lstm_conf, 4, 100, cv2.THRESH_BINARY)
            output[:, :, j] = lstm_conf
            o_min = lstm_conf.min()
            o_max = lstm_conf.max()
            time = "{:02d}:00".format(j % 24) + "to" + "{:02d}:59".format(j % 24) + "day{}".format(j//24)
            lstm_conf = ((lstm_conf - o_min) * (1 / (o_max - o_min) * 255)).astype('uint8')

            imsave(output_path+'/'+location+'_'+time+'.png', lstm_conf)
            # plt.imshow(lstm_conf, cmap='Greys')
            # plt.title(time)
            # plt.show()

    def find(self, pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    # def evaluate_mIoU_firms(self, location, reference_satellite, satellites):
    #     preprocessing = PreprocessingService()
    #     path = 'data/' + location + reference_satellite + '/'
    #     data_file_list = glob(str(Path(path) / "*.tif"))
    #     data_file_list.sort()
    #     for satellite in satellites:
    #         eval_path = 'data/' + location + satellite + '/'
    #         eval_file_list = glob(str(Path(eval_path) / "*.tif"))
    #         eval_file_list.sort()
    #         independent_ec = 0
    #         independent_eo = 0
    #         tp = 0
    #         tn = 0
    #         for file in data_file_list:
    #             firms_label, _ = preprocessing.read_tiff(file)
    #            if satellite != 'GOES':
    #                 acc_img = []
    #                 for eval_file in eval_file_list:
    #                     label_img, _ = preprocessing.read_tiff(eval_file)
    #                     label_img = np.flip(label_img[0,:,:], axis=0)
    #                     if satellite == 'FIRMS':
    #                         eval_date = datetime.datetime.strptime(eval_file.replace('data/label/'+location+' label/Cal_fire_'+location+'FIRMS-', '').replace('.tif', '')[:-17], '%Y-%m-%d')
    #                     else:
    #                         eval_date = datetime.datetime.strptime(
    #                             eval_file.replace('data/' + location + satellite+'/'+location+satellite+'_Cal_fire_' + location + satellite + '-',
    #                                               '').replace('.tif', ''), '%Y-%m-%d')
    #                     acc_img.append(label_img)
    #                 eval_img = np.nan_to_num(np.stack(acc_img, axis=2).sum(axis=2))
    #             else:
    #                 eval_img = np.load('data/evaluate/' + location + '/output_' + location + '_acc246*230.npy')
    #                 eval_img_resize = np.zeros((eval_img.shape[0]+10, eval_img.shape[1]+10))
    #                 eval_img_resize[5:-5, 5:-5] = eval_img
    #                 eval_img = eval_img_resize
    #             lb = 0
    #             ub = 1
    #             # plt.imshow(eval_img[round(lb*eval_img.shape[0]):round(ub*eval_img.shape[0]),round(lb*eval_img.shape[1]):round(ub*eval_img.shape[1])])
    #             plt.imshow(eval_img[round(lb*eval_img.shape[0]):round(ub*eval_img.shape[0]), round(lb*eval_img.shape[1]):round(ub*eval_img.shape[1])])
    #             plt.show()
    #             plt.imshow(img[round(lb*img.shape[0]):round(ub*img.shape[0]), round(lb*img.shape[1]):round(ub*img.shape[1])])
    #             plt.show()
    #             mapping_factor = (img.shape[0]/eval_img.shape[0], img.shape[1]/eval_img.shape[1])
    #             for i in range(round(lb*eval_img.shape[0]), round(ub*eval_img.shape[0])):
    #                 for j in range(round(lb*eval_img.shape[1]), round(ub*eval_img.shape[1])):
    #                     s2_patch = img[round(i*mapping_factor[0]):round((i+1)*mapping_factor[0]), round(j*mapping_factor[1]):round((j+1)*mapping_factor[1])]
    #                     if (s2_patch).max() == False and eval_img[i, j] != 0:
    #                         independent_ec += 1
    #                     elif (s2_patch).max() != False and eval_img[i, j] == 0:
    #                         independent_eo += 1
    #                     elif (s2_patch).max() != False and eval_img[i, j] != 0:
    #                         tp += 1
    #                     elif (s2_patch).max() == False and eval_img[i, j] == 0:
    #                         tn += 1
    #
    #         print('Error of Commission:{}'.format(independent_ec/(tp+independent_ec)))
    #         print('Error of Ommission:{}'.format(independent_eo/(tp+independent_eo)))
    #         print(tp, tn)
