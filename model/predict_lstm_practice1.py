import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from IPython.display import Image, display
from array2gif import write_gif

if __name__ == '__main__':
    predict_dataset = np.load('/Users/zhaoyu/PycharmProjects/CalFireMonitoring/data/evaluate/creek_fire/reference_dataset/creek_fire dataset_ref.npy')
    predict_dataset = predict_dataset.transpose((1,0,2))
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
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    lstm_model = tf.keras.models.load_model('/Users/zhaoyu/PycharmProjects/CalFireMonitoring/model/lstm_model3', custom_objects={'f1_m':f1_m})


    predict_dataset_image = predict_dataset[:,:,:25]
    predict_dataset_mean = predict_dataset_image.mean()
    predict_dataset_std = predict_dataset_image.std()
    predict_dataset_image_norm = (predict_dataset_image - predict_dataset_mean) / predict_dataset_std
    oupput_lstm = lstm_model.predict(predict_dataset_image_norm)


    x_size = 310
    y_size = 252
    lstm_gif_list = []
    origin_gif_list = []
    label_gif_list = []
    concat_gif = []

    output = np.zeros((x_size, y_size, oupput_lstm.shape[1]))
    for j in range(oupput_lstm.shape[1]):
        index_day = j

        lstm_conf = oupput_lstm[:, index_day, 0].reshape((x_size, y_size))
        origin_pic = predict_dataset[:, index_day, 12].reshape((x_size, y_size))
        # ret, lstm_conf = cv2.threshold(lstm_conf, 2, 15, cv2.THRESH_BINARY)
        output[:,:,j] = lstm_conf

        lstm_conf = ((lstm_conf - lstm_conf.min()) * (1/(lstm_conf.max() - lstm_conf.min()) * 255)).astype('uint8')
        origin_pic = ((origin_pic - origin_pic.min()) * (1/(origin_pic.max() - origin_pic.min()) * 255)).astype('uint8')

        origin_gif = np.zeros((x_size, y_size, 3))
        lstm_conf_gif = np.zeros((x_size, y_size, 3))

        origin_gif[:, :, 0] = origin_pic
        origin_gif[:, :, 1] = origin_pic
        origin_gif[:, :, 2] = origin_pic

        lstm_conf_gif[:, :, 0] = lstm_conf
        lstm_conf_gif[:, :, 1] = lstm_conf
        lstm_conf_gif[:, :, 2] = lstm_conf

        lstm_gif_list.append(lstm_conf_gif.transpose((1,0,2)))
        origin_gif_list.append(origin_pic.transpose((1,0)))

        concat_gif.append(np.concatenate((lstm_conf_gif, origin_gif), axis=1).transpose((1,0,2)))
        # imsave('/content/drive/My Drive/predict/creek fire original' + str(j) + '.png', origin_pic.astype(np.uint8))
        # imsave('/content/drive/My Drive/predict/creek fire lstm_conf' + str(j) + '.png', lstm_conf.astype(np.uint8))

    write_gif(concat_gif, 'data/evaluate/gif/concat_gif.gif', fps=10)

    with open('data/evaluate/gif/concat_gif.gif','rb') as file:
        display(Image(file.read()))



