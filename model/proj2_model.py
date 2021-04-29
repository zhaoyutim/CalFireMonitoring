from keras.applications.vgg16 import VGG16
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imsave
import tensorflow as tf
import keras.backend as K
import datetime
import cv2

if __name__=='__main__':
    class SemisupervisedCNN(tf.keras.Model):
        def __init__(self):
            super(SemisupervisedCNN, self).__init__()
            self.vgg16_first = VGG16(include_top=False, input_shape=(dataset_arr.shape[2], dataset_arr.shape[3], 3))
            self.vgg16_second = VGG16(include_top=False, input_shape=(dataset_arr.shape[2], dataset_arr.shape[3], 3))
            self.upsampling_1 = tf.keras.layers.UpSampling2D(name='upsampling_16', size=(32, 32))
            self.conv_1 = tf.keras.layers.Convolution2D(1, 1, 1, name='deconv', activation='sigmoid')

        def call(self, input):
            input_viirs = input[None,:, :, 3:6]
            print(input_viirs.shape)
            input_goes = input[None,:, :, 0:3]
            embedding_viirs = self.vgg16_first(input_viirs)
            embedding_goes = self.vgg16_second(input_goes)
            embedding_concat = tf.concat([embedding_goes, embedding_viirs], 3)
            upsampling = self.upsampling_1(embedding_concat)
            output = self.conv_1(upsampling)
            return output

    dataset_arr = np.load('/Users/zhaoyu/PycharmProjects/CalFireMonitoring/data/train/dataset_proj2.npy')

    model = SemisupervisedCNN()


    optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    model.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy)

    train_dataset_as_tensor = tf.data.Dataset.from_tensor_slices(
        (dataset_arr[:, :6, :, :].transpose((0, 2, 3, 1)), dataset_arr[:, 6, :, :]))

    MAX_EPOCHS = 10
    logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
    history = model.fit(train_dataset_as_tensor, epochs=MAX_EPOCHS,
                        callbacks=[early_stopping, tensorboard_callback])