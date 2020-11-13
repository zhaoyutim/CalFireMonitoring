import keras.backend as K
import numpy as np
import tensorflow as tf
from imageio import imsave

if __name__ == '__main__':
    dataset = np.load('data/train/August_complexGOES/August_complex dataset.npy')
    predict_dataset = np.load('data/train/creek_fireGOES/creek_fire dataset.npy')

    BATCH_SIZE = 64


    num_of_datapoints = dataset.shape[0]
    np.random.shuffle(dataset)

    train_dataset, val_dataset, test_dateset = dataset[:int(0.63*num_of_datapoints), :, :],  dataset[int(0.7*num_of_datapoints):int(0.9*num_of_datapoints), :, :], dataset[int(0.9*num_of_datapoints):, :, :]
    train_dataset, train_dataset_label = train_dataset[:,:,:27], train_dataset[:,:,27]
    val_dataset, val_dataset_label = val_dataset[:,:,:27], val_dataset[:,:,27]
    test_dateset, test_dateset_label = test_dateset[:,:,:27], test_dateset[:,:,27]

    train_mean = train_dataset.mean()
    train_std = train_dataset.std()

    train_label_mean = train_dataset_label.mean()
    train_label_std = train_dataset_label.std()

    train_dataset_norm = (train_dataset - train_mean) / train_std
    val_dataset_norm = (val_dataset - train_mean) / train_std
    test_dateset_norm = (test_dateset - train_mean) / train_std

    train_dataset_label_norm = (train_dataset_label - train_label_mean) / train_label_std
    val_dataset_label_norm = (val_dataset_label - train_label_mean) / train_label_std
    test_dateset_label_norm = (test_dateset_label - train_label_mean) / train_label_std


    train_dataset_as_tensor = tf.data.Dataset.from_tensor_slices((train_dataset_norm, train_dataset_label_norm))
    val_dataset_as_tensor = tf.data.Dataset.from_tensor_slices((val_dataset_norm, val_dataset_label_norm))
    test_dataset_as_tensor = tf.data.Dataset.from_tensor_slices((test_dateset_norm, test_dateset_label_norm))

    train_dataset_as_tensor = train_dataset_as_tensor.batch(BATCH_SIZE)
    val_dataset_as_tensor = val_dataset_as_tensor.batch(BATCH_SIZE)
    test_dataset_as_tensor = test_dataset_as_tensor.batch(BATCH_SIZE)
    #
    # train_dataset_for_dense = np.reshape(train_dataset_norm, (train_dataset_norm.shape[0]*train_dataset_norm.shape[1], train_dataset_norm.shape[2]))
    # train_dataset_for_dense_label = np.reshape(train_dataset_label_norm, (train_dataset_label_norm.shape[0]*train_dataset_label_norm.shape[1]))
    # train_dataset_for_dense = tf.data.Dataset.from_tensor_slices((train_dataset_for_dense, train_dataset_for_dense_label)).batch(BATCH_SIZE)
    #
    # val_dataset_for_dense = np.reshape(val_dataset_norm, (val_dataset_norm.shape[0]*val_dataset_norm.shape[1], val_dataset_norm.shape[2]))
    # val_dataset_for_dense_label = np.reshape(val_dataset_label_norm, (val_dataset_label_norm.shape[0]*val_dataset_label_norm.shape[1]))
    # val_dataset_for_dense = tf.data.Dataset.from_tensor_slices((val_dataset_for_dense, val_dataset_for_dense_label)).batch(BATCH_SIZE)
    #
    # test_dataset_for_dense = np.reshape(test_dateset_norm, (test_dateset_norm.shape[0]*test_dateset_norm.shape[1], test_dateset_norm.shape[2]))
    # test_dataset_for_dense_label = np.reshape(test_dateset_label_norm, (test_dateset_label_norm.shape[0]*test_dateset_label_norm.shape[1]))
    # test_dataset_for_dense = tf.data.Dataset.from_tensor_slices((test_dataset_for_dense, test_dataset_for_dense_label)).batch(BATCH_SIZE)

    # dense = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=1)
    # ])
    MAX_EPOCHS = 20
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


    def compile_and_fit(model, train_dataset, val_data_set, patience=2):
      early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')

      model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanSquaredError(), f1_m,])

      history = model.fit(train_dataset, epochs=MAX_EPOCHS,
                          validation_data=val_data_set,
                          callbacks=[early_stopping])
      return history

    # history = compile_and_fit(dense, train_dataset_for_dense, val_dataset_for_dense)
    val_performance={}
    performance={}
    # val_performance['Dense'] = dense.evaluate(val_dataset_for_dense)
    # performance['Dense'] = dense.evaluate(test_dataset_for_dense, verbose=0)


    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(130, input_shape=(130, 27), return_sequences=True),
        tf.keras.layers.LSTM(130, input_shape=(130, 27), return_sequences=True),
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(lstm_model, train_dataset_as_tensor, val_dataset_as_tensor)

    val_performance['LSTM'] = lstm_model.evaluate(val_dataset_as_tensor)
    performance['LSTM'] = lstm_model.evaluate(test_dataset_as_tensor, verbose=0)

    lstm_model.save('lstm_model')
    # dense.save('dense_model')

    predict_dataset_image = predict_dataset[:, :, :27]
    predict_dataset_mean = predict_dataset_image.mean()
    predict_dataset_std = predict_dataset_image.std()

    predict_dataset_label = predict_dataset[:, :, 27]
    predict_dataset_label_mean = predict_dataset_label.mean()
    predict_dataset_label_std = predict_dataset_label.std()

    predict_dataset_image_norm = (predict_dataset_image - predict_dataset_mean) / predict_dataset_std

    predict_dataset_label_norm = (predict_dataset_label - predict_dataset_label_mean) / predict_dataset_label_std

    predict_dataset_for_dense = np.reshape(predict_dataset_image, (predict_dataset_image.shape[0]*predict_dataset_image.shape[1], predict_dataset_image.shape[2]))


    # dense =  tf.keras.models.load_model('dense_model')
    # lstm_model = tf.keras.models.load_model('lstm_model')
    # output = dense.predict(predict_dataset_for_dense)
    # output = np.reshape(output, (predict_dataset.shape[0], predict_dataset.shape[1]))
    oupput_lstm = lstm_model.predict(predict_dataset_image_norm)
    x_size = 57
    y_size = 46
    for j in range(predict_dataset.shape[1]):
        index_day = j
        res = np.zeros((x_size, y_size, 3))
        res_confidence = np.zeros((x_size, y_size))
        conf_predicted = np.zeros((x_size, y_size))
        lstm_conf = np.zeros((x_size, y_size))
        for i in range(predict_dataset[:, index_day, 12].shape[0]):
            res_confidence[i // y_size, i % y_size] = predict_dataset[i, index_day, 27]*100
            # conf_predicted[i // y_size, i % y_size] = output[i, index_day]
            lstm_conf[i // y_size, i % y_size] = oupput_lstm[i, index_day]
            res[i // y_size, i % y_size, 0] = predict_dataset[i, index_day, 12]
            res[i // y_size, i % y_size, 1] = predict_dataset[i, index_day, 13]
            res[i // y_size, i % y_size, 2] = predict_dataset[i, index_day, 14]
        imsave('predict/creek fire original' + str(j) + '.png', res)
        imsave('predict/creek fire label' + str(j) + '.png', res_confidence)
        imsave('predict/creek fire conf_predicted' + str(j) + '.png', conf_predicted)
        imsave('predict/creek fire lstm_conf' + str(j) + '.png', lstm_conf)
    print(performance, val_performance)