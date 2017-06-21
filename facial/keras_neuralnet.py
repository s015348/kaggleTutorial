import os
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model as plot
from constants import KERAS_BACKEND, IMAGE_SIZE
from utility import append_result


def neuralnet1(imput_dim):
    neuralnet = Sequential()
    neuralnet.add(Dense(100, input_dim=imput_dim))
    neuralnet.add(Activation('relu'))
    neuralnet.add(Dense(30))
    neuralnet.compile(loss='mean_squared_error',
                      optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
    return neuralnet


def neuralnet2(input_shape):
    neuralnet = Sequential()

    neuralnet.add(Convolution2D(32, (3, 3), input_shape=input_shape))
    neuralnet.add(Activation('relu'))
    neuralnet.add(MaxPooling2D(pool_size=(2, 2)))

    neuralnet.add(Convolution2D(64, (2, 2)))
    neuralnet.add(Activation('relu'))
    neuralnet.add(MaxPooling2D(pool_size=(2, 2)))

    neuralnet.add(Convolution2D(128, (2, 2)))
    neuralnet.add(Activation('relu'))
    neuralnet.add(MaxPooling2D(pool_size=(2, 2)))

    neuralnet.add(Flatten())
    neuralnet.add(Dense(500))
    neuralnet.add(Activation('relu'))
    neuralnet.add(Dense(500))
    neuralnet.add(Activation('relu'))
    neuralnet.add(Dense(30))

    neuralnet.compile(loss='mean_squared_error',
                      optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
    return neuralnet


if KERAS_BACKEND == 'th':
    input_shape = (1, (IMAGE_SIZE, IMAGE_SIZE))
elif KERAS_BACKEND == 'tf':
    input_shape = (IMAGE_SIZE, (IMAGE_SIZE, 1))
else:
    print("ERROR: please check constant KERAS_BACKEND which should be either th or tf")
NeuralNet1 = neuralnet1(IMAGE_SIZE ** 2)
NeuralNet2 = neuralnet2(input_shape)


def convert_to_submission_format(features, predict):
    row_num = predict.shape[0]
    feature_num = predict.shape[1] / 2
    submission = pd.DataFrame()
    for row in range(row_num):
        for i in range(feature_num):
            submission = append_result(submission, [row + 1], features[i * 2], predict[row][i], -1)
            submission = append_result(submission, [row + 1], features[i * 2 + 1], predict[row][i + 1], -1)
    return submission


def load_model_if_exists(neuralnet, filename):
    if os.path.exists(filename):
        neuralnet = load_model(filename)
    return neuralnet


def predict(neuralnet, data, history, image_size, features):
    # RMSE is sqrt of last 'valid_loss' with scale back to image size
    last_valid_loss = history.history['val_loss'][-1]
    score = np.sqrt(last_valid_loss) * image_size /2

    # Data has to be reshaped to 4 dimensions
    predict = neuralnet.predict(data)
    predict = (predict + 1) * image_size / 2

    # Submission format is different to predict
    submission = convert_to_submission_format(features, predict)

    return submission, pd.DataFrame(predict, columns=features), score


def plot_neural_net(neuralnet, filename='net-visualzation.png'):
    plot(neuralnet, to_file=filename, show_shapes=True, show_layer_names=True)


def reshape_data(train, test, image_size):
    if KERAS_BACKEND == 'th':
        train = train.reshape(-1, 1, image_size, image_size)
        test = test.reshape(-1, 1, image_size, image_size)
    elif KERAS_BACKEND == 'tf':
        train = train.reshape(-1, image_size, image_size, 1)
        test = test.reshape(-1, image_size, image_size, 1)
    else:
        print("ERROR: please check constant KERAS_BACKEND which should be either th or tf")
    return train, test
