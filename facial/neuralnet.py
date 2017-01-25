import os
import numpy as np
import pandas as pd
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from constants import IMAGE_SIZE
from utility import append_result


NeuralNet1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, IMAGE_SIZE * IMAGE_SIZE),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=50,  # we want to train this many epochs
    verbose=1,
    )

NeuralNet2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, IMAGE_SIZE, IMAGE_SIZE),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=5,
    verbose=1,
    )


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
        neuralnet.load_params_from(filename)
    return neuralnet


def predict(neuralnet, data, image_size, features):
     # RMSE is sqrt of last 'valid_loss' with scale back to image size
    last_valid_loss = neuralnet.train_history_[-1]['valid_loss']
    score = np.sqrt(last_valid_loss) * image_size /2
    
    # Data has to be reshaped to 4 dimensions
    predict = neuralnet.predict(data)
    predict = (predict + 1) * image_size / 2
    
    # Submission format is different to predict
    submission = convert_to_submission_format(features, predict)
    
    return submission, pd.DataFrame(predict, columns=features), score


def plot_neural_net(neuralnet, layer='input'):
    visualize.plot_conv_weights(neuralnet.layers_[layer])


def reshape_data(train, test, image_size):
    train = train.reshape(-1, 1, image_size, image_size)
    test = test.reshape(-1, 1, image_size, image_size)
    return train, test