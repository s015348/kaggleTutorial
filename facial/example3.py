# Ref: https://elix-tech.github.io/ja/2016/06/02/kaggle-facial-keypoints-ja.html

import time, datetime
import matplotlib.pyplot as plt
import keras
from constants import DEV_MODE, IMAGE_SIZE, SIMPLE_3LAYERS_FILENAME_KERAS, \
                    LENET5_CNN_FILENAME_KERAS, SIMPLE_3LAYERS_VISUALIZATION_FILENAME, \
                    LENET5_CNN_VISUALIZATION_FILENAME
from keras_neuralnet import NeuralNet1, NeuralNet2, load_model_if_exists, \
                            predict, plot_neural_net, reshape_data
from utility import clean_data, data_preview, get_label_args, load_data, save_result, scale_data
from visualization import plot_images, plot_label_distribution, plot_keras_learning_curves


# Count running time
starttime = datetime.datetime.now() 

# Load and preview data
train, test, label = load_data(dev_mode = DEV_MODE)
data_preview(train, test, label)
plot_images(train, label)
 
# Fill NA with average.
# TODO: remove abnormal samples
label = clean_data(label)
label_min, label_max, label_median, label_var = get_label_args(label)
plot_label_distribution(label)
 
# Convert dataframe to array and normalize the data
train_array, label_array, test_array = scale_data(train, test, label)
 
# Train and predict
 
# Simple 3-layers neural nets
print "\n[Simple 3-layers neural nets]\n"
NeuralNet1 = load_model_if_exists(NeuralNet1, SIMPLE_3LAYERS_FILENAME_KERAS)
training_history = NeuralNet1.fit(train_array, label_array, nb_epoch=10, validation_split=0.2)
NeuralNet1.save(SIMPLE_3LAYERS_FILENAME_KERAS)
submission, prediction, score = predict(NeuralNet1, test_array, 
                                        training_history, IMAGE_SIZE, label.columns.values)
save_result(submission)
plot_keras_learning_curves(training_history)
plot_neural_net(NeuralNet1, filename=SIMPLE_3LAYERS_VISUALIZATION_FILENAME)
plot_images(test, prediction)
  
# Count running time
endtime = datetime.datetime.now()
print("Simple 3-layers neural nets score: %f, spend %d seconds" %(score, (endtime - starttime).seconds))
 
# LeNet5-style convolutional neural nets
print "\n[LeNet5-style convolutional neural nets]\n"
train_array, test_array = reshape_data(train_array, test_array, IMAGE_SIZE)
NeuralNet2 = load_model_if_exists(NeuralNet2, LENET5_CNN_FILENAME_KERAS)
training_history = NeuralNet2.fit(train_array, label_array, nb_epoch=2, validation_split=0.2)
NeuralNet2.save(LENET5_CNN_FILENAME_KERAS)
submission, prediction, score = predict(NeuralNet2, test_array, 
                                        training_history, IMAGE_SIZE, label.columns.values)
save_result(submission)
plot_keras_learning_curves(training_history)
plot_neural_net(NeuralNet2, filename=LENET5_CNN_VISUALIZATION_FILENAME)
plot_images(test, prediction)
 
# Count running time
endtime = datetime.datetime.now()
print("LeNet5-style convolutional neural nets score: %f, spend %d seconds" %(score, (endtime - starttime).seconds))
 
plt.show()