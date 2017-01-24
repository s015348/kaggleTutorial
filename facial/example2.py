# Ref: http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

import time, datetime
import matplotlib.pyplot as plt
from constants import DEV_MODE, IMAGE_SIZE
from neuralnet import NeuralNet1, NeuralNet2, predict, reshape_data
from utility import clean_data, data_preview, get_label_args, load_data, save_result, scale_data
from visualization import plot_images, plot_label_distribution, plot_learning_curves


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
NeuralNet1.fit(train_array, label_array)
submission, prediction, score = predict(NeuralNet1, test_array, IMAGE_SIZE, label.columns.values)
save_result(submission)
plot_learning_curves(NeuralNet1)
plot_images(test, prediction)
 
# Count running time
endtime = datetime.datetime.now()
print("Simple 3-layers neural nets score: %f, spend %d seconds" %(score, (endtime - starttime).seconds))

# LeNet5-style convolutional neural nets
print "\n[LeNet5-style convolutional neural nets]\n"
train_array, test_array2 = reshape_data(train_array, test_array, IMAGE_SIZE)
NeuralNet2.fit(train_array, label_array)
submission, prediction, score = predict(NeuralNet2, test_array2, IMAGE_SIZE, label.columns.values)
save_result(submission)
plot_learning_curves(NeuralNet2)
plot_images(test, prediction)

# Count running time
endtime = datetime.datetime.now()
print("LeNet5-style convolutional neural nets score: %f, spend %d seconds" %(score, (endtime - starttime).seconds))

plt.show()