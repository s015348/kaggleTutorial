import numpy as np
import matplotlib.pyplot as plt
from constants import IMAGE_SIZE
from utility import get_one_image_in_matrix

def plot_images(images, label):
    # some visualization of train data and labels
    label_matrix = label.as_matrix()
    fig, ax_array  = plt.subplots(2, 3, figsize=(20, 20))
    # plot first six faces together with their labels
    for i in range(6):
        ax_array[i/3, i%3].imshow(get_one_image_in_matrix(images, IMAGE_SIZE, i), 
                                  cmap='gray', interpolation='nearest')
        ax_array[i/3, i%3].plot(label_matrix[i][0::2], label_matrix[i][1::2], 'ro')
    # plot all labels over first face
    for i in range(label_matrix.shape[0]):
        ax_array[0, 0].plot(label_matrix[i][0::2], label_matrix[i][1::2], 'go')       
    #plt.show()
    #pass


def plot_label_distribution(label):
    features = label.columns.values
    fig, ax_array  = plt.subplots(6, 5, figsize=(20, 20))
    index = 0
    for feature in features:
        x = label[feature].as_matrix()
        # the histogram of the data
        ax_array[index/5, index%5].set_title(feature)
        ax_array[index/5, index%5].hist(x, 50, normed=False, facecolor='g', alpha=0.75)
        # next feature to next subplot
        index = index + 1
    #plt.show()
    

def plot_learning_curves(neuralnet):
    train_loss = np.array([i["train_loss"] for i in neuralnet.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in neuralnet.train_history_])
    y_min = np.asarray([train_loss, valid_loss]).min() #1e-3
    y_max = np.asarray([train_loss, valid_loss]).max() #1e-0

    plt.figure()
    plt.plot(train_loss, linewidth=3, label="Train loss")
    plt.plot(valid_loss, linewidth=3, label="Valid loss")
    plt.grid()
    plt.legend()
    plt.ylim(y_min, y_max)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")