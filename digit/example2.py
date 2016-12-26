import numpy as np
import pandas as pd
import csv
import time, datetime
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def data_preview(train, test, label):
    print "Train data looks like:"
    print train.head(5)
    print "They are labeled as:"
    print label.head(5)
    print "Each pixel is an integer value:"
    print train.head(1).dtypes
    print "Train data overview:"
    print train.describe()
    print "Test data overview:"
    print test.describe()


def load_data(dev_mode = False):
    # get digit csv files as DataFrames
    train = pd.read_csv('./train.csv', dtype=int)
    if dev_mode:
        train = train.head(100)
    label = train['label']
    label.column = ['label']
    train.drop('label', axis=1, inplace=True)
    test = pd.read_csv('./test.csv', dtype=int)
    return train, test, label


def dataframe_to_list(X_train, Y_train, X_test):
    X_train = X_train.values.tolist()
    Y_train = Y_train.values.tolist()
    X_test = X_test.values.tolist()
    return X_train, X_test, Y_train


def mlp(X_train, Y_train, X_test):
    #Multi-layer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1, warm_start=True)
    warmup_steps = 1
    subset_size = len(X_train) / warmup_steps
    for i in range(warmup_steps):
        start = subset_size * i
        end = subset_size * (i + 1) - 1
        X_piece = X_train[start : end]
        Y_piece = Y_train[start : end]
        print("Using data subsets from #%d to #%d" % (start, end))
        mlp.fit(X_piece, Y_piece) 
    Y_pred = mlp.predict(X_test)
    score = mlp.score(X_train, Y_train)
    return Y_pred, score, mlp


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    # scale train data
    X_train = scaler.transform(X_train) 
    # apply same transformation to test data
    X_test = scaler.transform(X_test)
    return X_train, X_test


def show_mlp_weights(mlp):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10)) 
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())
    
    plt.show()


def save_result(test, predict):
    # Create a data frame with two columns: ImageId & Label
    ImageId = np.array(range(1, len(test) + 1)).astype(int)
    my_solution = pd.DataFrame(predict, ImageId, columns=['Label'])
    print "We guess the test data are:"
    print my_solution.head(10)
    
    # Write your solution to a csv file with the name my_solution.csv
    #np.savetxt("./my_solution.csv", my_solution, fmt='%d', delimiter=",")
    suffix = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filename = 'mysolution' + suffix + '.csv'
    my_solution.to_csv(filename, index_label=["ImageId"])
    print "Prediction was written to my_solution.csv"



# Set Dev mode for using small samples
DEV_MODE = False

# Count running time
starttime = datetime.datetime.now() 

# Load and preview data    
train, test, label = load_data(dev_mode = DEV_MODE)
#data_preview(train, test, label)

# Format and scale data
train, test, label = dataframe_to_list(train, label, test)
train, test = scale_data(train, test)  
#print(train)

# Train and predict
predict, score, mlp_model = mlp(train, label, test)
save_result(test, predict)

# Count running time
endtime = datetime.datetime.now()  
print("MLP score: %f, spend %d seconds" %(score, (endtime - starttime).seconds))
show_mlp_weights(mlp_model)