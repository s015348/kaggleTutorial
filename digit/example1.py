import numpy as np
import pandas as pd
import csv
import time, datetime
from sklearn.neighbors import KNeighborsClassifier



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


def knn(X_train, Y_train, X_test, parameter_n=2):
    # k-nearest neighbors algorithm
    knn = KNeighborsClassifier(n_neighbors = parameter_n)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    score = knn.score(X_train, Y_train)
    return Y_pred, score

    
def normalize_data(data):
    # Replace all non-zero values as one
    return data.applymap(lambda x:1 if x > 0 else 0)


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
DEV_MODE = True

# Count running time
starttime = datetime.datetime.now() 

# Load and preview data    
train, test, label = load_data(dev_mode = DEV_MODE)
#data_preview(train, test, label)

# Replace all non-zero values as one
train = normalize_data(train)
#print(train)

# Train and predict
predict, score = knn(train, label, test, parameter_n = 10)
save_result(test, predict)

# Count running time
endtime = datetime.datetime.now()  
print("KNN score: %f, spend %d seconds" %(score, (endtime - starttime).seconds))