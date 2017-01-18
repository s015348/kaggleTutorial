import numpy as np
import pandas as pd
import csv
import time, datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler



def data_preview(train, test, label):
    print "Train data looks like:"
    print train.head(5)
    print "They are labeled as:"
    print label.head(5)
    print "Each pixel is an integer value:"
    print train.head(1).dtypes
    print "test data looks like:"
    print test.head(5)
    print "Each pixel is an integer value:"
    print test.head(1).dtypes


def plot_images(train, label):
    train_matrix = dataframe_row_to_matrix(train)
    fig, ax_array  = plt.subplots(2, 3, figsize=(20, 20))
    for i in range(6):
        ax_array[i/3, i%3].imshow(train_matrix[i], cmap='gray', interpolation='nearest');
    plt.show();

    
def dataframe_to_list(X_train, Y_train, X_test):
    X_train = X_train.values.tolist()
    Y_train = Y_train.values.tolist()
    X_test = X_test.values.tolist()
    return X_train, X_test, Y_train


def dataframe_row_to_matrix(df):
    result = []
    matrix_temp = df.as_matrix()
    for i in range(matrix_temp.shape[0]):
        result.append(matrix_temp[i].reshape((96,96)))
    return result


def get_image_data(data):
    return data['Image'].str.split(' ').apply(pd.Series).astype(int)


def load_data(dev_mode = False):
    # get facial key csv files as DataFrames
    # columns are:
    data_type = {'left_eye_center_x': np.float32, 'left_eye_center_y': np.float32, \
                 'right_eye_center_x': np.float32, 'right_eye_center_y': np.float32, \
                 'left_eye_inner_corner_x': np.float32, 'left_eye_inner_corner_y': np.float32, \
                 'left_eye_outer_corner_x': np.float32, 'left_eye_outer_corner_y': np.float32, \
                 'right_eye_inner_corner_x': np.float32, 'right_eye_inner_corner_y': np.float32, \
                 'right_eye_outer_corner_x': np.float32, 'right_eye_outer_corner_y': np.float32, \
                 'left_eyebrow_inner_end_x': np.float32, 'left_eyebrow_inner_end_y': np.float32, \
                 'left_eyebrow_outer_end_x': np.float32, 'left_eyebrow_outer_end_y': np.float32, \
                 'right_eyebrow_inner_end_x': np.float32, 'right_eyebrow_inner_end_y': np.float32, \
                 'right_eyebrow_outer_end_x': np.float32, 'right_eyebrow_outer_end_y': np.float32, \
                 'nose_tip_x': np.float32, 'nose_tip_y': np.float32, 'mouth_left_corner_x': np.float32, \
                 'mouth_left_corner_y': np.float32, 'mouth_right_corner_x': np.float32, \
                 'mouth_right_corner_y': np.float32, 'mouth_center_top_lip_x': np.float32, \
                 'mouth_center_top_lip_y': np.float32, 'mouth_center_bottom_lip_x': np.float32, \
                 'mouth_center_bottom_lip_y': np.float32, 'Image': np.str \
                 }
    train = pd.read_csv('./training.csv', dtype = data_type)
    if dev_mode:
        train = train.head(100)
    label = train.drop('Image', axis=1, inplace=False)
    train = get_image_data(train)
    test = pd.read_csv('./test.csv', dtype=np.str)
    test = get_image_data(test)
    return train, test, label


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


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    # scale train data
    X_train = scaler.transform(X_train) 
    # apply same transformation to test data
    X_test = scaler.transform(X_test)
    return X_train, X_test



# Set Dev mode for using small samples
DEV_MODE = True

# Count running time
starttime = datetime.datetime.now() 

# Load and preview data    
train, test, label = load_data(dev_mode = DEV_MODE)
data_preview(train, test, label)
plot_images(train, label)

# Format and scale data
#train, test, label = dataframe_to_list(train, label, test)
#train, test = scale_data(train, test)  
#print(label)


# Train and predict
#predict, score, mlp_model = mlp(train, label, test)
#save_result(test, predict)

# Count running time
endtime = datetime.datetime.now()  
#print("MLP score: %f, spend %d seconds" %(score, (endtime - starttime).seconds))
#show_mlp_weights(mlp_model)