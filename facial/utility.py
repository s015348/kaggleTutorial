import numpy as np
import pandas as pd
import csv
import time, datetime
from constants import DEV_TRAIN_LEN, DEV_TEST_LEN, IMAGE_SIZE, PATCH_SIZE


def append_result(dataframe, image_id, feature_name, location, max_index):
    data = pd.DataFrame({'ImageId':image_id, 'FeatureName':feature_name, 
                         'Location':location, 'Max_index':max_index})
    dataframe = dataframe.append(data, ignore_index=True)
    return dataframe


def clean_data(data):
    data = data.apply(lambda x:x.fillna(x.median()))
    margin = IMAGE_SIZE - PATCH_SIZE
    data = data.applymap(lambda x:PATCH_SIZE if x < PATCH_SIZE else margin if x > margin else x)
    return data


def convert_matrix_to_array(matrix, size):
    return matrix.reshape(1, size)[0]


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

    
# def dataframe_to_list(X_train, Y_train, X_test):
#     X_train = X_train.values.tolist()
#     Y_train = Y_train.values.tolist()
#     X_test = X_test.values.tolist()
#     return X_train, X_test, Y_train


def get_image_data(data):
    return data['Image'].str.split(' ').apply(pd.Series).astype(int)


def get_label_args(label):
    return label.min(axis=0), label.max(axis=0), \
            label.median(axis=0), label.var(axis=0)
    
    
def get_one_image_in_matrix(images, size, i=0):
    if len(images.shape) == 2:
        image_matrix = images.as_matrix()[i].reshape((size, size))
    elif len(images.shape) == 1:
        image_matrix = images.as_matrix().reshape((size, size))
    else:
        print "ERROR: input data has wrong dimensions, check it."
        image_matrix = []
    return image_matrix


def load_data(dev_mode = False):
    print "Loading data..."
    
    # Count running time
    starttime = datetime.datetime.now() 

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
    # Count running time
    endtime = datetime.datetime.now()
    print("Training data loaded, spend %d seconds" %(endtime - starttime).seconds)
    starttime = endtime

    test = pd.read_csv('./test.csv', dtype=np.str)
    # Count running time
    endtime = datetime.datetime.now()
    print("Testing data loaded, spend %d seconds" %(endtime - starttime).seconds)
    starttime = endtime

    if dev_mode:
        print("In Dev mode, only use %d train images and %d test images" %(DEV_TRAIN_LEN, DEV_TEST_LEN))
        train = train.head(DEV_TRAIN_LEN)
        test = test.head(DEV_TEST_LEN)

    print "Parsing data..."
    label = train.drop('Image', axis=1, inplace=False)
    train = get_image_data(train)
    test = get_image_data(test)

    # Count running time
    endtime = datetime.datetime.now()
    print("Data parsed, spend %d seconds" %(endtime - starttime).seconds)
    return train, test, label


def save_result(result):
    # Load lookup table with RowId and ImageId
    id_lookup_table = pd.read_csv('./IdLookupTable.csv')
    
    # Drop useless columns
    result.drop(['Max_index'], axis=1, inplace=True)
    id_lookup_table.drop(['Location'], axis=1, inplace=True)
    print "Prediction and lookup table look like:"
    print result.head(5)
    print id_lookup_table.head(5)
    
    # Join result with lookup table,
    # then sort by RowId, reorder the columns,
    # confirm column types, and set Rowid as index
    result = result.merge(id_lookup_table, how='left', sort=False, 
                          on=['FeatureName','ImageId'])\
                    .sort(["RowId"], ascending=True)\
                    [['RowId', 'ImageId', 'FeatureName', 'Location']]\
                    .apply(lambda x:x.fillna(0))\
                    .astype({'RowId': np.int32, 'ImageId': np.int32, 
                             'FeatureName': np.str, 'Location': np.float32})\
                    .set_index('RowId', inplace=False)
    print "We guess the locations are:"
    print result.head(10)
    # Write your solution to a csv file with the name my_solution.csv
    suffix = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filename = 'mysolution' + suffix + '.csv'
    result.to_csv(filename, index_label=["RowId"])
    print "Prediction was written to my_solution.csv"    
 

# def scale_data(X_train, X_test):
#     scaler = StandardScaler()
#     scaler.fit(X_train)
#     # scale train data
#     X_train = scaler.transform(X_train) 
#     # apply same transformation to test data
#     X_test = scaler.transform(X_test)
#     return X_train, X_test

