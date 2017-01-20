import numpy as np
import pandas as pd
import time, datetime
import matplotlib.pyplot as plt
from constants import DEV_MODE, DEV_TRAIN_LEN, IMAGE_SIZE, PATCH_SIZE, \
                        FEATURE_NUM, SEARCH_STEP, EVALUATION_LEN, LOG_INTERVAL
from utility import append_result, convert_matrix_to_array, get_one_image_in_matrix
from visualization import plot_images                 
                    
                    
def extract_patch(images, label, x_column, y_column):
    label_matrix = label.as_matrix()
    patch_matrix = []
    
    # extract patches from all images for one specified feature
    for i in range(label_matrix.shape[0]):
        x_center = int(label_matrix[i][x_column])
        y_center = int(label_matrix[i][y_column])
        image_matrix = get_one_image_in_matrix(images, IMAGE_SIZE, i)
        patch = get_image_grid(image_matrix, x_center, y_center, PATCH_SIZE)
        patch_matrix.append(convert_matrix_to_array(patch, (PATCH_SIZE * 2) ** 2))
    
    # output the average over all images    
    patch_df = pd.DataFrame(patch_matrix)
    patch_avg = patch_df.apply(lambda x:x.median())
    return patch_avg


def evaluate_patch_error(label, label_predict, features):
    error = pd.DataFrame()
    for feature in features:
        error[feature] = label[feature] - label_predict[feature]
    
#     print "Error over all features looks like:"
#     print error.head(10)
    
    return error.stack().std()


def find_optimal_range(label, features, label_predict, label_median, label_var):
    result = np.empty((0, 2), float)
    for r in np.arange(0, 3, 0.1):
        label_temp = label_predict.copy(deep=True)
        label_temp = replace_abnormal_prediction(label_temp, features, label_median, label_var, r=r)
        score = evaluate_patch_error(label, label_temp, features)
        result = np.append(result, np.array([[r, score]]), axis=0)
    
    #plot_re(result)
    min_index = np.argmin(result, axis=0)[1]
    optimal_r = result[min_index][0]
    print("Got optimal r: %f" %(optimal_r))
    return optimal_r


def get_image_grid(image_matrix, x_center, y_center, size):
    return image_matrix[x_center - size: x_center + size, 
                        y_center - size: y_center + size]


def get_patches(images, label, features_num):
    print "Calculating patches of all features..."
    
    # Count running time
    starttime = datetime.datetime.now() 
    
    patches = []
    fig, ax_array  = plt.subplots(3, 5, figsize=(20, 20))
    
    # process features one by one
    for i in range(features_num):
        x_column = i * 2
        y_column = x_column + 1
        patch = extract_patch(images, label, x_column, y_column)
        patches.append(patch)
        # plot all 15 features one by one
        ax_array[i/5, i%5].imshow(get_one_image_in_matrix(patch, PATCH_SIZE * 2), 
                                  cmap='gray', interpolation='nearest')
    #plt.show()
    
    # Count running time
    endtime = datetime.datetime.now()
    print("Training end, spend %d seconds" %(endtime - starttime).seconds)
    return patches


def predict_one_image(images, image_index, patches, features, submission, starttime):
    # as submission format is different with labels, 
    # create both format in submission and labels
    label_list = []
    # note ImageId isn't image index
    image_id = image_index + 1
    
    # process each feature
    for i in range(FEATURE_NUM):
        patch = patches[i]
        result = scan_one_image(images, patch, image_index)
        max_index = np.argmax(result, axis=0)[2]
        #print image_index, features[i * 2], result[max_index][0], max_index
        submission = append_result(submission, [image_id], features[i * 2], result[max_index][0], max_index)
        submission = append_result(submission, [image_id], features[i * 2 + 1], result[max_index][1], max_index)
        label_list.append([features[i * 2], result[max_index][0:1]])
        label_list.append([features[i * 2 + 1], result[max_index][1:2]])

    if image_id%LOG_INTERVAL == 0:
        # Only print log info for each 100 predictions
        # Count running time
        endtime = datetime.datetime.now()
        print("Predicted #%d image, spend %d seconds" %(image_id, (endtime - starttime).seconds))
        # update starttime for next counting
        starttime = endtime
    return submission, dict(label_list), starttime


def replace_abnormal_prediction(label, features, median, var, r=1):
    for feature in features:
        start = int(median[feature] - var[feature] * r)
        end = int(median[feature] + var[feature] * r)
        label.loc[(label[feature].between(start, end)) == False, 
                  feature] = median[feature]
    return label 


def scan_one_image(images, patch, i=0):
    result = np.empty((0,3), int)
    size = (PATCH_SIZE * 2) ** 2
    # scan one image by taking grids in same size as patch, 
    # then calculate their correlation and find the maximum grid
    for x in range(PATCH_SIZE, IMAGE_SIZE - PATCH_SIZE, SEARCH_STEP):
        for y in range(PATCH_SIZE, IMAGE_SIZE - PATCH_SIZE, SEARCH_STEP):
            image_matrix = get_one_image_in_matrix(images, IMAGE_SIZE, i)
            search_grid = get_image_grid(image_matrix, x, y, PATCH_SIZE)
            patch = convert_matrix_to_array(patch, size)
            search = convert_matrix_to_array(search_grid, size)
            patch = patch / np.linalg.norm(patch)
            search = search / np.linalg.norm(search)
            result = np.append(result, np.array([[x, y, np.correlate(patch, search)[0]]]), axis=0)
            #result.append([x, y, np.cov(np.asarray([patch, search]))[0][1]])
#     size = int((result.size/3) ** 0.5)
#     print "size is:"
#     print size
#     if size == (IMAGE_SIZE - PATCH_SIZE * 2)/SEARCH_STEP:
#         result_plot = result[:, 2:3].reshape(size, size)
#         print "maximum correlation is:"
#         print result_plot.max()
#         plt.figure(1)
#         plt.imshow(result_plot, cmap='gray', interpolation='nearest')
#         plt.show()
    return result


def use_image_patches(train, test, label, label_median, label_var):
    # Get average values of patches located surround label points
    patches = get_patches(train, label, FEATURE_NUM)
    
    features = label.columns.values
    
    # Before predict test data, apply prediction on train and score
    submission = pd.DataFrame()
    label_predict = pd.DataFrame()
    print "Evaluating model..."
    timer_p = datetime.datetime.now()
    if DEV_MODE:
        eval_len = DEV_TRAIN_LEN
    else:
        eval_len = EVALUATION_LEN
    train_evaluation = train.head(eval_len).copy(deep=True)
    for i in range(eval_len):
        submission, label_dict, timer_p = predict_one_image(train_evaluation, i, 
                                            patches, features, submission, timer_p)
        label_predict = label_predict.append(pd.DataFrame(label_dict), ignore_index=True)

    if DEV_MODE:
        print "Optimizing model..."
        optimal_r = find_optimal_range(label, features, 
                        label_predict, label_median, label_var)
    else:
        print "Filter prediction out of circle (median, variance) ..."
        optimal_r = 0.5
    label_predict = replace_abnormal_prediction(label_predict, features, 
                                    label_median, label_var, r=optimal_r)
    plot_images(train, label_predict)
    
    score = evaluate_patch_error(label, label_predict, features)
    print("Score evaluated by partial train data: %f" %(score))    
    
    # Make prediction by search grid with maximum correlation
    submission = pd.DataFrame()
    label_predict = pd.DataFrame()
    print "Predicting test data..."
    timer_p = datetime.datetime.now() 
    for i in range(test[0].count()):
        submission, label_dict, timer_p = predict_one_image(test, i, 
                                            patches, features, submission, timer_p)
        label_predict = label_predict.append(pd.DataFrame(label_dict))
    label_predict = replace_abnormal_prediction(label_predict, features, 
                                    label_median, label_var, r=optimal_r)         
    # Plot images for manual check
    plot_images(test, label_predict)

    print "Prediction in submission format looks like:"
    print submission
    print "Prediction in label format looks like:"
    print label_predict
    return submission, score

