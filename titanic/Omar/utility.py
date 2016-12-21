# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import pandas as pd
import seaborn as sns

def drop_columes(data, column_names=[]):
    # drop unnecessary columns, these columns won't be useful in analysis and prediction
    return data.drop(column_names, axis=1)

def load_data():
    # get titanic & test csv files as a DataFrame
    titanic_df = pd.read_csv('../train.csv')
    test_df = pd.read_csv('../test.csv')
    return titanic_df, test_df

def preview_data(data_list):
    for data in data_list:
        print (data.head())
        data.info()
        print ("----------------------------")

def set_print_style():
    return sns.set_style('whitegrid')
