# From: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

import pandas as pd
import seaborn as sns
from pandas import Series,DataFrame


def fill_data(data, filter_column, filter_value_list, filled_column, fill_value):
    for value in filter_value_list:
        data.loc[data[filter_column] == value, filled_column] = fill_value
        print(data.loc[data[filter_column] == value])
    return data


def fillna(data, column):
    # Replace missing fare value with median fare for class/embarkment
    data[column] = data[column].fillna(data[column].mean())
    return data


def filter_data(data, column, filter_value_list):
    # Passengers 62 and 830 are missing Embarkment
    # Get rid of our missing passenger IDs
    for value in filter_value_list:
        copy = data.loc[data[column] != value]
    #print data[column].describe()
    #print copy[column].describe()
    return copy


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
