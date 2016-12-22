# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import pandas as pd
import seaborn as sns
from pandas import Series,DataFrame


def get_features_correlation(data, logreg):
    # get Correlation Coefficient for each feature using Logistic Regression
    coeff_df = DataFrame(data.columns.delete(0))
    coeff_df.columns = ['Features']
    coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
    return coeff_df

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
