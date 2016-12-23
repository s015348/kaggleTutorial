# From: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utility import load_data, preview_data, set_print_style, get_features_correlation







def show_embarked_pclass_fare(data):
    # Visualize embarkment, passenger class, & median fare
    embark_fare = data[['Embarked', 'Fare']]
    embark_pclass = data[['Embarked', 'Pclass']]
    sns.barplot(x='Embarked', y='Fare', order=['C', 'S', 'Q'], data=embark_fare)
    sns.factorplot('Embarked', 'Pclass', order=['C', 'S', 'Q'], data=embark_pclass)

def filter_data(data, column, filter_value_list):
    # Passengers 62 and 830 are missing Embarkment
    # Get rid of our missing passenger IDs
    for value in filter_value_list:
        copy = data.loc[data[column] != value]
    #print data[column].describe()
    #print copy[column].describe()
    return copy

def get_deck(data):
    # Create a Deck variable. Get passenger deck A - F:
    data['Cabin'] = data['Cabin'].fillna('-')
    deck = data['Cabin'].str.get(0).to_frame()
    deck.columns = ['Deck']
    #print deck.head(10)
    data = data.join(deck['Deck'])
    #print data.head(10)
    return data

def show_family_survival(data):
    # Visualize the relationship between family size & survival
    family_survival = data[['Survived', 'Family size']]
    print family_survival.head(10)
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.countplot(x='Family size', data=family_survival, ax=axis1)
    sns.factorplot('Family size', 'Survived', order=range(1, 10), data=family_survival, ax=axis2)


def get_family_size(data):
    # Create a family size variable including the passenger themselves
    data['Family size'] = data['Parch'] + data['SibSp'] + 1
    #print data.head(10)


def show_title_by_sex(data):
    female = data.loc[data['Sex'].str.match('female')]
    male = data.loc[data['Sex'].str.match('male')]
    print female['Title'].value_counts()
    print male['Title'].value_counts()
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.countplot(x='Title', data=female, ax=axis1)
    sns.countplot(x='Title', data=male, ax=axis2)
    #sns.plt.show()


def merge_rare_title(data):
    # Titles with very low cell counts to be combined to 'rare' level
    data['Title'] = data['Title'].fillna('rare')
    title = data['Title'].str.replace('Mlle', 'Ms').\
                        str.replace('Miss', 'Ms').\
                        str.replace('Lady', 'Ms').\
                        str.replace('Mme', 'Mrs').\
                        str.replace('Dr', 'rare').\
                        str.replace('Col', 'rare').\
                        str.replace('Major', 'rare').\
                        str.replace('Jonkheer', 'rare').\
                        str.replace('Sir', 'rare').\
                        str.replace('Don', 'rare').\
                        str.replace('Capt', 'rare').\
                        str.replace('Rev', 'rare').\
                        to_frame()
    data = data.drop('Title', axis=1, inplace=False).join(title)
    return data


def get_person_title(data):
    # Grab surname_tf from passenger names
    surname_tf = data['Name'].str.split(',').str.get(1).\
                            str.split('.').str.get(0).\
                            to_frame()
    surname_tf.columns = ['Title']

    surname_tf = surname_tf.join(data['Sex'])
    #print surname_tf.head(10)
    return surname_tf


def get_person_surname(data):
    # Grab grab surname from passenger name
    surname_tf = data['Name'].str.split(',').str.get(1).\
                            str.split('.').str.get(1).\
                            to_frame()
    surname_tf.columns = ['Surname']
    #print surname_tf.head(10)
    return surname_tf

def fill_data(data, filter_column, filter_value_list, filled_column, fill_value):
    for value in filter_value_list:
        data.loc[data[filter_column] == value, filled_column] = fill_value
        print(data.loc[data[filter_column] == value])
    return data

def fillna(data, column):
    # Replace missing fare value with median fare for class/embarkment
    data[column] = data[column].fillna(data[column].mean())
    return data

def clean_data(data):
    data = fill_data(data, 'PassengerId', [62, 830], 'Embarked', 'C')
    data = fillna(data, 'Fare')    
    return data

# load data
train, test = load_data()
preview_data([train, test])

# Show title counts by sex_dummies
surname_tf = get_person_title(train)
show_title_by_sex(surname_tf)

surname_tf = merge_rare_title(surname_tf)
show_title_by_sex(surname_tf)

# Finally, grab surname from passenger name
surname_tf = get_person_surname(train)
print(surname_tf['Surname'].value_counts())
#TODO: how to get number of types when value_counts output long list?


# Show survival of family size
get_family_size(train)
show_family_survival(train)
#TODO: why there is an empty figure?

# Get deck from Cabin
train = get_deck(train)

# Find out empty embarked passenger and fill the value by fare
copy = filter_data(train, 'PassengerId', [62, 830])
show_embarked_pclass_fare(copy)
train = clean_data(train)

print(train.head(10))

#sns.plt.show()
