# From: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

import seaborn as sns
import matplotlib.pyplot as plt


def show_embarked_pclass_fare(data):
    # Visualize embarkment, passenger class, & median fare
    embark_fare = data[['Embarked', 'Fare']]
    embark_pclass = data[['Embarked', 'Pclass']]
    sns.barplot(x='Embarked', y='Fare', order=['C', 'S', 'Q'], data=embark_fare)
    sns.factorplot('Embarked', 'Pclass', order=['C', 'S', 'Q'], data=embark_pclass)
    #sns.plt.show()


def show_family_survival(data):
    # Visualize the relationship between family size & survival
    family_survival = data[['Survived', 'Family size']]
    print family_survival.head(10)
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.countplot(x='Family size', data=family_survival, ax=axis1)
    sns.factorplot('Family size', 'Survived', order=range(1, 10), data=family_survival, ax=axis2)
    #sns.plt.show()


def show_title_by_sex(data):
    female = data.loc[data['Sex'].str.match('female')]
    male = data.loc[data['Sex'].str.match('male')]
    print female['Title'].value_counts()
    print male['Title'].value_counts()
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.countplot(x='Title', data=female, ax=axis1)
    sns.countplot(x='Title', data=male, ax=axis2)
    #sns.plt.show()


def show_surname_amount(surname_tf):
    print surname_tf['Surname'].value_counts()
