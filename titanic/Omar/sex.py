# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utility import drop_columes





def show_sex(titanic_df, test_df):
    # Sex
    
    titanic_df['Person'] = titanic_df[['Age','Sex']].apply(_get_person,axis=1)
    test_df['Person']    = test_df[['Age','Sex']].apply(_get_person,axis=1)
    
    titanic_df, test_df = _drop_sex(titanic_df, test_df)
    
    # create dummy variables for Person column, 
    person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
    person_dummies_titanic.columns = ['Child','Female','Male']
    
    person_dummies_test  = pd.get_dummies(test_df['Person'])
    person_dummies_test.columns = ['Child','Female','Male']
    
    # drop Male as it has the lowest average of survived passengers
    person_dummies_titanic, person_dummies_test = _drop_male(person_dummies_titanic, person_dummies_test)
    
    titanic_df = titanic_df.join(person_dummies_titanic)
    test_df    = test_df.join(person_dummies_test)
    
    fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))
    
    # sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
    sns.countplot(x='Person', data=titanic_df, ax=axis1)
    
    # average of survived for each Person(male, female, or child)
    person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
    sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])
    #sns.plt.show()
    return _drop_person(titanic_df, test_df)


def _drop_sex(titanic_df, test_df):
    # No need to use Sex column since we created Person column
    #titanic_df.drop(['Sex'], axis=1, inplace=True)
    #test_df.drop(['Sex'], axis=1, inplace=True)
    titanic_df = drop_columes(titanic_df, ['Sex'])
    test_df = drop_columes(test_df, ['Sex'])
    return titanic_df, test_df


def _drop_male(person_dummies_titanic, person_dummies_test):
    person_dummies_titanic.drop(['Male'], axis=1, inplace=True)
    person_dummies_test.drop(['Male'], axis=1, inplace=True)
    return person_dummies_titanic, person_dummies_test


def _drop_person(titanic_df, test_df):
    titanic_df.drop(['Person'], axis=1, inplace=True)
    test_df.drop(['Person'], axis=1, inplace=True)
    return titanic_df, test_df


# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def _get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
