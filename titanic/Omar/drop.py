# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import pandas as pd


def drop_useless(titanic_df, test_df):
    titanic_df, test_df = _drop_embarked(titanic_df, test_df)
    titanic_df, test_df = _drop_parch_sibsp(titanic_df, test_df)
    
    titanic_df, test_df = _drop_sex(titanic_df, test_df)
    titanic_df, test_df = _drop_person(titanic_df, test_df)
    # drop Male as it has the lowest average of survived passengers
    #person_dummies_titanic, person_dummies_test = _drop_male(person_dummies_titanic, person_dummies_test)
    titanic_df, test_df = _drop_male(titanic_df, test_df)
    titanic_df, test_df = _drop_pclass(titanic_df, test_df)
    titanic_df, test_df = _drop_class(titanic_df, test_df)
    
    titanic_df, test_df = _drop_cabin(titanic_df, test_df)
    titanic_df = drop_columes(titanic_df, ['PassengerId', 'Name', 'Ticket'])
    test_df = drop_columes(test_df, ['Name', 'Ticket'])

    return titanic_df, test_df
    

def _drop_class(pclass_dummies_titanic, pclass_dummies_test):
    pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)
    pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)
    return pclass_dummies_titanic, pclass_dummies_test


def _drop_cabin(titanic_df, test_df):
    # Cabin
    # It has a lot of NaN values, so it won't cause a remarkable impact on prediction
    #titanic_df.drop("Cabin",axis=1,inplace=True)
    #test_df.drop("Cabin",axis=1,inplace=True)
    titanic_df = drop_columes(titanic_df, ['Cabin'])
    test_df = drop_columes(test_df, ['Cabin'])
    return titanic_df, test_df   


def drop_columes(data, column_names=[], axis=1, inplace=False):
    # drop unnecessary columns, these columns won't be useful in analysis and prediction
    return data.drop(column_names, axis=axis, inplace=inplace)

 
def _drop_embarked(titanic_df, test_df):    
    # Either to consider Embarked column in predictions,
    # and remove "S" dummies variable, 
    # and leave "C" & "Q", since they seem to have a good rate for Survival.
    
    # OR, don't create dummies variables for Embarked column, just drop it, 
    # because logically, Embarked doesn't seem to be useful in prediction.
    
    embark_dummies_titanic = _remove_dummy_variable(titanic_df, 'Embarked', 'S')
    embark_dummies_test = _remove_dummy_variable(test_df, 'Embarked', 'S')
    
    titanic_df, test_df = _join_dummies(titanic_df, test_df, embark_dummies_titanic, embark_dummies_test)
    return _drop_embarked2(titanic_df, test_df)


def _drop_embarked2(titanic_df, test_df):
    titanic_df = drop_columes(titanic_df, ['Embarked'])
    test_df = drop_columes(test_df, ['Embarked'])
    return titanic_df, test_df


def _drop_parch_sibsp(titanic_df, test_df):
    # drop Parch & SibSp
    #titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
    #test_df    = test_df.drop(['SibSp','Parch'], axis=1)
    titanic_df = drop_columes(titanic_df, ['SibSp','Parch'])
    test_df = drop_columes(test_df, ['SibSp','Parch'])
    return titanic_df, test_df


def _drop_pclass(titanic_df, test_df):
    # drop Parch & SibSp
    #titanic_df.drop(['Pclass'],axis=1,inplace=True)
    #test_df.drop(['Pclass'],axis=1,inplace=True)
    titanic_df = drop_columes(titanic_df, ['Pclass'])
    test_df = drop_columes(test_df, ['Pclass'])
    return titanic_df, test_df


def _drop_male(person_dummies_titanic, person_dummies_test):
    person_dummies_titanic.drop(['Male'], axis=1, inplace=True)
    person_dummies_test.drop(['Male'], axis=1, inplace=True)
    return person_dummies_titanic, person_dummies_test


def _drop_sex(titanic_df, test_df):
    # No need to use Sex column since we created Person column
    #titanic_df.drop(['Sex'], axis=1, inplace=True)
    #test_df.drop(['Sex'], axis=1, inplace=True)
    titanic_df = drop_columes(titanic_df, ['Sex'])
    test_df = drop_columes(test_df, ['Sex'])
    return titanic_df, test_df


def _drop_person(titanic_df, test_df):
    titanic_df.drop(['Person'], axis=1, inplace=True)
    test_df.drop(['Person'], axis=1, inplace=True)
    return titanic_df, test_df


def _join_dummies(titanic_df, test_df, embark_dummies_titanic, embark_dummies_test):
    titanic_df = titanic_df.join(embark_dummies_titanic)
    test_df = test_df.join(embark_dummies_test)
    return titanic_df, test_df


def _remove_dummy_variable(data, column, value):
    dummies = pd.get_dummies(data[column])
    dummies.drop([value], axis=1, inplace=True)
    return dummies