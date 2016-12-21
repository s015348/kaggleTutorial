# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utility import drop_columes

def show_pclass(titanic_df, test_df):
    # Pclass
    
    # sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])
    sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)
    
    # create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
    pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
    pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
    
    pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
    pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
    pclass_dummies_titanic, pclass_dummies_test = _drop_class(pclass_dummies_titanic, pclass_dummies_test)
    
    titanic_df = titanic_df.join(pclass_dummies_titanic)
    test_df    = test_df.join(pclass_dummies_test)
    sns.plt.show()
    
    return _drop_pclass(titanic_df, test_df)


def _drop_class(pclass_dummies_titanic, pclass_dummies_test):
    pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)
    pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)
    return pclass_dummies_titanic, pclass_dummies_test

def _drop_pclass(titanic_df, test_df):
    # drop Parch & SibSp
    #titanic_df.drop(['Pclass'],axis=1,inplace=True)
    #test_df.drop(['Pclass'],axis=1,inplace=True)
    titanic_df = drop_columes(titanic_df, ['Pclass'])
    test_df = drop_columes(test_df, ['Pclass'])
    return titanic_df, test_df