# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import pandas as pd
import seaborn as sns


def create_sub_pclass(titanic_df, test_df):
    # create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
    pclass_dummies_titanic = pd.get_dummies(titanic_df['Pclass'])
    pclass_dummies_titanic.columns = ['Class_1', 'Class_2', 'Class_3']
    
    pclass_dummies_test = pd.get_dummies(test_df['Pclass'])
    pclass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']
    
    titanic_df = titanic_df.join(pclass_dummies_titanic)
    test_df = test_df.join(pclass_dummies_test)
    
    return titanic_df, test_df


def show_pclass(titanic_df, test_df):
    # Pclass
   
    # sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])
    sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)
    #sns.plt.show()
    return titanic_df, test_df
