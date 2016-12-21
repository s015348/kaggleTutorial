# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import matplotlib.pyplot as plt
import seaborn as sns

from utility import drop_columes


def show_family(titanic_df, test_df):
    # Family
    
    # Instead of having two columns Parch & SibSp, 
    # we can have only one column represent if the passenger had any family member aboard or not,
    # Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
    titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]
    titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
    titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0
    
    test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
    test_df['Family'].loc[test_df['Family'] > 0] = 1
    test_df['Family'].loc[test_df['Family'] == 0] = 0
    
    # plot
    fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))
    
    # sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)
    sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)
    
    # average of survived for those who had/didn't have any family member
    family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
    sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)
    
    axis1.set_xticklabels(["With Family","Alone"], rotation=0)
    #sns.plt.show()
    
    return _drop_parch_sibsp(titanic_df, test_df)


def _drop_parch_sibsp(titanic_df, test_df):
    # drop Parch & SibSp
    #titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
    #test_df    = test_df.drop(['SibSp','Parch'], axis=1)
    titanic_df = drop_columes(titanic_df, ['SibSp','Parch'])
    test_df = drop_columes(test_df, ['SibSp','Parch'])
    return titanic_df, test_df