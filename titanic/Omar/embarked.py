# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import matplotlib.pyplot as plt
import seaborn as sns

def show_embarked(titanic_df):
    # Embarked
    # only in titanic_df, fill the two missing values with the most occurred value, which is "S".
    titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S") # plot
    sns.factorplot('Embarked', 'Survived', data=titanic_df, size=4, aspect=3) #sns.plt.show()
    fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15, 5))
    # sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
    # sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
    sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
    sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1, 0], ax=axis2)
    # group by embarked, and get the mean for survived passengers for each value in Embarked
    embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
    sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3)
    #sns.plt.show()