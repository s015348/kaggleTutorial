# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_age(titanic_df, test_df):
    # Age 
    
    fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
    axis1.set_title('Original Age values - Titanic')
    axis2.set_title('New Age values - Titanic')
    
    # axis3.set_title('Original Age values - Test')
    # axis4.set_title('New Age values - Test')
    
    # get average, std, and number of NaN values in titanic_df
    average_age_titanic   = titanic_df["Age"].mean()
    std_age_titanic       = titanic_df["Age"].std()
    count_nan_age_titanic = titanic_df["Age"].isnull().sum()
    
    # get average, std, and number of NaN values in test_df
    average_age_test   = test_df["Age"].mean()
    std_age_test       = test_df["Age"].std()
    count_nan_age_test = test_df["Age"].isnull().sum()
    
    # generate random numbers between (mean - std) & (mean + std)
    rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
    rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)
    
    # plot original Age values
    # NOTE: drop all null values, and convert to int
    titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
    # test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
    
    # fill NaN values in Age column with random values generated
    titanic_df.loc[np.isnan(titanic_df["Age"]), "Age"] = rand_1
    test_df.loc[np.isnan(test_df["Age"]), "Age"] = rand_2
    
    # convert from float to int
    titanic_df['Age'] = titanic_df['Age'].astype(int)
    test_df['Age']    = test_df['Age'].astype(int)
            
    # plot new Age Values
    titanic_df['Age'].hist(bins=70, ax=axis2)
    # test_df['Age'].hist(bins=70, ax=axis4)
    
    # peaks for survived/not survived passengers by their age
    facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)
    facet.map(sns.kdeplot,'Age',shade= True)
    facet.set(xlim=(0, titanic_df['Age'].max()))
    facet.add_legend()
    
    # average survived passengers by age
    fig, axis1 = plt.subplots(1,1,figsize=(18,4))
    average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
    sns.barplot(x='Age', y='Survived', data=average_age)
    #sns.plt.show()