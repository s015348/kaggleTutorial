# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

def show_fare(titanic_df, test_df):
    # Fare
    
    # only for test_df, since there is a missing "Fare" values
    test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
    
    # convert from float to int
    titanic_df['Fare'] = titanic_df['Fare'].astype(int)
    test_df['Fare']    = test_df['Fare'].astype(int)
    
    # get fare for survived & didn't survive passengers 
    fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
    fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]
    
    # get average and std for fare of survived/not survived passengers
    avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
    std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])
    
    # plot
    titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))
    
    avgerage_fare.index.names = std_fare.index.names = ["Survived"]
    avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
    #sns.plt.show()