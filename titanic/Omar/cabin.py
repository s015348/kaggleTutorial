# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

from utility import drop_columes

def drop_cabin(titanic_df, test_df):
    # Cabin
    # It has a lot of NaN values, so it won't cause a remarkable impact on prediction
    #titanic_df.drop("Cabin",axis=1,inplace=True)
    #test_df.drop("Cabin",axis=1,inplace=True)
    titanic_df = drop_columes(titanic_df, ['Cabin'])
    test_df = drop_columes(test_df, ['Cabin'])
    return titanic_df, test_df