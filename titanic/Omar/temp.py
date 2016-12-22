import pandas as pd

data = pd.read_csv('../test.csv')
data = data.head(10)
print(data)

data = data.drop(['Fare'], axis=1)
print(data)

data = data.drop(0, axis=0)
print(data)


data = data.drop(['Cabin'], axis=1)
print(data)

data.drop(['Parch', 'SibSp'], axis=1, inplace=True)
print(data)