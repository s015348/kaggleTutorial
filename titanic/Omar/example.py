# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic



import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from drop import drop_useless
from embarked import show_embarked
from fare import show_fare
from age import show_age
from family import show_family
from sex import show_sex, create_person
from pclass import show_pclass, create_sub_pclass
from utility import load_data, preview_data, set_print_style
from mock.mock import inplace

# load data
titanic_df, test_df = load_data()
preview_data([titanic_df, test_df])

# prepare data by split male & female and sub classes
titanic_df, test_df = create_person(titanic_df, test_df)
titanic_df, test_df = create_sub_pclass(titanic_df, test_df)

# plot data
set_print_style()
show_embarked(titanic_df)
show_fare(titanic_df, test_df)
show_age(titanic_df, test_df)
show_family(titanic_df, test_df)
show_sex(titanic_df, test_df)
show_pclass(titanic_df, test_df)
sns.plt.show()

# drop useless columns, or take useful columns for training
titanic_df, test_df = drop_useless(titanic_df, test_df)

# define training and testing sets

X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()

# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

score = logreg.score(X_train, Y_train)
print(score)


# Support Vector Machines

# svc = SVC()

# svc.fit(X_train, Y_train)

# Y_pred = svc.predict(X_test)

# svc.score(X_train, Y_train)



# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

score = random_forest.score(X_train, Y_train)
print(score)



# knn = KNeighborsClassifier(n_neighbors = 3)

# knn.fit(X_train, Y_train)

# Y_pred = knn.predict(X_test)

# knn.score(X_train, Y_train)


# Gaussian Naive Bayes

# gaussian = GaussianNB()

# gaussian.fit(X_train, Y_train)

# Y_pred = gaussian.predict(X_test)

# gaussian.score(X_train, Y_train)



# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
print(coeff_df)