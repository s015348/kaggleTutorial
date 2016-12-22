# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import seaborn as sns

from algorithms import logistic_regression, naive_bayes, knn, random_forests, svm
from drop import drop_useless
from embarked import show_embarked
from fare import show_fare
from age import show_age
from family import show_family
from sex import show_sex, create_person
from pclass import show_pclass, create_sub_pclass
from utility import load_data, preview_data, set_print_style, get_features_correlation
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


score, logreg = logistic_regression(X_train, Y_train, X_test)
coeff_df = get_features_correlation(titanic_df, logreg)
print("Correlation among features:")
print coeff_df
print("Logistic regression score: %f" %score)
score = svm(X_train, Y_train, X_test)
print("SVM score: %f" %score)
score = random_forests(X_train, Y_train, X_test)
print("Random forest score: %f" %score)
score = knn(X_train, Y_train, X_test)
print("KNN score: %f" %score)
score = naive_bayes(X_train, Y_train, X_test)
print("Naive Bayes score: %f" %score)