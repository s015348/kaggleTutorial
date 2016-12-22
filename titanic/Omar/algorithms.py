# From: https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

import pandas as pd

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def random_forests(X_train, Y_train, X_test):
    # Random Forests
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    score = random_forest.score(X_train, Y_train)
    return score


def svm(X_train, Y_train, X_test):
    # Support Vector Machines
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    score = svc.score(X_train, Y_train)
    return score


def knn(X_train, Y_train, X_test):
    # k-nearest neighbors algorithm
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    score = knn.score(X_train, Y_train)
    return score


def naive_bayes(X_train, Y_train, X_test):
    # Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    score = gaussian.score(X_train, Y_train)
    return score


def logistic_regression(X_train, Y_train, X_test):
    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    score = logreg.score(X_train, Y_train)
    return score, logreg