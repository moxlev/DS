import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set()

digits = load_digits()
X = digits.data
y = digits.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
y_pred = GaussianNB().fit(Xtrain, ytrain).predict(Xtest)

print(accuracy_score(ytest, y_pred))

sns.heatmap(confusion_matrix(ytest, y_pred), square=True, annot=True, cbar=True)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

##################################################################
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

X = load_iris().data
y = load_iris().target
clf = KNeighborsClassifier(n_neighbors=1)
print(cross_val_score(clf, X, y, cv=5))
