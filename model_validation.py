import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve, learning_curve, GridSearchCV

sns.set()

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


def make_data(N, err=1.0, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1./(X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y

X, y = make_data(40)

X_test = np.linspace(-0.1, 1.1, 500)[:, np.newaxis]

plt.scatter(X.ravel(), y, c='k')
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree:{0}'.format(degree))

plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best')
plt.show()

#Plotting validation curve
degree = np.arange(0, 21)
X2, y2 = make_data(200)

train_score, val_score = validation_curve(PolynomialRegression(), X, y,
        'polynomialfeatures__degree', degree, cv=7)
train_score2, val_score2 = validation_curve(PolynomialRegression(), X2, y2,
        'polynomialfeatures__degree', degree, cv=7)

plt.plot(degree, np.median(train_score2, 1), c='b', label='training score')
plt.plot(degree, np.median(val_score2, 1), c='r', label='validation score')
plt.plot(degree, np.median(train_score, 1), c='b', alpha=0.3, linestyle='dashed')
plt.plot(degree, np.median(val_score, 1), c='r', alpha=0.3, linestyle='dashed')
plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.show()

#Plotting learning curve
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree), X, y,
            cv=7, train_sizes=np.linspace(0.3, 1, 25))

    ax[i].plot(N, np.mean(train_lc, 1), c='b', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), c='r', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='g', linestyle='dashed')
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree:{0}'.format(degree), size=14)
    ax[i].legend(loc='best')

plt.show()

#GridSearchCV
param_grid = {'polynomialfeatures__degree': np.arange(21),
        'linearregression__fit_intercept': [True, False],
        'linearregression__normalize': [True, False]}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(X, y)
print(grid.best_params_)

y_test = grid.fit(X, y).predict(X_test)
plt.scatter(X.ravel(), y)
plt.plot(X_test.ravel(), y_test)
plt.show()

