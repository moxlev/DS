import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


sns.set()

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = sns.load_dataset('iris')
X = iris.drop('species', axis=1)
y = iris['species']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)
y_model = GaussianNB().fit(Xtrain, ytrain).predict(Xtest)

print(accuracy_score(ytest, y_model))

#######################################################################
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

X_2D = PCA(n_components=2).fit(X).transform(X)
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]

iris['cluster'] = GaussianMixture(n_components=3, covariance_type='full').fit(X).predict(X)

#plot
sns.lmplot('PCA1', 'PCA2', data=iris, hue='species', col='cluster', fit_reg=False)
plt.show()

#######################################################################
from sklearn.datasets import load_digits
digits = load_digits()

fig, axes = plt.subplots(10, 10, figsize=(8, 8),
        subplot_kw={'xticks': [], 'yticks': []}, gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')
plt.show()

#######################################################################
from sklearn.manifold import Isomap

data_proj = Isomap(n_components=2).fit(digits.data).transform(digits.data)

plt.scatter(data_proj[:, 0], data_proj[:, 1], c=digits.target, edgecolor='none', alpha=0.5,
        cmap=plt.get_cmap('nipy_spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
