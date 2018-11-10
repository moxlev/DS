import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs, fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

sns.set()

#Generate isotropic Gaussian blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

#Generate some new data
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
clf = GaussianNB().fit(X, y)
ynew = clf.predict(Xnew)

#Plotting the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.show()

#Computing the probabilistic classification
yprob = clf.predict_proba(Xnew)
print(yprob.shape)
print(yprob[-8:].round(2))

#Classifying text
data = fetch_20newsgroups()
categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']

train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

labels = make_pipeline(TfidfVectorizer(), MultinomialNB()).fit(
        train.data, train.target).predict(test.data)

sns.heatmap(confusion_matrix(test.target, labels), square=True, annot=True,
        fmt='d', cbar=True, xticklabels=train.target_names, yticklabels=test.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
