import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline


sns.set()

faces = fetch_lfw_people(min_faces_per_person=60)
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone',)
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])

clf = make_pipeline(RandomizedPCA(n_components=150, whiten=True, random_state=42),
        SVC(kernel='rbf', class_weight='balanced'))

Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

param = {'SVC__C': [1, 5, 10, 50], 'SVC__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(clf, param)

print(grid.get_params())
