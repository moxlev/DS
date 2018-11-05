import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.linear_model import LinearRegression

sns.set()

iris = sns.load_dataset('iris')
#sns.pairplot(iris, hue='species')
#plt.show()

X = iris.drop('species', axis=1)
y = iris['species']


print(iris.head())
