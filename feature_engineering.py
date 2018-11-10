import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Imputer

sns.set()

#DictVectorizer
from sklearn.feature_extraction import DictVectorizer
data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

vec = DictVectorizer(sparse=False, dtype=int)
print(vec.fit_transform(data))
print(vec.get_feature_names())

sample = ['problem of evil',
          'evil queen',
          'horizon problem']

vec = CountVectorizer()
print(pd.DataFrame(vec.fit_transform(sample).toarray(), columns=vec.get_feature_names()))

vec = TfidfVectorizer()
print(pd.DataFrame(vec.fit_transform(sample).toarray(), columns=vec.get_feature_names()))

X = np.array([[ np.nan, 0,   3  ],
              [ 3,   7,   9  ],
              [ 3,   5,   2  ],
              [ 4,   np.nan, 6  ],
              [ 8,   8,   1  ]])

print(X)
print(Imputer(strategy='mean').fit_transform(X))
