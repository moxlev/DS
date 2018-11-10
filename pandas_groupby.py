import numpy as np
import pandas as pd
import seaborn as sns


sns.set()

planets = sns.load_dataset('planets')
decade = (planets['year'] // 10) * 10
planets['decade'] = decade.astype(str) + 's'

print(planets.head())
print(planets.groupby(['method', 'decade'])['number'].sum().unstack().fillna(0))
