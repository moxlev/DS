import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


sns.set()

titanic = sns.load_dataset('titanic')
#print(titanic.head())

#groupby
#print(titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack())
#pivot_table
#print(titanic.pivot_table('survived', index='sex', columns='class'))

#multi-level pivot_table
age = pd.cut(titanic['age'], [0, 18, 80])
fare = pd.qcut(titanic['fare'], 2)
#print(titanic.pivot_table('survived', index=['sex', age], columns=[fare, 'class']))
#Additional options(aggfunc, margins)
#print(titanic.pivot_table(index='sex', columns='class', aggfunc={'survived': sum, 'fare': 'mean'}))
#print(titanic.pivot_table('survived', index='sex', columns='class', margins=True))


births = pd.read_csv('./data/births.csv')
births['decade'] = 10 * (births['year'] // 10)
#print(births.pivot_table('births', index='decade', columns='gender', aggfunc='sum'))
#births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()
#plt.ylabel('total births per year')

#robust sigma-clipping
quartiles = np.percentile(births['births'], [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])

#data manipulation
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000 * births['year'] + 100 * births['month'] + births['day'], format='%Y%m%d')
births['dayofweek'] = births.index.dayofweek
print(births.head())

#plot by dayofweek
births.pivot_table('births', index='dayofweek', columns='decade', aggfunc='mean').plot()
plt.gca().set_xticks(range(0, 7))
plt.gca().set_xticklabels(['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day')
plt.show()


births_by_date = births.pivot_table('births', [births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day) for month, day in births_by_date.index]

fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

style = dict(size=10, color='gray')
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)

ax.set(title='USA births by day of year (1969-1988)',
       ylabel='average daily births')

ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))

plt.show()
