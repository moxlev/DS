import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()
#Extract the data we're interested in
cities = pd.read_csv('./data/california_cities.csv')
lat, lon = cities['latd'], cities['longd']
pop, area = cities['population_total'], cities['area_total_km2']

#Scatter the points
plt.scatter(lon, lat, label=None, c=np.log10(pop), cmap='viridis',
        s=area, linewidth=0, alpha=0.5)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

#Create a legend
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area, label=str(area)+'km$^2$')

plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
plt.title('California Cities: Area and Population')

plt.show()
