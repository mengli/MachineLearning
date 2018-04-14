import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

properties_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\zillow\\properties_2017.csv', low_memory=False)

sns.jointplot(x=properties_data.latitude.values, y=properties_data.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)

plt.show()
