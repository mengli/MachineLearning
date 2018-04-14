import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\zillow\\train_2017.csv', low_memory=False)

plt.figure(figsize=(10, 10))
plt.scatter(range(train_data.shape[0]), train_data.sort_values(by='logerror').logerror)
plt.xlabel('index')
plt.ylabel('logerror')

plt.show()
