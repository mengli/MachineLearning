import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\zillow\\train_2017.csv', low_memory=False)

plt.figure(figsize=(10, 10))
up_limit = np.percentile(train_data.logerror, 99)
low_limit = np.percentile(train_data.logerror, 1)
tmp_data = train_data[train_data.logerror < up_limit][train_data.logerror > low_limit]
plt.hist(tmp_data.logerror, bins=50)
plt.xlabel('logerror')

plt.show()
