import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\zillow\\train_2017.csv', low_memory=False)

plt.figure(figsize=(10, 10))
datetime_data = pd.to_datetime(train_data.transactiondate)
datetime_data.dt.month.value_counts().sort_index(axis=0).plot(kind='bar')
plt.xlabel('month')

plt.show()
