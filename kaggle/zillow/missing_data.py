import pandas as pd
import matplotlib.pyplot as plt

properties_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\zillow\\properties_2017.csv', low_memory=False)

missing_df = properties_data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df[missing_df.missing_count > 0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df.plot(kind='barh')
plt.yticks(range(missing_df.shape[0]), missing_df.column_name.values)

plt.show()
