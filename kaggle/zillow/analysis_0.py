import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_data = pd.read_csv('/usr/local/google/home/limeng/Downloads/kaggle/zillow/train_2016_v2.csv')
properties_data = pd.read_csv('/usr/local/google/home/limeng/Downloads/kaggle/zillow/properties_2016.csv')

plt.figure(figsize=(10, 10))
plt.scatter(range(train_data.shape[0]), train_data.sort_values(by='logerror').logerror)
plt.xlabel('index')
plt.ylabel('logerror')

plt.figure(figsize=(10, 10))
up_limit = np.percentile(train_data.logerror, 99)
low_limit = np.percentile(train_data.logerror, 1)
tmp_data = train_data[train_data.logerror < up_limit][train_data.logerror > low_limit]
plt.hist(tmp_data.logerror, bins=50)
plt.xlabel('logerror')

plt.figure(figsize=(10, 10))
datetime_data = pd.to_datetime(train_data.transactiondate)
datetime_data.dt.month.value_counts().sort_index(axis=0).plot(kind='bar')
plt.xlabel('month')

missing_df = properties_data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df[missing_df.missing_count > 0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df.plot(kind='barh')
plt.yticks(range(missing_df.shape[0]), missing_df.column_name.values)

sns.jointplot(x=properties_data.latitude.values, y=properties_data.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)

train_df = pd.merge(train_data, properties_data, on='parcelid', how='left')

merged_missing_df = train_df.isnull().sum(axis=0).reset_index()
merged_missing_df.columns = ['column_name', 'missing_count']
limit = np.percentile(merged_missing_df.missing_count, 90)
print(merged_missing_df[merged_missing_df.missing_count >= limit])

# Let us just impute the missing values with mean values to compute correlation coefficients #
mean_values = train_df.mean(axis=0)
train_df_new = train_df.fillna(mean_values, inplace=True)

# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype=='float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")

corr_df_sel = corr_df.loc[(corr_df.corr_values>0.02) | (corr_df.corr_values<-0.01)]


plt.show()
