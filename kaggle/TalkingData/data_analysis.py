import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

train_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\kaggle\\talkingdata\\train_sample.csv', usecols=train_columns, dtype=dtypes)
target = train_data['is_attributed']
train_data['click_datetime'] = pd.to_datetime(train_data['click_time'])
train_data['dow'] = train_data['click_datetime'].dt.dayofweek
train_data["doy"] = train_data["click_datetime"].dt.dayofyear
train_data.drop(['is_attributed', 'click_time', 'click_datetime'], axis=1, inplace=True)

test_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\kaggle\\talkingdata\\test_sample.csv', usecols=test_columns, dtype=dtypes)
click_id = test_data['click_id']
test_data['click_datetime'] = pd.to_datetime(test_data['click_time'])
test_data['dow'] = test_data['click_datetime'].dt.dayofweek
test_data["doy"] = test_data["click_datetime"].dt.dayofyear
test_data.drop(['click_id', 'click_time', 'click_datetime'], axis=1, inplace=True)

clf = RandomForestClassifier(n_estimators=50)
clf = clf.fit(train_data, target)

features = pd.DataFrame()
features['feature'] = train_data.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
print(features)

output = clf.predict(test_data).astype(int)
df_output = pd.DataFrame()
df_output['click_id'] = click_id
df_output['is_attributed'] = output
df_output[['click_id', 'is_attributed']].to_csv('talking_data.csv',index=False)
