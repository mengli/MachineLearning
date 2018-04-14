import pandas as pd

train_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\zillow\\train_2017.csv', low_memory=False)
properties_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\zillow\\properties_2017.csv', low_memory=False)

print(train_data.shape)
print(properties_data.shape)
