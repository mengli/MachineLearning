import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

data_train = pd.read_csv("/usr/local/google/home/limeng/Downloads/kaggle/titanic/Train.csv")

print(data_train.info())

fig = plt.figure()

ax1=fig.add_subplot(231)
has_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts().sort_index(axis=0)
no_cabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts().sort_index(axis=0)
df = DataFrame({'Has':has_cabin, 'No':no_cabin})
df.plot(kind='bar', stacked=True)
plt.xticks([0, 1], ['No', 'Yes'])
plt.legend([u"Has Cabin", "No Cabin"], loc='best')

plt.show()