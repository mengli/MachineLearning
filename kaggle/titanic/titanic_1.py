import pandas as pd
import matplotlib.pyplot as plt

width = 0.5
data_train = pd.read_csv("/usr/local/google/home/limeng/Downloads/kaggle/titanic/Train.csv")

print(data_train.info())

plt.subplot2grid((3, 3), (0, 0))
data_train.Survived.value_counts().sort_index(axis=0).plot(kind='bar', color='r')
plt.title('Survived Result')
plt.xticks([0, 1], ['No', 'Yes'])

plt.subplot2grid((3, 3), (0, 1))
data_train.Pclass.value_counts().sort_index(axis=0).plot(kind='bar', color='y')
plt.title('Cabin Class')
plt.xticks([0, 1, 2], ['1st', '2nd', '3rd'])

plt.subplot2grid((3, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age, c='b')
plt.title('Age-Survived')
plt.xticks([0, 1], ['No', 'Yes'])

plt.subplot2grid((3, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde', color='r')
data_train.Age[data_train.Pclass == 2].plot(kind='kde', color='g')
data_train.Age[data_train.Pclass == 3].plot(kind='kde', color='y')
plt.title('Age-Cabin_Class')
plt.legend(['1st', '2nd', '3rd'], loc='best')

plt.subplot2grid((3, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar', color='b')
plt.title('Embarked')

plt.subplot2grid((3, 3), (2, 0))
survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts().sort_index(axis=0)
survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts().sort_index(axis=0)
plt.bar(survived_0.index, survived_0.values, color='r', width=width)
plt.bar(survived_1.index, survived_1.values, bottom=survived_0.values, color='g', width=width)
plt.title('Cabin-Survived')
plt.xticks([1 + width / 2, 2 + width / 2, 3 + width / 2], ['1st', '2nd', '3rd'])
plt.legend(['No', 'Yes'], loc='best')

plt.show()