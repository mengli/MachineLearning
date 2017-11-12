import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv("/usr/local/google/home/limeng/Downloads/kaggle/titanic/Train.csv")

print(data_train.info())

fig = plt.figure()

ax1 = fig.add_subplot(231)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 1].value_counts().sort_index(axis=0).plot(
    kind='bar', color='red')
ax1.set_xticklabels(['No', 'Yes'])
plt.legend([u"Female High"], loc='best')

ax2 = fig.add_subplot(232, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 2].value_counts().sort_index(axis=0).plot(
    kind='bar', color='pink')
ax2.set_xticklabels(['No', 'Yes'])
plt.legend([u"Female Mid"], loc='best')

ax3 = fig.add_subplot(233, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().sort_index(axis=0).plot(
    kind='bar', color='#FA2479')
ax2.set_xticklabels(['No', 'Yes'])
plt.legend([u"Female Low"], loc='best')

ax4 = fig.add_subplot(234, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 1].value_counts().sort_index(axis=0).plot(kind='bar',
                                                                                                             color='lightblue')
ax4.set_xticklabels(['No', 'Yes'])
plt.legend([u"Male High"], loc='best')

ax5 = fig.add_subplot(235, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 2].value_counts().sort_index(axis=0).plot(kind='bar',
                                                                                                             color='blue')
ax5.set_xticklabels(['No', 'Yes'])
plt.legend([u"Male Mid"], loc='best')

ax6 = fig.add_subplot(236, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().sort_index(axis=0).plot(kind='bar',
                                                                                                             color='darkblue')
ax6.set_xticklabels(['No', 'Yes'])
plt.legend([u"Male Low"], loc='best')

plt.show()
