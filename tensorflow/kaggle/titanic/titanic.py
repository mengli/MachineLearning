import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import linear_model


def set_missing_ages(df):
    train_input = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    has_age = train_input[pd.notnull(train_input.Age)].as_matrix()
    no_age = train_input[pd.isnull(train_input.Age)].as_matrix()

    y = has_age[:, 0]
    X = has_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predicted_age = rfr.predict(no_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = predicted_age

    return df, rfr


def binarize_column(df, column_name):
    df.loc[(df.Cabin.isnull()), column_name] = 'No'
    df.loc[(df.Cabin.notnull()), column_name] = 'Yes'
    return df


data_train = pd.read_csv("/usr/local/google/home/limeng/Downloads/kaggle/titanic/train.csv")

# Add missing data

# Predict missing data
data_train, rfr = set_missing_ages(data_train)
data_train = binarize_column(data_train, 'Cabin')

# One hot encoding columns
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
# Drop the one hot encoded columns
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df[['Age']])
df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)
fare_scale_param = scaler.fit(df[['Fare']])
df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], fare_scale_param)

train_df = df.filter(
    regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

y = train_np[:, 0]
X = train_np[:, 1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

data_test = pd.read_csv("/usr/local/google/home/limeng/Downloads/kaggle/titanic/test.csv")
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = binarize_column(data_test, 'Cabin')
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test,
                     dummies_Cabin,
                     dummies_Embarked,
                     dummies_Sex,
                     dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']], fare_scale_param)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),
                       'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv", index=False)

cof_data = pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})
print(cof_data)
