import pandas as pd
import numpy as np
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
#from sklearn import linear_model


def set_missing_ages(df, rfr=None):
    # Age
    input_data = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    has_age = input_data[pd.notnull(input_data.Age)].as_matrix()
    no_age = input_data[pd.isnull(input_data.Age)].as_matrix()

    y = has_age[:, 0]
    X = has_age[:, 1:]

    if rfr is None:
        rfr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
        rfr.fit(X, y)

    predicted_age = rfr.predict(no_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = predicted_age

    return df, rfr


def set_missing_data(df):
    # Replace missing values with "U0"
    df.loc[df.Cabin.isnull(), 'Cabin'] = 'U0'

    # Take the median of all non-null Fares and use that for all missing values
    df.loc[np.isnan(df['Fare']), 'Fare'] = df['Fare'].median()

    # Replace missing values with most common port
    df.loc[df.Embarked.isnull(), 'Embarked'] = df.Embarked.dropna().mode().values

    return df


def derive_data(df, test=False, drops=None):
    df['Parch'] = df['Parch'] + 1
    df['SibSp'] = df['SibSp'] + 1
    df['Deck'] = df['Deck'] + 1
    df['Fare'] = df['Fare'] + 0.0001
    numerics = df.loc[:, ['Age', 'Fare', 'Pclass', 'Parch', 'SibSp', 'Deck']]

    # for each pair of variables, determine which mathmatical operators to use based on redundancy
    for i in range(0, numerics.columns.size-1):
        for j in range(0, numerics.columns.size-1):
            col1 = str(numerics.columns.values[i])
            col2 = str(numerics.columns.values[j])
            # multiply fields together (we allow values to be squared)
            if i <= j:
                name = col1 + "*" + col2
                df = pd.concat(
                    [df, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)
            # add fields together
            if i < j:
                name = col1 + "+" + col2
                df = pd.concat(
                    [df, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis=1)
            # divide and subtract fields from each other
            if not i == j:
                name = col1 + "/" + col2
                df = pd.concat(
                    [df, pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j], name=name)], axis=1)
                name = col1 + "-" + col2
                df = pd.concat(
                    [df, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)

    if test:
        # calculate the correlation matrix (ignore survived and passenger id fields)
        df_corr = df.drop(['PassengerId'], axis=1).corr(method='spearman')
    else:
        df_corr = df.drop(['Survived', 'PassengerId'], axis=1).corr(method='spearman')

    # create a mask to ignore self-
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr

    if drops is None:
        drops = []
        # loop through each variable
        for col in df_corr.columns.values:
            # if we've already determined to drop the current variable, continue
            if np.in1d([col], drops):
                continue

            # find all the variables that are highly correlated with the current variable
            # and add them to the drop list
            corr = df_corr[abs(df_corr[col]) > 0.85].index
            drops = np.union1d(drops, corr)
    print "Dropping", drops.shape[0], "highly correlated features...n", drops
    df.drop(drops, axis=1, inplace=True)
    return df, drops


def categorize_data(df):
    df['Sex'] = pd.factorize(df['Sex'])[0]
    df['Embarked'] = pd.factorize(df['Embarked'])[0]
    df['Deck'] = df.Cabin.str.extract("([a-zA-Z]+)", expand=False)
    df['Deck'] = pd.factorize(df['Deck'])[0]
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df.loc[df.Title == 'Jonkheer', 'Title'] = 'Master'
    df.loc[df.Title.isin(['Ms','Mlle']), 'Title'] = 'Miss'
    df.loc[df.Title == 'Mme', 'Title'] = 'Mrs'
    df.loc[df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir']), 'Title'] = 'Sir'
    df.loc[df.Title.isin(['Dona', 'Lady', 'the Countess', 'Countess']), 'Title'] = 'Lady'
    df['Title'] = pd.factorize(df['Title'])[0]

    df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

    return df


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Train set count")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(train_sizes,
                         train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes,
                         test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Train score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Val score")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()


data_train = pd.read_csv("/usr/local/google/home/limeng/Downloads/kaggle/titanic/train.csv")
# Predict missing data
data_train = set_missing_data(data_train)
data_train, rfr = set_missing_ages(data_train)
data_train = categorize_data(data_train)
data_train, drops = derive_data(data_train)

# Split data set
split_train, split_cv = cross_validation.train_test_split(data_train, test_size=0.3)
#clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf = RandomForestClassifier(oob_score=True, n_estimators=200)
#clf.fit(split_train.as_matrix()[:,2:], split_train.as_matrix()[:,1])
clf.fit(data_train.as_matrix()[:,2:], data_train.as_matrix()[:,1])

features_list = data_train.columns.values[2::]
feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)[::-1]
plt.barh(range(len(features_list)), np.sort(feature_importance)[::-1])
plt.yticks(range(len(features_list)), features_list[sorted_idx])
plt.show()

# Predict on val set
print(clf.score(split_cv.as_matrix()[:,2:], split_cv.as_matrix()[:,1]))

plot_learning_curve(clf, "Learning curve",
                    data_train.as_matrix()[:,2:],
                    data_train.as_matrix()[:,1])

# Test

data_test = pd.read_csv("/usr/local/google/home/limeng/Downloads/kaggle/titanic/test.csv")
data_test = set_missing_data(data_test)
data_test, rfr = set_missing_ages(data_test, rfr=rfr)
data_test = categorize_data(data_test)
data_test, _ = derive_data(data_test, test=True, drops=drops)

#print(data_test.head())
predictions = clf.predict(data_test.as_matrix()[:,1:])
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),
                       'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv", index=False)

