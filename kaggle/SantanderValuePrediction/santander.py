# -*- coding: utf-8 -*-

from xgboost import XGBRegressor
import pandas as pd

train = pd.read_csv("C:\\Users\\jowet\\Downloads\\Santander\\train.csv")
test = pd.read_csv("C:\\Users\\jowet\\Downloads\\Santander\\test.csv")

train.drop('ID', axis=1, inplace=True)

y_train = train.pop('target')
pred_index = test.pop('ID')

reg = XGBRegressor()
reg.fit(train, y_train)
y_pred = reg.predict(test)

submit = pd.DataFrame()
submit['ID'] = pred_index
submit['target'] = y_pred
submit.to_csv('my_XGB_prediction.csv', index=False)
