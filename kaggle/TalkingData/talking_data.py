import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def prepLGB(data, classCol='', IDCol='', fDrop=[]):
    # Drop class column
    if classCol != '':
        labels = data[classCol]
        fDrop = fDrop + [classCol]
    else:
        labels = None

    if IDCol != '':
        IDs = data[IDCol]
    else:
        IDs = None

    if fDrop != []:
        data = data.drop(fDrop, axis=1)

    # Create LGB mats
    lData = lgb.Dataset(data, label=labels, free_raw_data=False,
                        feature_name=list(data.columns),
                        categorical_feature='auto')

    return lData, labels, IDs, data


if __name__ == '__main__':
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

    train_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\kaggle\\talking_data\\train_sample.csv',
                             usecols=train_columns,
                             dtype=dtypes)
    train_data['click_datetime'] = pd.to_datetime(train_data['click_time'])
    train_data['dow'] = train_data['click_datetime'].dt.dayofweek
    train_data["doy"] = train_data["click_datetime"].dt.dayofyear

    test_data = pd.read_csv('C:\\Users\\jowet\\Downloads\\kaggle\\talking_data\\test_sample.csv', usecols=test_columns,
                            dtype=dtypes)
    test_data['click_datetime'] = pd.to_datetime(test_data['click_time'])
    test_data['dow'] = test_data['click_datetime'].dt.dayofweek
    test_data["doy"] = test_data["click_datetime"].dt.dayofyear

    testDataL, _, click_id, testData = prepLGB(test_data,
                                               IDCol='click_id',
                                               fDrop=['click_id', 'click_time', 'click_datetime'])

    params = {'boosting_type': 'gbdt',
              'max_depth': -1,
              'objective': 'binary',
              'nthread': 16,
              'num_leaves': 64,
              'learning_rate': 0.05,
              'max_bin': 512,
              'subsample_for_bin': 200,
              'subsample': 1,
              'subsample_freq': 1,
              'colsample_bytree': 0.8,
              'reg_alpha': 5,
              'reg_lambda': 10,
              'min_split_gain': 0.5,
              'min_child_weight': 1,
              'min_child_samples': 5,
              'scale_pos_weight': 1,
              'num_class': 1,
              'metric': 'binary_error'}

    mdl = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='binary',
                             n_jobs=5,
                             silent=True,
                             max_depth=params['max_depth'],
                             max_bin=params['max_bin'],
                             subsample_for_bin=params['subsample_for_bin'],
                             subsample=params['subsample'],
                             subsample_freq=params['subsample_freq'],
                             min_split_gain=params['min_split_gain'],
                             min_child_weight=params['min_child_weight'],
                             min_child_samples=params['min_child_samples'],
                             scale_pos_weight=params['scale_pos_weight'])

    print('Start training...')
    # Kit k models with early-stopping on different training/validation splits
    k = 12
    predsValid = 0
    predsTrain = 0
    predsTest = 0
    for i in range(0, k):
        print('Fitting model %d' % i)

        trainData, validData = train_test_split(train_data, test_size=0.4, stratify=train_data['is_attributed'])
        trainDataL, trainLabels, trainIDs, trainData = prepLGB(trainData,
                                                               classCol='is_attributed',
                                                               fDrop=['click_time', 'click_datetime'])
        validDataL, validLabels, validIDs, validData = prepLGB(validData,
                                                               classCol='is_attributed',
                                                               fDrop=['click_time', 'click_datetime'])
        # Train
        gbm = lgb.train(params,
                        trainDataL,
                        100000,
                        valid_sets=[trainDataL, validDataL],
                        early_stopping_rounds=50,
                        verbose_eval=4)

        # Predict
        print('Start predicting using model %d' % i)
        predsValid += gbm.predict(validData, num_iteration=gbm.best_iteration) / k
        predsTrain += gbm.predict(trainData, num_iteration=gbm.best_iteration) / k
        predsTest += gbm.predict(testData, num_iteration=gbm.best_iteration) / k

        print('Save model %d' % i)
        # save model k to file
        gbm.save_model('model_%d.txt' % i)

    # Save submission
    sub = pd.DataFrame()
    sub['click_id'] = click_id
    sub['is_attributed'] = np.int32(predsTest > 0.5)
    sub.to_csv('talking_data.csv', index=False)
