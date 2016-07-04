#!/usr/bin/env python3

# The purpose of this script is to stack multiple predictors along with
# original variables using XGBoost

# Each ensemble member must have out of fold predictions on the validation fold
# (Last 10% of the training set), and on the test set. The Validation set
# MAP@3 must be specified in both the validation and test csv files.

# This script ASSUMES that if a model performs well for a certain region in the
# input space on the validation set, it will also perform well for that region
# in the same space on the test set. This is surely an imperfect assumption,
# but we can probably gain some useful insights regardless. It would be better
# to do this over the whole train set using out of fold K-fold predictions,
# however that would increase the computational time required substantially,
# which I cannot afford to do. I hope to avoid overfitting the validation set
# by enforcing appropriate regularization on the second level model
# coefficients.

# This script should be robust to missing rows in the predictions files,
# as long as at least one complete predictions file exists for both sets.
# Missing rows are usually a product of
#   1. The job hasn't yet finished running on the cluster
#   2. Exceeded the allotted computational time (for example: Kernel SVM)
#   3. Hit a bad node on the cluster, so the job crashed.

# Each of the input files must look something like this, but the columns
# can be in any order. `map3` is the **Validation** map3 score for that grid
# cell, even if it is in the test csv file for that ensemble member.

# row_id,pred1,pred2,pred3,map3
# 15097,2403464987,8378301865,5349374802,0.639999985695
# 18192,1006316884,5445221293,1556214113,0.639999985695
# 28420,7065354365,5445221293,2685315047,0.639999985695
# 30451,5445221293,3271917785,8815983898,0.639999985695
# 35843,1006316884,8453876154,1556214113,0.639999985695
# 44149,7014475222,5349374802,2403464987,0.639999985695
# 54664,5357692951,7014475222,5829074077,0.639999985695
# 65875,2403464987,8370753254,9727638738,0.639999985695
# 85142,5445221293,1006316884,3271917785,0.639999985695


import pandas as pd
import numpy as np
import datetime
import time
import os
import sys
import itertools
import xgboost
import pprint

ENSEMBLE_MEMBERS = {
    # 'ET': ('extratrees_vali_2016-06-26-20-34.csv',
    #        'extratrees_test_2016-06-26-20-34.csv'),
    # 'GB': ('gradient_boost_vali_2016-06-26-20-35.csv',
    #        'gradient_boost_test_2016-06-26-20-35.csv'),
    # 'NN1': ('knn_01_vali_2016-06-26-20-39.csv',
    #         'knn_01_test_2016-06-26-20-39.csv'),
    # 'NN5': ('knn_05_vali_2016-06-26-20-39.csv',
    #         'knn_05_test_2016-06-26-20-39.csv'),
    # 'NN9': ('knn_09_vali_2016-06-26-20-39.csv',
    #         'knn_09_test_2016-06-26-20-39.csv'),
    # 'NN25': ('knn_25_vali_2016-06-26-20-39.csv',
    #          'knn_25_test_2016-06-26-20-39.csv'),
    # 'NN37': ('knn_37_vali_2016-06-26-20-39.csv',
    #          'knn_37_test_2016-06-26-20-39.csv'),
    # 'NN51': ('knn_51_vali_2016-06-26-20-39.csv',
    #          'knn_51_test_2016-06-26-20-39.csv'),
    # 'NB': ('naive_bayes_vali_2016-06-26-22-03.csv',
    #        'naive_bayes_test_2016-06-26-22-03.csv'),
    # 'NXGB': ('naive_xgboost_vali_2016-06-26-20-34.csv',
    #          'naive_xgboost_test_2016-06-26-20-34.csv'),
    # 'RF': ('random_forest_vali_2016-06-26-20-34.csv',
    #        'random_forest_test_2016-06-26-20-34.csv'),
    # 'RBFSVM': ('rbf_svm_vali_2016-06-26-20-34.csv',
    #            'rbf_svm_test_2016-06-26-20-34.csv'),

    # these runs are with XGBoost with NN features
    # 'XGNN0': ('run0_vali.csv',
    #           'run0_test.csv'),
    'XGNN1': ('run1_vali.csv',
              'run1_test.csv'),
    'XGNN2': ('run2_vali.csv',
              'run2_test.csv'),
}

xgb_params = {
    'objective': 'multi:softprob',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'gamma': 0.03,
    'subsample': 0.5,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.6,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'max_depth': 4
}

# Set important parameters of the script.
size = 10.
NX = 50
NY = 50
x_step = size/float(NX)
y_step = size/float(NY)
x_cell_margin = x_step*0.1
y_cell_margin = y_step*0.1

RUN_NAME = 'comb3'


def map_k_precision(truthvalues, predictions):
    '''
    This is a faster implementation of MAP@k valid for numpy arrays.
    It is only valid when there is one single truth value.

    m ~ number of observations
    k ~ MAP at k -- in this case k should equal 3

    truthvalues.shape = (m,)
    predictions.shape = (m, k)
    '''
    z = (predictions == truthvalues[:, None]).astype(np.float32)
    weights = 1./(np.arange(predictions.shape[1], dtype=np.float32) + 1.)
    z = z * weights[None, :]
    return float(np.mean(np.sum(z, axis=1)))


def prepare_data_xgboost(dataframe):
    minute = dataframe['time'] % 60
    dataframe['hour'] = dataframe['time']//60
    dataframe['weekday'] = dataframe['hour']//24
    dataframe['month'] = dataframe['weekday']//30
    dataframe['year'] = (dataframe['weekday']//365+1)
    dataframe['hour'] = ((dataframe['hour'] % 24+1)+minute/60.0)
    dataframe['weekday'] = (dataframe['weekday'] % 7+1)
    dataframe['month'] = (dataframe['month'] % 12+1)
    dataframe['log10acc'] = np.log10(dataframe['accuracy'].values)
    dataframe.drop(['time'], axis=1, inplace=True)


def load_data():
    df_train = pd.read_csv('../input/train.csv')
    df_train.sort_values('time', inplace=True)

    df_test = pd.read_csv('../input/test.csv')

    ninety_percent_mark = int(df_train.shape[0]*0.9)
    df_valifold = df_train[ninety_percent_mark:].copy()
    df_trainfold = df_train[:ninety_percent_mark].copy()

    return df_train, df_test, df_trainfold, df_valifold


def load_ensemble(validation):

    idx = 0 if validation else 1

    d = {}
    for k in ENSEMBLE_MEMBERS:
        d[k] = pd.read_csv(ENSEMBLE_MEMBERS[k][idx])

        cols = list(d[k].columns)
        newcols = []
        for x in cols:
            if x == 'row_id':
                newcols.append(x)
            else:
                newcols.append('{}_{}'.format(k, x))
                d[k][x] = d[k][x].astype(str)
        d[k].columns = newcols
    return d


def main():

    print("Loading data")
    (train_data, test_data, trainfold, valifold) = load_data()

    # save some memory
    del train_data, trainfold

    print("Generating time features")
    for x in (test_data, valifold):
        prepare_data_xgboost(x)
    print("Loading model predictions")
    vali_predictions = load_ensemble(validation=True)
    test_predictions = load_ensemble(validation=False)

    print("Performing table joins")
    for k in vali_predictions:
        valifold = pd.merge(valifold, vali_predictions[k], left_on='row_id',
                            how='left', right_on='row_id')
        test_data = pd.merge(test_data, test_predictions[k], left_on='row_id',
                             how='left', right_on='row_id')

    del vali_predictions, test_predictions

    if 'NJOBS' in os.environ:
        njobs = int(os.environ['NJOBS'])
    else:
        njobs = 1
    if 'IJOB' in os.environ:
        ijob = int(os.environ['IJOB'])
    else:
        ijob = 0

    print("Iterating over grid")
    x, y = np.meshgrid(np.arange(NX), np.arange(NY))
    pairs = np.array([x.ravel(), y.ravel()]).T
    pairs = pairs[ijob::njobs]

    for i, j in pairs:
        print((i, j))
        test_filename = '{0}/test_{1:03d}_{2:03d}.csv'.format(RUN_NAME, i, j)

        if os.path.isfile(test_filename):
            continue

        x_min, x_max, y_min, y_max = \
            (i*x_step, (i+1)*x_step, j*y_step, (j+1)*y_step)

        inbin = valifold[(valifold.x >= x_min) &
                         (valifold.x < x_max) &
                         (valifold.y >= y_min) &
                         (valifold.y < y_max)].copy()
        inbin_test = test_data[(test_data.x >= x_min) &
                               (test_data.x < x_max) &
                               (test_data.y >= y_min) &
                               (test_data.y < y_max)].copy()

        Y_valifold = inbin.place_id.values
        test_rowids = inbin_test.row_id.values

        inbin.drop(['place_id', 'row_id',], axis=1, inplace=True)
        inbin_test.drop(['row_id',], axis=1, inplace=True)

        print(inbin)
        print(inbin_test)

        N_vali = inbin.shape[0]
        inbin = pd.concat((inbin, inbin_test), axis=1)
        inbin = pd.get_dummies(inbin)
        print(inbin)
        print(inbin_test)
        sys.exit(9)

        inbin_test = inbin.iloc[N_vali:,:]
        inbin = inbin.iloc[:N_vali, :]

        # assert(np.all(inbin.columns == inbin_test.columns))
        X = inbin.values
        X_test = inbin_test.values



        clf = xgboost.XGBClassifier(**xgb_params)
        clf.fit(X, Y_valifold)
        predict_y = clf.predict_proba(X_test)
        predict_y_test_idx = np.argsort(
            predict_y, axis=1)[:, -3:][:, ::-1]
        predicted_test_place_id = clf.classes_.take(predict_y_test_idx)

        idx = np.argsort(clf.feature_importances_)[::-1]
        pprint.pprint(list(zip(inbin.columns[idx], clf.feature_importances_[idx])))

        p = predicted_test_place_id

        with open(test_filename, 'w') as fh:
            fh.write('row_id,place_id\n')
            for z in zip(test_rowids, p[:, 0], p[:, 1], p[:, 2]):
                fh.write('{},{} {} {}\n'.format(*z))


    print("All done")

if __name__ == '__main__':
    main()
