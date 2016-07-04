#!/usr/bin/env python3

# Original Source
# https://www.kaggle.com/xmarvin/facebook-v-predicting-check-ins/python-starter-0-55/code
# Substantial modifications by Ravi Shekhar < ravi dot shekhar at gmail
# dot com >

import pandas as pd
import numpy as np
import datetime
import time
import os
import sys
import itertools
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb


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


def main():

    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    start_time = time.time()

    size = 10.0

    x_step = size/100.
    y_step = size/100.
    margin = 0.015

    x_ranges = zip(np.arange(0, size, x_step),
                   np.arange(x_step, size + x_step, x_step))
    y_ranges = zip(np.arange(0, size, y_step),
                   np.arange(y_step, size + y_step, y_step))

    print('Calculate hour, weekday, month and year for train and test')
    minute = train['time'] % 60
    train['hour'] = train['time']//60
    train['weekday'] = train['hour']//24
    train['month'] = train['weekday']//30
    train['year'] = (train['weekday']//365+1)
    train['hour'] = ((train['hour'] % 24+1)+minute/60.0)
    train['weekday'] = (train['weekday'] % 7+1)
    train['month'] = (train['month'] % 12+1)
    train['log10acc'] = np.log10(train['accuracy'].values)

    minute = train['time'] % 60
    test['hour'] = test['time']//60
    test['weekday'] = test['hour']//24
    test['month'] = test['weekday']//30
    test['year'] = (test['weekday']//365+1)
    test['hour'] = ((test['hour'] % 24+1)+minute/60.0)
    test['weekday'] = (test['weekday'] % 7+1)
    test['month'] = (test['month'] % 12+1)
    test['log10acc'] = np.log10(test['accuracy'].values)

    # We're doing two separate things in this script.
    # 1. Fitting on the entire training set and predicting the test set
    # 2. Fitting on the first 90% of the training set (I call this the trainfold),
    #    and predicting the last 10% of the training set (I call this the
    #    validationfold). This will be later used for model stacking.

    # All splitting is done in the time dimension to simulate future predictions
    # the same as the train/test split in the competition.
    train.sort_values('time', inplace=True)

    # For the training on the full data
    X_train_total = train[
        'row_id x y accuracy time hour weekday month year log10acc place_id'
        .split()]
    y_train_total = train.place_id.values

    # For the training on the trainfold and predictions of the validationfold.
    # First 90% as training fold
    X_train_trainfold = train[:int(0.9*train.shape[0])]
    X_train_valifold = train[int(0.9*train.shape[0]):]

    X_test = test[
        'row_id x y accuracy time hour weekday month year log10acc'.split()]

    preds_test = []
    preds_vali = []

    starttime = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    i = 0

    for xlims, ylims in itertools.product(x_ranges, y_ranges):
        x_min, x_max = xlims
        y_min, y_max = ylims

        start_time_cell = time.time()
        x_max = round(x_max, 4)
        x_min = round(x_min, 4)

        y_max = round(y_max, 4)
        y_min = round(y_min, 4)

        if x_max == size:
            x_max = x_max + 0.001

        if y_max == size:
            y_max = y_max + 0.001

        # -------------
        X_train_total_cell = X_train_total[
            (X_train_total['x'] >= x_min - margin) &
            (X_train_total['x'] < x_max + margin) &
            (X_train_total['y'] >= y_min - margin) &
            (X_train_total['y'] < y_max + margin)]

        place_counts = X_train_total_cell.place_id.value_counts()
        mask = (place_counts[X_train_total_cell.place_id.values] >= 4).values
        X_train_total_cell = X_train_total_cell.loc[mask]
        Y_train_total_cell = X_train_total_cell.place_id.values

        # -------------
        X_trainfold_cell = X_train_trainfold[
            (X_train_trainfold['x'] >= x_min - margin) &
            (X_train_trainfold['x'] < x_max + margin) &
            (X_train_trainfold['y'] >= y_min - margin) &
            (X_train_trainfold['y'] < y_max + margin)]

        place_counts = X_trainfold_cell.place_id.value_counts()
        mask = (place_counts[X_trainfold_cell.place_id.values] >= 4).values
        X_trainfold_cell = X_trainfold_cell.loc[mask]
        Y_trainfold_cell = X_trainfold_cell.place_id.values

        # -------------
        X_valifold_cell = X_train_valifold[
            (X_train_valifold['x'] >= x_min) &
            (X_train_valifold['x'] < x_max) &
            (X_train_valifold['y'] >= y_min) &
            (X_train_valifold['y'] < y_max)]
        Y_valifold_cell = X_valifold_cell.place_id.values

        # -------------
        X_test_cell = X_test[
            (X_test['x'] >= x_min) &
            (X_test['x'] < x_max) &
            (X_test['y'] >= y_min) &
            (X_test['y'] < y_max)]

        # Obviously this field doesn't exist :)
        # Y_test_cell = X_test_cell.place_id.values

        # -------------

        X_train_total_cell = X_train_total_cell[
            'x y accuracy log10acc hour weekday month year'.split()]
        X_trainfold_cell = X_trainfold_cell[
            'x y accuracy log10acc hour weekday month year'.split()]
        valifold_row_ids = X_valifold_cell.row_id.values
        X_valifold_cell = X_valifold_cell[
            'x y accuracy log10acc hour weekday month year'.split()]
        test_row_ids = X_test_cell.row_id.values
        X_test_cell = X_test_cell[
            'x y accuracy log10acc hour weekday month year'.split()]

        # -------------

        clf_total = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr',
            probability=True)
        clf_total.fit(X_train_total_cell.values, Y_train_total_cell)
        Y_pred_test_cell = clf_total.predict_proba(X_test_cell.values)
        classes_total = clf_total.classes_

        # this is very memory intensive, so I'm going to delete it for the GC
        del clf_total 
        # -------------

        clf_trainfold = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr',
            probability=True)
        clf_trainfold.fit(X_trainfold_cell.values, Y_trainfold_cell)
        Y_pred_valifold_cell = clf_trainfold.predict_proba(
            X_valifold_cell.values)
        classes_trainfold = clf_trainfold.classes_

        # this is very memory intensive, so I'm going to delete it for the GC
        del clf_trainfold

        # -------------
        Y_pred_test_cell = (np.argsort(Y_pred_test_cell, axis=1)
                            [:, ::-1][:, :3])
        Y_pred_valifold_cell = (np.argsort(Y_pred_valifold_cell, axis=1)
                                [:, ::-1][:, :3])

        test_predicted_placeids = classes_total.take(Y_pred_test_cell)
        valifold_predicted_placeids = classes_trainfold.take(
            Y_pred_valifold_cell)

        # print(test_row_ids)
        # print(test_predicted_placeids)

        # print(valifold_row_ids)
        # print(valifold_predicted_placeids)

        map3 = map_k_precision(Y_valifold_cell, valifold_predicted_placeids)

        N_test = test_row_ids.shape[0]
        N_vali = valifold_predicted_placeids.shape[0]

        preds_test.append(pd.DataFrame({'row_id': test_row_ids,
                      'pred1': test_predicted_placeids[:, 0],
                      'pred2': test_predicted_placeids[:, 1],
                      'pred3': test_predicted_placeids[:, 2],
                      'map3' : np.ones(N_test, dtype=np.float32)*map3
                      }))
        preds_vali.append(pd.DataFrame({'row_id': valifold_row_ids,
                      'pred1': valifold_predicted_placeids[:, 0],
                      'pred2': valifold_predicted_placeids[:, 1],
                      'pred3': valifold_predicted_placeids[:, 2],
                      'map3' : np.ones(N_vali, dtype=np.float32)*map3
                      }))

        print("X: [{:.4f},{:.4f}) Y: [{:.4f},{:.4f}) MAP3: {:.4f}"
            .format(x_min, x_max, y_min, y_max, map3))

        if i % 100 == 0:
            print("Updating rbf_svm_test_{}.csv".format(starttime))
            pd.concat(preds_test).to_csv(
                'rbf_svm_test_{}.csv'.format(starttime), index=False)
            pd.concat(preds_vali).to_csv(
                'rbf_svm_vali_{}.csv'.format(starttime), index=False)

        i += 1
    print("Writing out final rbf_svm_test_{}.csv".format(starttime))
    pd.concat(preds_test).to_csv('rbf_svm_test_{}.csv'.format(starttime), index=False)
    pd.concat(preds_vali).to_csv('rbf_svm_vali_{}.csv'.format(starttime), index=False)

    print("All done")

if __name__ == '__main__':
    main()
