#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import os.path
import datetime
import time
import sys
import functools
import itertools
import copy
import sklearn.preprocessing
import sklearn.neighbors
from bayes_opt import BayesianOptimization
import argparse
import numba
import xgboost
from matplotlib import pyplot as plt
from lru import LRU
import json

# This is a relic from an earlier version of the script where I was using the
# threading module to parallelize. This would save data with kNN features 
# between threads and multiple calls of the Bayesian Optimizer.
data_cache = LRU(2) 


size = 10.
NX = 100
NY = 100
x_step = size/float(NX)
y_step = size/float(NY)
x_cell_margin = x_step*0.1
y_cell_margin = y_step*0.1

# This is from Run A3 in my notebook.
# With last 10% in time as validation set,
# this produces a nearest neighbor MAP3
# of 0.54844. NX = NY = 10. 
# the leaderboard score is 0.57920
#
# check knn_bayes_4.log
knn_opt_params = {
    'th': 5.4848,
    'w_x': 487.2522,
    'w_y': 1001.9209,
    'w_hour': 3.6170,
    'w_log10acc': 14.3511,
    'w_weekday': 3.1571,
    'w_month': 2,
    'w_year': 8.8743,
    'n_neighbors': 29,
}



def load_data(validation=False):
    df_train = pd.read_csv('../input/train.csv',
                           usecols=[
                               'row_id', 'x', 'y', 'accuracy', 'time', 'place_id'],
                           index_col=0)
    df_train.sort_values('time', inplace=True)

    if validation:
        ninety_percent_mark = int(df_train.shape[0]*0.9)
        df_test = df_train[ninety_percent_mark:]
        df_train = df_train[:ninety_percent_mark]
    else:
        df_test = pd.read_csv('../input/test.csv',
                              usecols=['row_id', 'x', 'y', 'accuracy', 'time'],
                              index_col=0)
    return df_train, df_test



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



# Call the Numba JIT compiler ... write the loops manually like I would in C
@numba.autojit
def generate_dist_features(encoded_placeids, distances, n_categories):

    assert(encoded_placeids.shape == distances.shape)
    assert(encoded_placeids.ndim == 2)

    sh = encoded_placeids.shape

    o = np.zeros((sh[0], n_categories, ), dtype=np.float32)

    for i in range(sh[0]):
        for j in range(sh[1]):
            o[i, encoded_placeids[i, j]] += distances[i, j]

    return o


def add_knn_features(df_train, df_test, knn_params):

    weighted_df_train = df_train.copy()
    weighted_df_test = df_test.copy()

    # divide the train set into two pieces,
    # 1. with place_ids that occur more than 'th' times
    # 2. less than 'th' times
    # and there's the test set, for a total of three pieces

    place_counts = weighted_df_train.place_id.value_counts()
    mask = (place_counts[weighted_df_train.place_id.values]
            >= knn_params['th']).values

    # weighted_df_train_masked contains less than 'th' times rows
    weighted_df_train_masked = weighted_df_train.loc[~mask]

    # more than 'th' times rows
    weighted_df_train = weighted_df_train.loc[mask]

    # save these for later.
    weighted_df_train_place_ids = weighted_df_train.place_id.values

    # drop some columns, 'place_id' can occur in test set under
    # certain validation or cross-validation conditions.
    weighted_df_train.drop(['accuracy', 'place_id'], axis=1, inplace=True)
    weighted_df_train_masked.drop(
        ['accuracy', 'place_id'], axis=1, inplace=True)
    weighted_df_test.drop('accuracy', axis=1, inplace=True)
    if 'place_id' in weighted_df_test.columns:
        weighted_df_test.drop('place_id', axis=1, inplace=True)

    # -------
    # This block reweights all three pieces
    weighted_df_train.loc[:, 'x'] *= knn_params['w_x']
    weighted_df_train.loc[:, 'y'] *= knn_params['w_y']
    weighted_df_train.loc[:, 'hour'] *= knn_params['w_hour']
    weighted_df_train.loc[:, 'weekday'] *= knn_params['w_weekday']
    weighted_df_train.loc[:, 'month'] *= knn_params['w_month']
    weighted_df_train.loc[:, 'year'] *= knn_params['w_year']
    weighted_df_train.loc[:, 'log10acc'] *= knn_params['w_log10acc']

    weighted_df_train_masked.loc[:, 'x'] *= knn_params['w_x']
    weighted_df_train_masked.loc[:, 'y'] *= knn_params['w_y']
    weighted_df_train_masked.loc[:, 'hour'] *= knn_params['w_hour']
    weighted_df_train_masked.loc[:, 'weekday'] *= knn_params['w_weekday']
    weighted_df_train_masked.loc[:, 'month'] *= knn_params['w_month']
    weighted_df_train_masked.loc[:, 'year'] *= knn_params['w_year']
    weighted_df_train_masked.loc[:, 'log10acc'] *= knn_params['w_log10acc']

    weighted_df_test.loc[:, 'x'] *= knn_params['w_x']
    weighted_df_test.loc[:, 'y'] *= knn_params['w_y']
    weighted_df_test.loc[:, 'hour'] *= knn_params['w_hour']
    weighted_df_test.loc[:, 'weekday'] *= knn_params['w_weekday']
    weighted_df_test.loc[:, 'month'] *= knn_params['w_month']
    weighted_df_test.loc[:, 'year'] *= knn_params['w_year']
    weighted_df_test.loc[:, 'log10acc'] *= knn_params['w_log10acc']
    # ------- End reweighting block -----

    # --- Find distances block -----

    nn_ = sklearn.neighbors.NearestNeighbors(metric='manhattan')

    nn_.fit(weighted_df_train.as_matrix())
    dists_train, indices_train = nn_.kneighbors(
        weighted_df_train[
            'x y hour weekday month year log10acc'.split()].values,
        n_neighbors=knn_params['n_neighbors']+1)
    dists_train_masked, indices_train_masked = nn_.kneighbors(
        weighted_df_train_masked[
            'x y hour weekday month year log10acc'.split()].values,
        n_neighbors=knn_params['n_neighbors'])
    dists_test, indices_test = nn_.kneighbors(
        weighted_df_test[
            'x y hour weekday month year log10acc'.split()].values,
        n_neighbors=knn_params['n_neighbors'])

    # --- Find distances complete -----

    # For a sensible feature that behaves identically in the training and testing
    # datasets, we have to eliminate the "memorization" that occurs with the
    # nearest neighbor algorithm. This means, eliminate points with identically
    # zero distance, as they are just the training example.

    # eliminate the self-distance in the training neighbors dataset,
    # and take reciprocal distance for all three pieces
    dists_train = 1. / dists_train[:, 1:]
    indices_train = indices_train[:, 1:]
    dists_train_masked = 1. / dists_train_masked
    dists_test = 1. / dists_test

    # Find the predicted place_id for all three pieces
    place_ids_knn_train = np.take(weighted_df_train_place_ids, indices_train)
    place_ids_knn_train_masked = np.take(
        weighted_df_train_place_ids, indices_train_masked)
    place_ids_knn_test = np.take(weighted_df_train_place_ids, indices_test)

    # By using this method, the labels in LabelEncoder are sorted by place_id
    unique_place_ids = np.unique(place_ids_knn_train.ravel())
    unique_place_ids.sort()
    le = sklearn.preprocessing.LabelEncoder()
    le.fit_transform(unique_place_ids)

    # encode place_id for all three pieces.
    place_ids_knn_train_enc = le.transform(place_ids_knn_train)
    place_ids_knn_train_masked_enc = le.transform(place_ids_knn_train_masked)
    place_ids_knn_test_enc = le.transform(place_ids_knn_test)

    # Take the encoded labels and build a weighted matrix of place_id
    # weights based on it. Shape is (m, p) where m is number of
    # observations in that piece (train, train_masked, test), and
    # p is the number of place_ids in the training dataset. By
    # construction, the encoded place_ids go from 0 to p-1, allowing
    # easy coding into a matrix.
    train_dist_feat = generate_dist_features(place_ids_knn_train_enc,
                                             dists_train,
                                             unique_place_ids.shape[0])
    train_dist_masked_feat = generate_dist_features(place_ids_knn_train_masked_enc,
                                                    dists_train_masked,
                                                    unique_place_ids.shape[0])
    test_dist_feat = generate_dist_features(place_ids_knn_test_enc,
                                            dists_test,
                                            unique_place_ids.shape[0])

    # Now we turn this into a dataframe.
    column_labels = ['d{}'.format(x) for x in unique_place_ids]

    new_df_train = pd.DataFrame(train_dist_feat, index=weighted_df_train.index.values,
                                columns=column_labels)
    new_df_train_masked = pd.DataFrame(train_dist_masked_feat,
                                       index=weighted_df_train_masked.index.values,
                                       columns=column_labels)
    new_df_test = pd.DataFrame(test_dist_feat,
                               index=weighted_df_test.index.values,
                               columns=column_labels)

    new_df_train = new_df_train.append(new_df_train_masked)

    # do an inner join on row_id
    new_df_train = pd.merge(
        df_train, new_df_train, left_index=True, right_index=True)
    new_df_test = pd.merge(
        df_test, new_df_test, left_index=True, right_index=True)

    return new_df_train, new_df_test


def xgboost_predict(train, test, xgboost_params={}, map_at_k_K=3):

    y_train = train.place_id.values
    x_train = train.drop('place_id', axis=1).values

    if 'place_id' in test.columns:
        x_test = test.drop('place_id', axis=1).values
    else:
        x_test = test.values

    clf = xgboost.XGBClassifier(objective='multi:softprob', seed=42,
                                nthread=-1,
                                **xgboost_params)
    clf.fit(x_train, y_train)
    predict_y_test = clf.predict_proba(x_test)

    # heapq top k algorithm is benchmarked to be slower in this case
    predict_y_test_idx = np.argsort(
        predict_y_test, axis=1)[:, -map_at_k_K:][:, ::-1]
    predicted_test_place_id = clf.classes_.take(predict_y_test_idx)

    return test.index.values, predicted_test_place_id


def process_one_cell(df_train, df_test, x_min, x_max, y_min, y_max,
                     # these parameters are taken from xgboost defaults
                     # http://xgboost.readthedocs.io/en/latest//parameter.html
                     # cut_threshold=1,
                     # n_estimators=100,
                     # learning_rate=0.3,
                     # gamma=0.0,
                     # subsample=1.0,
                     # colsample_bytree=1.0,
                     # colsample_bylevel=1.0,
                     # reg_alpha=0.0,
                     # reg_lambda=1.0,
                     # min_child_weight=1,
                     # max_depth=5

                     # first attempt at optimum parameters (run1)
                     # cut_threshold = 1,
                     # n_estimators=100,
                     # learning_rate=0.075,
                     # gamma=0.05,
                     # subsample=0.6,
                     # colsample_bytree=0.4,
                     # colsample_bylevel=0.3,
                     # reg_alpha=0.04,
                     # reg_lambda=1.2,
                     # min_child_weight=0.5,
                     # max_depth=4

                     # second attempt at optimum parameters (run2)
                     cut_threshold=1,
                     n_estimators=120,
                     learning_rate=0.05,
                     gamma=0.03,
                     subsample=0.8,
                     colsample_bytree=0.6,
                     colsample_bylevel=0.7,
                     reg_alpha=0.05,
                     reg_lambda=0.7,
                     min_child_weight=0.7,
                     max_depth=5
                     ):

    if x_max == size:
        x_max += 1.0e-5
    if y_max == size:
        y_max += 1.0e-5

    if (x_min, x_max, y_min, y_max, cut_threshold) in data_cache.keys():
        train_in_cell, test_in_cell = data_cache[(x_min, x_max, y_min, y_max, cut_threshold)]
    else:
        train_in_cell = df_train[(df_train.x >= x_min - x_cell_margin) &
                                 (df_train.x < x_max + x_cell_margin) &
                                 (df_train.y >= y_min - y_cell_margin) &
                                 (df_train.y < y_max + y_cell_margin)
                                 ]

        test_in_cell = df_test[(df_test.x >= x_min) &
                               (df_test.x < x_max) &
                               (df_test.y >= y_min) &
                               (df_test.y < y_max)
                               ]

        train_in_cell, test_in_cell = add_knn_features(
            train_in_cell, test_in_cell, knn_opt_params)

        place_counts = train_in_cell.place_id.value_counts()
        mask = (place_counts[train_in_cell.place_id.values]
                >= cut_threshold).values

        # more than 'th' times rows
        train_in_cell = train_in_cell.loc[mask]
        data_cache[(x_min, x_max, y_min, y_max, cut_threshold)] = train_in_cell, test_in_cell

    validation_mode = 'place_id' in test_in_cell.columns

    xgboost_params = {
        'n_estimators': int(round(n_estimators)),
        'learning_rate': learning_rate,
        'gamma': gamma,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'min_child_weight': min_child_weight,
        'max_depth': int(round(max_depth)),
    }

    if validation_mode:
        row_id, pred_place_id = xgboost_predict(train_in_cell, test_in_cell,
                                                xgboost_params=xgboost_params)
        truthvalues = test_in_cell.ix[row_id].place_id.values

        map3 = map_k_precision(truthvalues, pred_place_id)
        return map3
    else:
        row_id, pred_place_id = xgboost_predict(train_in_cell, test_in_cell,
                                                xgboost_params=xgboost_params)
        df = pd.DataFrame(pred_place_id, columns='pred1 pred2 pred3'.split(),
                          index=row_id)
        df.index.rename('row_id', inplace=True)
        return df


def iterate_over_grid(train_set, test_set, njobs, ijob, validation=False):

    dfs = []

    # assumes NX = NY
    x, y = np.meshgrid(np.arange(NX), np.arange(NY))
    pairs = np.array([x.ravel(), y.ravel()]).T
    pairs = pairs[ijob::njobs]

    if validation:
        np.random.shuffle(pairs)

    for i, j in pairs:
        print((i, j))

        (x_min, x_max, y_min, y_max) = \
            (i*x_step, (i+1)*x_step, j*y_step, (j+1)*y_step)


        if validation:
            if os.path.isfile('opt_params/{0}_{1}.{2}_{3}.json'.format(NX, NY, i, j)):
                print("Continuing")
                continue

            f = functools.partial(process_one_cell, df_train=train_set,
                                  df_test=test_set, x_min=x_min, x_max=x_max,
                                  y_min=y_min, y_max=y_max)

            bo = BayesianOptimization(f=f,
                                      pbounds={  
                                          # 'n_estimators': (30, 300),
                                          'cut_threshold': (0.9, 10), 
                                          'learning_rate': (0.001, 0.15),
                                          'gamma': (0.03, 0.18),
                                          'subsample': (0.2, 0.7),
                                          'colsample_bytree': (0.3, 0.85),
                                          'colsample_bylevel': (0.3, 0.85),
                                          'reg_alpha': (0.03, 0.07),
                                          'reg_lambda': (0.8, 2.0),
                                          'min_child_weight': (0.2, 0.85),
                                          'max_depth': (3, 6),
                                      },
                                      verbose=True
                                      )

            initial_points = {  
                # 'n_estimators': (40, 80, 140),
                'cut_threshold': (1, 3, 5),
                'learning_rate': (0.1, 0.07, 0.075),
                'gamma': (0.04, 0.12, 0.05),
                'subsample': (0.5, 0.8, 0.7),
                'colsample_bytree': (0.5, 0.7, 0.4),
                'colsample_bylevel': (0.5, 0.8, 0.3),
                'reg_alpha': (0.03, 0.04, 0.06),
                'reg_lambda': (0.8, 1.1, 1.3),
                'min_child_weight': (0.3, 0.5, 0.45),
                'max_depth': (4, 5, 4),
            }

            bo.explore(initial_points)
            bo.maximize(init_points=2, n_iter=59, acq="ei", xi=0.1)
            bo.maximize(n_iter=64, acq="ei", xi=0.0)

            with open('opt_params/{0}_{1}.{2}_{3}.json'.format(NX, NY, i, j), 'w') as fh:
                fh.write(json.dumps(bo.res, sort_keys=True,
                                    indent=4, separators=(',', ': ')))
        else:
            if os.path.isfile('run2/{0:03d}_{1:03d}.csv'.format(i, j)):
                print("Continuing")
                continue

            dfs.append(
                process_one_cell(train_set, test_set, x_min, x_max, y_min, y_max))
            dfs[-1].to_csv('run2/{0:03d}_{1:03d}.csv'.format(i, j))
            print("processed 'run2/{0:03d}_{1:03d}.csv'".format(i, j))

    return


def main(validation=True):
    train_data, test_data = load_data(validation=validation)
    prepare_data_xgboost(train_data)
    prepare_data_xgboost(test_data)

    if 'NJOBS' in os.environ:
        njobs = int(os.environ['NJOBS'])
    else:
        njobs = 1
    if 'IJOB' in os.environ:
        ijob = int(os.environ['IJOB'])
    else:
        ijob = 0

    iterate_over_grid(train_data, test_data, njobs, ijob,
                      validation=validation)


main(False)

