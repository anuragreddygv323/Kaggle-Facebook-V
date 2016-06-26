#!/usr/bin/env python
# coding: utf-8
# Author : Ravi Shekhar < ravi_dot_shekhar_at_gmail_dot_com >


from bayes_opt import BayesianOptimization

import argparse
import collections
import functools
import itertools
import json
import numpy as np
import os.path
import pandas as pd
import sklearn.neighbors
import sklearn.preprocessing
import sys
import xgboost

NBINS_X = 100
NBINS_Y = 100


# Without bayesian hyperparameter optimization
UNOPTIMIZED_KNN_PARAMS = {
    'w_x': 500.,
    'w_y': 1000.,
    'w_hour': 4.,
    'w_weekday': 3.,
    'w_year': 10.,
    'w_log10acc': 10.,
    'margin': 0.,
    'n_neighbors': 25,
    'cut_threshold': 10,
    'metric': 'manhattan'
}


def load_data():
    '''Load data, generate time features'''

    print("Loading data.")

    if os.path.isfile('train.pickle'):
        train_set = pd.read_pickle('train.pickle')
    else:
        train_set = pd.read_csv('train.csv')
        initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')
        d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm')
                                   for mn in train_set.time.values)
        train_set['hour'] = (d_times.hour + d_times.minute/60)
        train_set['weekday'] = d_times.weekday
        train_set['month'] = d_times.month
        train_set['year'] = (d_times.year - 2013)
        train_set['log10acc'] = np.log10(train_set.accuracy)
        train_set.to_pickle('train.pickle')

    if os.path.isfile('test.pickle'):
        test_set = pd.read_pickle('test.pickle')
    else:
        test_set = pd.read_csv('test.csv')
        d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm')
                                   for mn in test_set.time.values)
        test_set['hour'] = (d_times.hour + d_times.minute/60)
        test_set['weekday'] = d_times.weekday
        test_set['month'] = d_times.month
        test_set['year'] = (d_times.year - 2013)
        test_set['log10acc'] = np.log10(test_set.accuracy)
        test_set.to_pickle('test.pickle')

    print("Data loaded.")

    return train_set, test_set


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
    return np.mean(np.sum(z, axis=1))


def parse_args():
    '''Parse the command line arguments, print help if needed.'''

    parser = argparse.ArgumentParser("Fit an optimized xgboost classifier")
    parser.add_argument('-N', '--njobs', action='store', metavar='N',
                        type=int, default=1,
                        help="Number of compute jobs")
    parser.add_argument('-i', '--ijob', action='store', metavar='i',
                        type=int, default=0,
                        help="Run job i")
    parser.add_argument('--knnoptimize', action='store_true',
                        default=False,
                        help="Bayesian search for optimum hyperparameters")

    # vars returns a dictionary, which is far easier to work with
    return vars(parser.parse_args())


def main():
    args = parse_args()
    train_set, test_set = load_data()

    np.random.seed(42)

    if args['knnoptimize']:
        kNNOptimize(train_set, test_set, args['njobs'], args['ijob'])


def validation_map3_kNN(train_set, xlower, xupper, ylower, yupper,
                        w_x=500., w_y=1000., w_hour=4.,
                        w_weekday=3., w_year=10., w_log10acc=10.,
                        margin=0., n_neighbors=25,
                        cut_threshold=10, metric='manhattan'
                        ):
    # print(locals())

    epsilon = 1.0e-5

    # include the points on the right and top edges
    if xupper == 10.:
        xupper += epsilon
    if yupper == 10.:
        yupper += epsilon

    train_set_bin_with_margin = \
        train_set[(train_set.x >= xlower - margin)
                  & (train_set.x < xupper + margin)
                  & (train_set.y >= ylower - margin)
                  & (train_set.y < yupper + margin)].copy()
    train_set_bin_with_margin.sort_values('time', inplace=True)
    Nrows = train_set_bin_with_margin.shape[0]

    eighty_percent_mark = int(0.8*Nrows)

    train_set_bin = train_set_bin_with_margin[:eighty_percent_mark]

    # take the final 20% and shuffle it
    validation_set = train_set_bin_with_margin[eighty_percent_mark:]

    # To decrease overfitting the details of the validation fold, I take the
    # final 20% of the data and split it into two parts randomly multiple times.
    # Each time (specified by N_iter), I calculate the MAP@3 under this
    # set of hyperparameters. I return the average MAP@3 under these random
    # random splits.
    N_iter = 400

    map3_values = []

    for i in range(N_iter):
        new_train_set_bin = train_set_bin.copy()

        # shuffle the samples around randomly.
        # Due to the interaction of margin with this shuffle step,
        # it's going to be non-trivial to reproduce a specific split.
        new_validation_set_bin = validation_set.sample(frac=1.)

        # take half the shuffled data, append to train set, and the other
        # half is for the validation set.
        half_mark = int(new_validation_set_bin.shape[0] * 0.5)
        new_train_set_bin.append(new_validation_set_bin[:half_mark])
        new_validation_set_bin = new_validation_set_bin[half_mark:]
        new_validation_set_bin = new_validation_set_bin[
            (new_validation_set_bin.x >= xlower)
            & (new_validation_set_bin.x < xupper)
            & (new_validation_set_bin.y >= ylower)
            & (new_validation_set_bin.y < yupper)
        ]

        place_counts = new_train_set_bin.place_id.value_counts()
        mask = place_counts[new_train_set_bin.place_id.values] >= cut_threshold
        new_train_set_bin = new_train_set_bin.loc[mask.values]

        # There is an unconstrained degree of freedom where all the values can
        # vary  simultaneously and maintain the same ratio. To counteract this,
        # I do not set w_month. All other distances must scale to this one.

        new_train_set_bin.x *= w_x
        new_train_set_bin.y *= w_y
        new_train_set_bin.hour *= w_hour
        new_train_set_bin.log10acc *= w_log10acc
        new_train_set_bin.weekday *= w_weekday
#         new_train_set_bin.month *= w_month
        new_train_set_bin.year *= w_year

        new_validation_set_bin.x *= w_x
        new_validation_set_bin.y *= w_y
        new_validation_set_bin.hour *= w_hour
        new_validation_set_bin.log10acc *= w_log10acc
        new_validation_set_bin.weekday *= w_weekday
#         new_validation_set_bin.month *= w_month
        new_validation_set_bin.year *= w_year

        Y_train = new_train_set_bin['place_id'].values
        new_train_set_bin.drop(
            'row_id time place_id accuracy'.split(), axis=1, inplace=True)
        X_train = new_train_set_bin.as_matrix()

        Y_vali = new_validation_set_bin['place_id'].values
        new_validation_set_bin.drop(
            'row_id time place_id accuracy'.split(), axis=1, inplace=True)
        X_vali = new_validation_set_bin.as_matrix()

        classifier = sklearn.neighbors.KNeighborsClassifier(
            int(round(n_neighbors)), metric=metric, n_jobs=1,
            weights='distance')
        classifier.fit(X_train, Y_train)
        predict_y_vali = classifier.predict_proba(X_vali)
        predicted_vali_idx = np.argsort(
            predict_y_vali, axis=1)[:, -3:][:, ::-1]
        map3 = map_k_precision(
            Y_vali, classifier.classes_.take(predicted_vali_idx))
        map3_values.append(map3)

    print(u" MAP@3 estimate: {0:.6g} +- {1:.6g}".format(
        np.mean(map3_values),
        2.0*np.std(map3_values)*float(N_iter)**-0.5))
    return float(np.mean(map3_values))  # cast for json serialization


def validation_map3_xgboost(train_set, xlower, xupper, ylower, yupper,
                            cut_threshold=10, n_estimators=100, gamma=0.01,
                            subsample=0.95, learning_rate=0.1,
                            colsample_bytree=1., colsample_bylevel=1.,
                            reg_alpha=1., reg_lambda=0., min_child_weight=0.3,
                            max_depth=4, margin=0.
                            ):
    # print(locals())

    epsilon = 1.0e-5

    if xupper == 10.:
        xupper += epsilon
    if yupper == 10.:
        yupper += epsilon

    train_set_bin_2 = train_set[(train_set.x >= xlower - margin)
                                & (train_set.x < xupper + margin)
                                & (train_set.y >= ylower - margin)
                                & (train_set.y < yupper + margin)].copy()
    train_set_bin_2.sort_values('time', inplace=True)
    Nrows = train_set_bin_2.shape[0]

    eighty_percent_mark = int(0.8*Nrows)

    train_set_bin = train_set_bin_2[:eighty_percent_mark]

    # take the final 20% and shuffle it
    validation_set = train_set_bin_2[eighty_percent_mark:]

    # this randomly reshuffles the data between the training and testing sets
    # and averages
    N_iter = 1
    # reduces overfitting by bayesian optimizer due to choice of train,
    # validation split

    map3_values = []

    for i in range(N_iter):
        new_train_set_bin = train_set_bin.copy()
        new_validation_set_bin = validation_set.sample(frac=1.)

        # take half the shuffled data, append to train set, and the other half
        # is for the validation set.
        half_mark = int(new_validation_set_bin.shape[0] * 0.5)
        new_train_set_bin.append(new_validation_set_bin[:half_mark])
        new_validation_set_bin = new_validation_set_bin[half_mark:]
        new_validation_set_bin = new_validation_set_bin[
            (new_validation_set_bin.x >= xlower)
            & (new_validation_set_bin.x < xupper)
            & (new_validation_set_bin.y >= ylower)
            & (new_validation_set_bin.y < yupper)
        ]

        place_counts = new_train_set_bin.place_id.value_counts()
        mask = place_counts[new_train_set_bin.place_id.values] >= cut_threshold
        new_train_set_bin = new_train_set_bin.loc[mask.values]

        X_train = new_train_set_bin[
            'x y accuracy hour weekday month year'.split()].as_matrix()
        Y_train = new_train_set_bin['place_id'].values

        X_vali = new_validation_set_bin[
            'x y accuracy hour weekday month year'.split()].as_matrix()
        Y_vali = new_validation_set_bin['place_id'].values

        classifier = xgboost.XGBClassifier(
            n_estimators=int(round(n_estimators)),
            objective='multi:softprob',
            learning_rate=float(learning_rate),
            gamma=gamma, subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            max_depth=int(round(max_depth))
        )
        classifier.fit(X_train, Y_train)
        predict_y_vali = classifier.predict_proba(X_vali)
        predicted_vali_idx = np.argsort(
            predict_y_vali, axis=1)[:, -3:][:, ::-1]
        map3 = mapkprecision(
            Y_vali, classifier.classes_.take(predicted_vali_idx))
        map3_values.append(map3)
    return float(np.mean(map3_values))


def kNNOptimize(train_set, test_set, njobs, ijob):

    delta_x = 10. / NBINS_X
    delta_y = 10. / NBINS_Y

    NBINS_TOTAL = NBINS_X * NBINS_Y
    ijob_bins = np.array_split(np.arange(NBINS_TOTAL), njobs)[ijob]

    for i_bin in ijob_bins:

        bin_filename = 'knn_bayes/{0:05d}_{1:02d}_{2:02d}.json'.format(
            i_bin, NBINS_X, NBINS_Y)
        if os.path.isfile(bin_filename):
            continue

        y_lower = int(i_bin / NBINS_X) * delta_y
        x_lower = (i_bin % NBINS_X) * delta_x

        x_upper = x_lower + delta_x
        y_upper = y_lower + delta_y

        # this block is needed because some points fall on the right or
        # top boundary of the domain exactly.
        if x_upper == 10.:
            x_upper += 1.0e-5
        if y_upper == 10.:
            y_upper += 1.0e-5

        initial_points = {"cut_threshold": (5, 7),
                          "w_x": (450, 550),
                          "w_y": (1050, 950),
                          "w_hour": (4, 2),
                          "w_log10acc": (10, 10),
                          "w_weekday": (2, 3),
                          "w_year": (9, 11),
                          "n_neighbors": (20, 25),
                          "margin": (0.02, 0.03)
                          }

        f = functools.partial(validation_map3_kNN,
                              train_set=train_set,
                              xlower=x_lower, xupper=x_upper,
                              ylower=y_lower, yupper=y_upper)
        bo = BayesianOptimization(f=f,
                                  pbounds={"cut_threshold": (3, 12),
                                           "w_x": (250, 1000),
                                           "w_y": (500, 2000),
                                           "w_hour": (1, 10),
                                           "w_log10acc": (5, 30),
                                           "w_weekday": (1, 10),
                                           "w_year": (2, 20),
                                           "n_neighbors": (10, 40),
                                           "margin": (0.01, 0.04)
                                           },
                                  verbose=True)

        # this little bit of code allows seeding of the bayesian optimizer
        # with a few points that you already know are decent parameter values.
        # initial points are based off @Sandro's kNN script.
        #
        # seed the bayesian optimizer with a couple of points.
        bo.explore(initial_points)

        # For some reason that I don't understand, the Bayesian optimizer slows
        # down greatly after 64 iterations. So to be more computationally
        # efficient, limit it to 64.

        # explore the space (xi=0.1)
        # 2 custom (above), 5 initial (implied), 25 exploration  = 32 total
        bo.maximize(n_iter=25, acq="ei", xi=0.1)

        # exploit the peaks for the other 32 iterations (xi=0.)
        bo.maximize(n_iter=32, acq="ei", xi=0.0)

        optimizer_output = bo.res['all']
        optimizer_output['max'] = bo.res['max']

        optimizer_output['i_bin'] = i_bin
        optimizer_output['nx'] = NBINS_X
        optimizer_output['ny'] = NBINS_Y
        optimizer_output['x_lower'] = x_lower
        optimizer_output['y_lower'] = y_lower
        optimizer_output['x_upper'] = x_upper
        optimizer_output['y_upper'] = y_upper

        with open(bin_filename, 'w') as fh:
            fh.write(json.dumps(optimizer_output, sort_keys=True,
                                indent=4, separators=(',', ': ')))
        # print(json.dumps(optimizer_output['max'], sort_keys=True,
        #                  indent=4, separators=(',', ': ')))


if __name__ == '__main__':
    main()
