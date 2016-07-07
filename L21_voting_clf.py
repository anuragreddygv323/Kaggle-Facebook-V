#!/usr/bin/env python3
# coding: utf-8

# Author Ravi Shekhar

import pandas as pd
import numpy as np
import numba
import datetime
import time
import os
import sys
import itertools
import bayes_opt
import copy
import functools
import tqdm
import sklearn.preprocessing
import multiprocessing

# This block gives accuracy ratings. In hindsight, the XGBoost seems to
# perform better or the same without kNN features. However, the trees might
# become more decorrelated with the kNN features, which would improve the
# ensemble.

# for x in *vali*csv ; do echo $x ; python agreement_matrix.py ../input/train.csv $x |grep Horiz ; done
# extratrees_vali_2016-06-26-20-34.csv
# Horiz Sum: [0.45204, 0.11639, 0.05789]
# extratrees_vali_2016-07-04-15-39.csv
# Horiz Sum: [0.44735, 0.11681, 0.05837]
# knn_16_run_A8_vali.csv
# Horiz Sum: [0.46846, 0.11925, 0.05289]
# knn_19_run_A7_vali.csv
# Horiz Sum: [0.46902, 0.11932, 0.05377]
# knn_24_run_A9_vali.csv
# Horiz Sum: [0.46648, 0.12272, 0.05716]
# knn_29_run_A3_vali.csv
# Horiz Sum: [0.46627, 0.12499, 0.05900]
# knn_31_run_A5_vali.csv
# Horiz Sum: [0.46634, 0.12539, 0.05936]
# naive_xgboost_vali_2016-06-26-20-34.csv
# Horiz Sum: [0.48718, 0.12212, 0.05811]
# naive_xgboost_vali_2016-07-04-15-14.csv
# Horiz Sum: [0.48547, 0.12280, 0.05858]
# random_forest_vali_2016-06-26-20-34.csv
# Horiz Sum: [0.47432, 0.11561, 0.05549]
# random_forest_vali_2016-07-04-15-46.csv
# Horiz Sum: [0.47466, 0.11636, 0.05578]
# run0_vali.csv
# Horiz Sum: [0.47531, 0.12069, 0.05806]
# run10_vali.csv
# Horiz Sum: [0.47158, 0.11801, 0.05763]
# run11_vali.csv
# Horiz Sum: [0.48480, 0.12098, 0.05722]
# run12_vali.csv
# Horiz Sum: [0.47416, 0.11461, 0.05543]
# run1_vali.csv
# Horiz Sum: [0.47804, 0.12339, 0.05910]
# run2_vali.csv
# Horiz Sum: [0.48303, 0.12185, 0.05768]
# run3_vali.csv
# Horiz Sum: [0.48263, 0.12385, 0.05903]
# run4_vali.csv
# Horiz Sum: [0.48588, 0.12296, 0.05838]
# run5_vali.csv
# Horiz Sum: [0.47463, 0.12136, 0.05821]
# run6_vali.csv
# Horiz Sum: [0.47888, 0.12288, 0.05875]
# run7_vali.csv
# Horiz Sum: [0.48435, 0.12168, 0.05727]
# run8_vali.csv
# Horiz Sum: [0.48267, 0.12202, 0.05761]
# run9_vali.csv
# Horiz Sum: [0.48451, 0.12318, 0.05803]


# These are the predictions to ensemble together. Less correlation will improve
# the ensemble more so we derive weights. The agreement_offdiag and
# agreement_trace are derived from L19_ens_weights_guess.py, with ExtraTrees ET1
# and ET2 part of the ensemble, which have since been dropped for very low MAP3.
# The agreement matrices were tested and explained in agreement_matrix.py.

# When classifiers are highly correlated, the agreement trace is large
# When classifiers are more independent, but reshufflings of the same
# categories, the offdiagonals are large.
ENSEMBLE_MEMBERS = [
    # ----- This block of classifiers is in submission format,
    # that is a space separated list of place_ids

    # ('ABHI', 'abhi.csv', 0.56951, ),
    # ('HARTMANN', 'hartmann.csv', 0.55282, ),

    ('NN16', 'knn_16_run_A8_vali.csv', 0.545708, ),
    ('NN19', 'knn_19_run_A7_vali.csv', 0.546601, ),
    ('NN24', 'knn_24_run_A9_vali.csv', 0.546892, ),
    ('NN29', 'knn_29_run_A3_vali.csv', 0.548435, ),
    ('NN31', 'knn_31_run_A5_vali.csv', 0.548821, ),

    # --------- This block of classifiers has cell-wise map3 as a column -----

    ('ET1', 'extratrees_vali_2016-06-26-20-34.csv', 0.529532, ),
    ('ET2', 'extratrees_vali_2016-07-04-15-39.csv', 0.525208, ),

    ('NXGB1', 'naive_xgboost_vali_2016-06-26-20-34.csv', 0.5676, ),
    ('NXGB2', 'naive_xgboost_vali_2016-07-04-15-14.csv', 0.5665, ),
    ('NXGB3', 'run9_vali.csv', 0.565445, ),

    ('RF1', 'random_forest_vali_2016-06-26-20-34.csv', 0.550621, ),
    ('RF2', 'random_forest_vali_2016-07-04-15-46.csv', 0.551428, ),

    # these runs are with XGBoost with NN features
    ('XGNN0', 'run0_vali.csv', 0.555004, ),
    ('XGNN1', 'run1_vali.csv', 0.559433, ),
    ('XGNN2', 'run2_vali.csv', 0.563182, ),
    ('XGNN3', 'run3_vali.csv', 0.564233, ),
    ('XGNN4', 'run4_vali.csv', 0.566821, ),
    ('XGNN5', 'run5_vali.csv', 0.554711, ),
    ('XGNN6', 'run6_vali.csv', 0.559900, ),
    ('XGNN7', 'run7_vali.csv', 0.564275, ),
    ('XGNN8', 'run8_vali.csv', 0.562885, ),
    ('XGNN9', 'run11_vali.csv', 0.564363, ),

    ('ETNN1', 'run10_vali.csv', 0.549796, ),
    ('RFNN1', 'run12_vali.csv', 0.549942, ),
]

ENSEMBLE_MEMBERS = pd.DataFrame(ENSEMBLE_MEMBERS, columns=[
    'key',
    'file',
    'map3', ])
ENSEMBLE_MEMBERS.set_index('key', inplace=True)

ens = {}
df = None


def load_ensemble():
    d = {}
    for k in ENSEMBLE_MEMBERS.index.values:
        print(k)
        filename = ENSEMBLE_MEMBERS.ix[k].file
        with open(filename, 'r') as fh:
            l = len(fh.readline().split(','))
            if l == 2:
                # submission format, parse manually
                fh.seek(0)
                r = []
                p1 = []
                p2 = []
                p3 = []

                for l in fh.readlines()[1:]:
                    row_id, place_id = l.split(',')
                    place_ids = place_id.split()
                    r.append(int(row_id))
                    p1.append(int(place_ids[0]))
                    p2.append(int(place_ids[1]))
                    p3.append(int(place_ids[2]))
                d[k] = pd.DataFrame({'row_id': r,
                                     'pred1': p1,
                                     'pred2': p2,
                                     'pred3': p3})
                d[k]['weight'] = 1.
                d[k]['clf'] = k
            else:
                d[k] = pd.read_csv(filename)
                d[k] = d[k]['row_id pred1 pred2 pred3'.split()]
                d[k]['weight'] = 1.
                d[k]['clf'] = k
        d[k].sort_values('row_id', inplace=True)

    return d


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


@numba.autojit
def top_3_preds(weights, pred_place_ids_block):
    n_classifiers = weights.shape[0]
    n_preds = 3

    unique_place_ids = np.sort(np.unique(pred_place_ids_block.ravel()))
    new_weights = np.zeros(unique_place_ids.shape[0], dtype=np.float64)

    for i in range(n_classifiers):
        for j in range(n_preds):
            idx = np.searchsorted(unique_place_ids, pred_place_ids_block[i, j])
            new_weights[idx] += 1./(j+1) * weights[i]

    idx = np.argsort(new_weights)[::-1][:3]
    return unique_place_ids[idx]


def val_err(data, traindata, **kwargs):
    # print(kwargs)
    # print(data.head())

    # reweight data frame
    data['weight'] = np.array([kwargs[k] for k in data.clf.values])
    # print(data.head())

    truthlist = []
    predlist = []

    # this is essentially a manual groupby operation. In my tests, using the
    # pandas groupby seems to make an unnecessary copy of the data, which in
    # my case, exceeds 16GB memory and causes swapping. This ends up being
    # much faster.
    data.sort_values('row_id', inplace=True)
    row_id_values = data.row_id.values
    unique_row_ids = np.sort(np.unique(row_id_values))
    row_indices = np.searchsorted(row_id_values, unique_row_ids)

    for i in tqdm.tqdm(range(unique_row_ids.shape[0])):
        row_id = unique_row_ids[i]
        truthlist.append(traindata.ix[row_id].place_id)

        if i == unique_row_ids.shape[0] - 1:
            z = data.iloc[row_indices[i]:]
        else:
            z = data.iloc[row_indices[i]:row_indices[i+1]]

        weights = z.weight.values
        z = z.drop(['row_id', 'clf', 'weight'], axis=1)
        p = top_3_preds(weights, z.values)
        predlist.append(p)

    truthlist = np.array(truthlist)
    predlist = np.array(predlist)

    return map_k_precision(truthlist, predlist)


def main():
    print("loading ensemble.")
    ens = load_ensemble()
    print("concatenating.")
    df = pd.concat(ens.values())
    del ens
    df.sort_values('row_id', inplace=True)
    print("reading train data")
    traindata = pd.read_csv('../input/train.csv')
    traindata.drop('x y accuracy time'.split(), axis=1, inplace=1)
    traindata.set_index('row_id')

    f = functools.partial(val_err, data=df, traindata=traindata)

    bo = bayes_opt.BayesianOptimization(f=f, verbose=True,
                                        pbounds={
                                            'NN16': (1., 30.),
                                            'NN19': (1., 30.),
                                            'NN24': (1., 30.),
                                            'NN29': (1., 30.),
                                            'NN31': (1., 30.),
                                            'ET1': (1., 30.),
                                            'ET2': (1., 30.),
                                            'NXGB1': (1., 30.),
                                            'NXGB2': (1., 30.),
                                            'NXGB3': (1., 30.),
                                            'RF1': (1., 30.),
                                            'RF2': (1., 30.),
                                            'XGNN0': (1., 30.),
                                            'XGNN1': (1., 30.),
                                            'XGNN2': (1., 30.),
                                            'XGNN3': (1., 30.),
                                            'XGNN4': (1., 30.),
                                            'XGNN5': (1., 30.),
                                            'XGNN6': (1., 30.),
                                            'XGNN7': (1., 30.),
                                            'XGNN8': (1., 30.),
                                            'XGNN9': (1., 30.),
                                            'ETNN1': (1., 30.),
                                            'RFNN1': (1., 30.),
                                        })

    bo.explore({
        'NN16': (11.3560, 2.4614),
        'NN19': (15.5824, 5.3674),
        'NN24': (2.0212, 7.0457),
        'NN29': (12.4096, 3.8345),
        'NN31': (1.2689, 13.7850),
        'ET1': (4.4007, 18.8751),
        'ET2': (3.8069, 6.6030),
        'NXGB1': (17.2929, 11.1042),
        'NXGB2': (26.3967, 16.7655),
        'NXGB3': (23.2920, 14.5272),
        'RF1': (12.2257, 28.5895),
        'RF2': (23.6920, 19.8300),
        'XGNN0': (25.9015, 17.7590),
        'XGNN1': (4.6944, 18.5898),
        'XGNN2': (17.6990, 13.3860),
        'XGNN3': (8.9944, 27.5454),
        'XGNN4': (13.6752, 29.6615),
        'XGNN5': (26.2349, 26.4529),
        'XGNN6': (3.1965, 28.3172),
        'XGNN7': (25.0879, 4.8585),
        'XGNN8': (9.7461, 9.6668),
        'XGNN9': (14.2613, 26.3092),
        'ETNN1': (2.2077, 14.6846),
        'RFNN1': (22.381, 12.3591),
    })

    bo.maximize(n_iter=2000)

    pass

if __name__ == '__main__':
    main()

# print("Starting groupby and write.")

# fh = open('comb_all_with_global_weight2.csv', 'w')
# fh.write('row_id,place_id\n')

# dt = datetime.datetime.now()

# for row_id, d in df.groupby(['row_id']):
#     z = d.drop('row_id', axis=1)
#     weights = z.weight.values
#     z.drop('weight', inplace=True, axis=1)
#     p = top_3_preds(weights, z.values)
#     fh.write('{},{} {} {}\n'.format(row_id, p[0], p[1], p[2]))

#     if (row_id % 10000 == 0):
#         mm = datetime.datetime.now()
#         print(mm - dt)
#         dt = mm
#         print(row_id)
# print("All done")
