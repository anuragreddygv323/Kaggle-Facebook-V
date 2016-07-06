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
import xgboost
import tqdm
import multiprocessing
import sklearn.preprocessing

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

    ('ABHI', 'abhi.csv', 0.56951, 0.675420, 1.414158),
    ('HARTMANN', 'hartmann.csv', 0.55282, 0.669720, 1.385110),

    ('NN16', 'knn_16_run_A8_pred.csv', 0.545708, 0.647591, 1.315414),
    ('NN19', 'knn_19_run_A7_pred.csv', 0.546601, 0.639670, 1.369810),
    ('NN24', 'knn_24_run_A9_pred.csv', 0.546892, 0.666214, 1.349287),
    ('NN29', 'knn_29_run_A3_pred.csv', 0.548435, 0.653935, 1.467409),
    ('NN31', 'knn_31_run_A5_pred.csv', 0.548821, 0.655597, 1.474633),

    # --------- This block of classifiers has cell-wise map3 as a column -----

    # ('ET1', 'extratrees_test_2016-06-26-20-34.csv', 0.529532, 0.702906, 1.185185),
    # ('ET2', 'extratrees_test_2016-07-04-15-39.csv', 0.525208, 0.706655, 1.160033),

    ('NXGB1', 'naive_xgboost_test_2016-06-26-20-34.csv', 0.5676, 0.691904, 1.376551),
    ('NXGB2', 'naive_xgboost_test_2016-07-04-15-14.csv', 0.5665, 0.697120, 1.367123),
    ('NXGB3', 'run9_test.csv', 0.565445, 0.681825, 1.425372),

    ('RF1', 'random_forest_test_2016-06-26-20-34.csv', 0.550621, 0.678264, 1.319189),
    ('RF2', 'random_forest_test_2016-07-04-15-46.csv', 0.551428, 0.680115, 1.327029),

    # these runs are with XGBoost with NN features
    ('XGNN0', 'run0_test.csv', 0.555004, 0.704860, 1.342139),
    ('XGNN1', 'run1_test.csv', 0.559433, 0.683740, 1.478337),
    ('XGNN2', 'run2_test.csv', 0.563182, 0.674729, 1.490334),
    ('XGNN3', 'run3_test.csv', 0.564233, 0.669054, 1.511995),
    ('XGNN4', 'run4_test.csv', 0.566821, 0.666355, 1.502586),
    ('XGNN5', 'run5_test.csv', 0.554711, 0.710070, 1.327578),
    ('XGNN6', 'run6_test.csv', 0.559900, 0.687056, 1.467433),
    ('XGNN7', 'run7_test.csv', 0.564275, 0.673021, 1.486621),
    ('XGNN8', 'run8_test.csv', 0.562885, 0.662386, 1.521970),
    ('XGNN9', 'run11_test.csv', 0.564363, 0.658710, 1.508172),

    ('ETNN1', 'run10_test.csv', 0.549796, 0.672748, 1.401481),
    ('RFNN1', 'run12_test.csv', 0.549942, 0.661569, 1.345943),
    ]

ENSEMBLE_MEMBERS = pd.DataFrame(ENSEMBLE_MEMBERS, columns=[
    'key',
    'file',
    'map3',
    'agreement_offdiag',
    'agreement_trace'])
ENSEMBLE_MEMBERS.set_index('key', inplace=True)

# Some educated guesses on what weights might be helpful.
# We want strong positive correlation with map3
#                positive correlation with agreement_offdiag
#                inverse correlation with agreement_trace
ENSEMBLE_MEMBERS['weight0'] = ENSEMBLE_MEMBERS.map3
ENSEMBLE_MEMBERS['weight1'] = ENSEMBLE_MEMBERS.map3**4. * (ENSEMBLE_MEMBERS.agreement_offdiag)**2 / (ENSEMBLE_MEMBERS.agreement_trace**2.)
ENSEMBLE_MEMBERS['weight2'] = ENSEMBLE_MEMBERS.map3**8. * (ENSEMBLE_MEMBERS.agreement_offdiag)**2 / (ENSEMBLE_MEMBERS.agreement_trace**2.)
ENSEMBLE_MEMBERS['weight3'] = ENSEMBLE_MEMBERS.map3**4. * (ENSEMBLE_MEMBERS.agreement_offdiag)**4 / (ENSEMBLE_MEMBERS.agreement_trace**4.)
ENSEMBLE_MEMBERS['weight4'] = ENSEMBLE_MEMBERS.map3**8. * (ENSEMBLE_MEMBERS.agreement_offdiag)**4 / (ENSEMBLE_MEMBERS.agreement_trace**4.)

ENSEMBLE_MEMBERS['weight0'] /= np.min(ENSEMBLE_MEMBERS.weight0.values)
ENSEMBLE_MEMBERS['weight1'] /= np.min(ENSEMBLE_MEMBERS.weight1.values)
ENSEMBLE_MEMBERS['weight2'] /= np.min(ENSEMBLE_MEMBERS.weight2.values)
ENSEMBLE_MEMBERS['weight3'] /= np.min(ENSEMBLE_MEMBERS.weight3.values)
ENSEMBLE_MEMBERS['weight4'] /= np.min(ENSEMBLE_MEMBERS.weight4.values)

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
                # MODIFY this line to use the appropriate weight formula
                d[k]['weight'] = ENSEMBLE_MEMBERS.ix[k].weight3
            else:
                d[k] = pd.read_csv(filename)
                d[k] = d[k]['row_id pred1 pred2 pred3'.split()]
                # MODIFY this line to use the appropriate weight formula
                d[k]['weight'] = ENSEMBLE_MEMBERS.ix[k].weight3
        d[k].sort_values('row_id', inplace=True)
        print(d[k].head())
        print('\n')

    return d


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


if not os.path.isfile('ens.h5'):
    print("WARNING: This script requires a LOT of memory, something around 2GB "
          "per ensemble member. As configured, around 45 GB was used to put "
          "the ensemble dataframe together. 60GB Amazon EC2 node suggested.")
    print("Loading ensemble")
    ens = load_ensemble()
    print("Ensemble loaded. Concatenating")
    df = pd.concat(ens.values())
    print("Concatenated, deleting stale data")
    del ens
    print("Stale data deleted, saving a copy to disk")
    # df.to_hdf('ens.h5', 'table', complevel=1, complib='blosc', fletcher32=True)
    print(df)
else:
    print("WARNING: This script requires a LOT of memory, around 1.5GB per "
          "ensemble member. As configured, it **barely** completed with 32GB, "
          "60GB Amazon EC2 node suggested.")
    print("Reading ens.h5 from disk with previous options. If you don't "
          "want this, delete ens.h5, "
          "and it will be recreated with current options")
    df = pd.read_hdf('ens.h5', 'table')
    print(df)

print("Starting groupby and write.")

fh = open('comb_all_with_global_weight3.csv', 'w')
fh.write('row_id,place_id\n')

dt = datetime.datetime.now()

for row_id, d in df.groupby(['row_id']):
    z = d.drop('row_id', axis=1)
    weights = z.weight.values
    z.drop('weight', inplace=True, axis=1)
    p = top_3_preds(weights, z.values)
    fh.write('{},{} {} {}\n'.format(row_id, p[0], p[1], p[2]))

    if (row_id % 10000 == 0):
        mm = datetime.datetime.now()
        print(mm - dt)
        dt = mm
        print(row_id)
print("All done")

