
# coding: utf-8

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
# perform better or the same without kNN features.

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

ENSEMBLE_MEMBERS = [
    # ----- This block of classifiers is in submission format,
    # that is a space separated list of place_ids

    ('ABHI', 'abhi.csv', 0.56951, 1.432, 0.694, 3., 0.),
    ('HARTMANN', 'hartmann.csv', 0.55282, 1.402, 0.693, 1.749, 0.584),

    ('NN16', 'knn_16_run_A8_pred.csv', 0.545708, 1.290, 0.668, 1.377, 0.659),
    ('NN19', 'knn_19_run_A7_pred.csv', 0.546601, 1.333, 0.663, 1.651, 0.587),
    ('NN24', 'knn_24_run_A9_pred.csv', 0.546892, 1.336, 0.696, 1.692, 0.608),
    ('NN29', 'knn_29_run_A3_pred.csv', 0.548435, 1.487, 0.685, 1.834, 0.570),
    ('NN31', 'knn_31_run_A5_pred.csv', 0.548821, 1.477, 0.690, 1.843, 0.564),

    # --------- This block of classifiers has cell-wise map3 as a column -----

    ('ET1', 'extratrees_test_2016-06-26-20-34.csv', 0.529532, 1.157, 0.730, 1.179, 0.705),
    ('ET2', 'extratrees_test_2016-07-04-15-39.csv', 0.525208, 1.128, 0.732, 1.152, 0.709),

    ('NXGB1', 'naive_xgboost_test_2016-06-26-20-34.csv', 0.5676, 1.536, 0.702, 1.285, 0.702),
    ('NXGB2', 'naive_xgboost_test_2016-07-04-15-14.csv', 0.5665, 1.513, 0.710, 1.273, 0.712),
    ('NXGB3', 'run9_test.csv', 0.565445, 1.578, 0.694, 1.373, 0.694),

    ('RF1', 'random_forest_test_2016-06-26-20-34.csv', 0.550621, 1.333, 0.691, 1.276, 0.692),
    ('RF2', 'random_forest_test_2016-07-04-15-46.csv', 0.551428, 1.345, 0.693, 1.281, 0.695),

    # these runs are with XGBoost with NN features
    ('XGNN0', 'run0_test.csv', 0.555004, 1.616, 0.694, 1.273, 0.727),
    ('XGNN1', 'run1_test.csv', 0.559433, 3, 0., 1.432, 0.694),
    ('XGNN2', 'run2_test.csv', 0.563182, 1.914, 0.593, 1.422, 0.703),
    ('XGNN3', 'run3_test.csv', 0.564233, 1.866, 0.600, 1.490, 0.682),
    ('XGNN4', 'run4_test.csv', 0.566821, 1.821, 0.605, 1.479, 0.693),
    ('XGNN5', 'run5_test.csv', 0.554711, 1.405, 0.740, 1.213, 0.729),
    ('XGNN6', 'run6_test.csv', 0.559900, 1.637, 0.699, 1.339, 0.712),
    ('XGNN7', 'run7_test.csv', 0.564275, 1.597, 0.695, 1.344, 0.707),
    ('XGNN8', 'run8_test.csv', 0.562885, 1.685, 0.673, 1.393, 0.692),
    ('XGNN9', 'run11_test.csv', 0.564363, 1.632, 0.677, 1.329, 0.686),

    ('ETNN1', 'run10_test.csv', 0.549796, 1.424, 0.706, 1.387, 0.688),
    ('RFNN1', 'run12_test.csv', 0.549942, 1.408, 0.675, 1.289, 0.691), ]

ENSEMBLE_MEMBERS = pd.DataFrame(ENSEMBLE_MEMBERS, columns=[
    'key',
    'file',
    'map3',
    'run1_agreement_trace',
    'run1_agreement_offdiag',
    'abhi_agreement_trace',
    'abhi_agreement_offdiag'])
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
                d[k]['weight'] = ENSEMBLE_MEMBERS.ix[k].map3
            else:
                d[k] = pd.read_csv(filename)
                d[k] = d[k]['row_id pred1 pred2 pred3'.split()]
                d[k]['weight'] = ENSEMBLE_MEMBERS.ix[k].map3
        d[k].sort_values('row_id', inplace=True)
        print(d[k].head())

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


ens = load_ensemble()
df = pd.concat(ens.values())
del ens

fh = open('comb_all_with_global_map3.csv', 'w')
fh.write('row_id,place_id\n')

dt = datetime.datetime.now()

for row_id, d in df.groupby(['row_id']):
    z = d.drop('row_id', axis=1)
    weights = z.weights.values
    z.drop('weight', inplace=True, axis=1)
    p = top_3_preds(weights, z.values)
    fh.write('{},{} {} {}\n'.format(row_id, p[0], p[1], p[2]))

    if (row_id % 10000 == 0):
        mm = datetime.datetime.now()
        print(mm - dt)
        dt = mm
        print(row_id)


