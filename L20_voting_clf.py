#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import time
import os
import sys
import itertools
import xgboost
import tqdm
import multiprocessing

ENSEMBLE_MEMBERS = {
    'ET': 'extratrees_test_2016-06-26-20-34.csv',
    'GB': 'gradient_boost_test_2016-06-26-20-35.csv',
    'NN1': 'knn_01_test_2016-06-26-20-39.csv',
    'NN5': 'knn_05_test_2016-06-26-20-39.csv',
    'NN9': 'knn_09_test_2016-06-26-20-39.csv',
    'NN25': 'knn_25_test_2016-06-26-20-39.csv',
    'NN37': 'knn_37_test_2016-06-26-20-39.csv',
    'NN51': 'knn_51_test_2016-06-26-20-39.csv',
    'NB': 'naive_bayes_test_2016-06-26-22-03.csv',
    'NXGB': 'naive_xgboost_test_2016-06-26-20-34.csv',
    'RF': 'random_forest_test_2016-06-26-20-34.csv',
    'RBFSVM': 'rbf_svm_test_2016-06-26-20-34.csv',

    # these runs are with XGBoost with NN features
    'XGNN0': 'run0_test.csv',
    'XGNN1': 'run1_test.csv',
    'XGNN2': 'run2_test.csv',
}

ens = {}

def load_ensemble():

    d = {}
    for k in ENSEMBLE_MEMBERS:
        d[k] = pd.read_csv(ENSEMBLE_MEMBERS[k])
        d[k].set_index('row_id', inplace=True)
        d[k].sort_index(inplace=True)
        d[k] = d[k]['map3 pred1 pred2 pred3'.split()]

    return d


def make_row(row_id):
    t = []

    for k in ens:
        if row_id in ens[k].index:
            t.append((
                k,
                float(ens[k].iloc[row_id]['map3']),
                int(ens[k].iloc[row_id]['pred1']),
                int(ens[k].iloc[row_id]['pred2']),
                int(ens[k].iloc[row_id]['pred3'])
                )
                )
    df = pd.DataFrame(t, columns='k map3 pred1 pred2 pred3'.split())
    df.set_index('k', inplace=True)
    d = {}
    for r in np.unique(df['pred1 pred2 pred3'.split()].values.ravel()):
        d[r] = 0.

    for idx, z in df.iterrows():
        d[z['pred1']] += z['map3']
        d[z['pred2']] += z['map3'] * 0.5
        d[z['pred3']] += z['map3'] * (1./3.)

    df2 = (pd.DataFrame([(m, d[m]) for m in d], 
        columns=['place_id','weight']))
    df2.sort_values('weight',inplace=True, ascending=False)

    s = ('{},{} {} {}\n'.format( row_id, 
        int(df2.iloc[0].place_id),
        int(df2.iloc[1].place_id),
        int(df2.iloc[2].place_id))
    )
    return s


def main():
    global ens

    ens = load_ensemble()
    N_test = 8607230

    fh = open('comb0.csv', 'w')
    fh.write('row_id,place_id\n')

    for i in tqdm.trange(N_test):
        fh.write(make_row(i))

    fh.close()



if __name__ == '__main__':
    main()