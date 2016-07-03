
# coding: utf-8

# In[1]:

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

ENSEMBLE_MEMBERS = {
    'ET': 'extratrees_test_2016-06-26-20-34.csv',
#    'GB': 'gradient_boost_test_2016-06-26-20-35.csv',
#    'NN1': 'knn_01_test_2016-06-26-20-39.csv',
#    'NN5': 'knn_05_test_2016-06-26-20-39.csv',
#    'NN9': 'knn_09_test_2016-06-26-20-39.csv',
#    'NN25': 'knn_25_test_2016-06-26-20-39.csv',
#    'NN37': 'knn_37_test_2016-06-26-20-39.csv',
#    'NN51': 'knn_51_test_2016-06-26-20-39.csv',
#    'NB': 'naive_bayes_test_2016-06-26-22-03.csv',
    'NXGB': 'naive_xgboost_test_2016-06-26-20-34.csv',
    'RF': 'random_forest_test_2016-06-26-20-34.csv',
#    'RBFSVM': 'rbf_svm_test_2016-06-26-20-34.csv',

#     these runs are with XGBoost with NN features
    'XGNN0': 'run0_test.csv',
    'XGNN1': 'run1_test.csv',
    'XGNN2': 'run2_test.csv',
    'XGNN3': 'run3_test.csv',
}


# In[2]:

ens = {}
df = None

def load_ensemble():

    d = {}
    for k in ENSEMBLE_MEMBERS:
        d[k] = pd.read_csv(ENSEMBLE_MEMBERS[k])
        d[k] = d[k]['row_id map3 pred1 pred2 pred3'.split()]

    return d


# In[3]:

@numba.autojit
def top_3_preds(map3_vals, pred_place_ids_block):
    n_classifiers = map3_vals.shape[0]
    n_preds = 3
    
    unique_place_ids = np.sort(np.unique(pred_place_ids_block.ravel()))
    weights = np.zeros(unique_place_ids.shape[0], dtype=np.float64)
    
    for i in range(n_classifiers):
        for j in range(n_preds):
            idx = np.searchsorted(unique_place_ids, pred_place_ids_block[i,j])
            weights[idx] += 1./(j+1) * map3_vals[i]
            
    idx = np.argsort(weights)[::-1][:3]
    return unique_place_ids[idx]

    


# In[4]:

ens = load_ensemble()
df = pd.concat(ens.values())
del ens

fh = open('comb2.csv', 'w')
fh.write('row_id,place_id\n')

dt = datetime.datetime.now()

for row_id, d in df.groupby(['row_id']):
    z = d.drop('row_id', axis=1)
    map3s = z.map3.values
    z.drop('map3', inplace=True, axis=1)
    p = top_3_preds(map3s, z.values)
    fh.write('{},{} {} {}\n'.format(row_id, p[0], p[1], p[2]))
    
    if (row_id%10000 == 0):
        mm = datetime.datetime.now()
        print(mm - dt)
        dt = mm
        print(row_id)


# In[ ]:



