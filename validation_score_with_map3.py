#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import os.path

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

def main():
    assert(os.path.isfile(sys.argv[1]))
    df_train = pd.read_csv('../input/train.csv')
    df_vali = pd.read_csv(sys.argv[1])

    N = df_vali.shape[0]
    # truthvalues = np.zeros(N, dtype=np.int64)
    predictions = np.zeros((N,3), dtype=np.int64)

    truthvalues = df_train.iloc[df_vali.row_id.values].place_id.values
    predictions[:, 0] = df_vali.pred1.values
    predictions[:, 1] = df_vali.pred2.values
    predictions[:, 2] = df_vali.pred3 .values

    # with open(sys.argv[1], 'r') as fh:
    #     lines=fh.readlines()[1:]

    #     truthvalues = np.zeros(len(lines), dtype=np.int64)
    #     predictions = np.zeros((len(lines), 3), dtype=np.int64)

    #     for i, line in enumerate(lines):
    #         row_id, placeids = line.strip().split(',')
    #         row_id = int(row_id)

    #         placeids = np.array([int(x) for x in placeids.split()], 
    #             dtype=np.int64)

    #         truthvalues[i] = df_train.place_id.values[row_id]
    #         predictions[i] = placeids


    print(map_k_precision(truthvalues, predictions))
    
    pass

if __name__ == '__main__':
    main()