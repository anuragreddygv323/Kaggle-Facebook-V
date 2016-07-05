#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys

df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])

if ('place_id' in df1.columns) and ('x' not in df1.columns):
    # this means it's in submission format, with space separators. parse manually

    r = []
    p1 = []
    p2 = []
    p3 = []

    fh = open(sys.argv[1], 'r')
    for l in fh.readlines()[1:]:
        row_id, place_id = l.split(',')
        place_ids = place_id.split()
        r.append(int(row_id))
        p1.append(int(place_ids[0]))
        p2.append(int(place_ids[1]))
        p3.append(int(place_ids[2]))
    df1 = pd.DataFrame({'row_id': r, 'pred1': p1, 'pred2': p2, 'pred3': p3})
    fh.close()
elif 'x' in df1.columns: #this should be triggered if train.csv is an input
    df1['pred1'] = df1['place_id']
    df1['pred2'] = -99
    df1['pred3'] = -99

if ('place_id' in df2.columns) and ('x' not in df2.columns):
    # this means it's in submission format, with space separators. parse manually

    r = []
    p1 = []
    p2 = []
    p3 = []

    fh = open(sys.argv[2], 'r')
    for l in fh.readlines()[1:]:
        row_id, place_id = l.split(',')
        place_ids = place_id.split()
        r.append(int(row_id))
        p1.append(int(place_ids[0]))
        p2.append(int(place_ids[1]))
        p3.append(int(place_ids[2]))
    df2 = pd.DataFrame({'row_id': r, 'pred1': p1, 'pred2': p2, 'pred3': p3})
    fh.close()    

df1.sort_values('row_id', inplace=True)
df2.sort_values('row_id', inplace=True)

merged = pd.merge(df1, df2, on='row_id', suffixes=['_1', '_2'])

N = merged.shape[0]

mat = np.array(np.zeros((3,3), dtype=np.float64))

for i in range(3):
    for j in range(3):
        # print((i, j, 'pred{}_1'.format(i+1), 'pred{}_2'.format(j+1)))
        mat[i,j] = (np.count_nonzero(
            merged['pred{}_1'.format(i+1)].values 
            == merged['pred{}_2'.format(j+1)].values)) / N

mat = np.round(mat, 5)
print(mat)
print("Trace: {0:0.3f}".format(np.trace(mat)))
print("Offdiag: {0:0.3f}".format(np.sum(mat) - np.trace(mat)))
print("Horiz Sum: [{0:.5f}, {1:.5f}, {2:.5f}]".format(*np.squeeze(np.sum(mat, axis=0)).tolist()))
print("Col Sum: [{0:.5f}, {1:.5f}, {2:.5f}]".format(*np.squeeze(np.sum(mat, axis=1)).tolist()))
# print(np.sum(mat, axis=1))
