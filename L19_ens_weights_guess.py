#!/usr/bin/env python3

import pandas as pd
import numpy as np


def agreement(filename1, filename2):
    'Calculate the agreement matrix, return the trace, and the sum of the offdiagonals'
    print("Reading files with pandas.")

    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    if ('place_id' in df1.columns) and ('x' not in df1.columns):
        # this means it's in submission format, with space separators. parse
        # manually

        r = []
        p1 = []
        p2 = []
        p3 = []

        print("Manually parsing {}".format(filename1))
        fh = open(filename1, 'r')
        for l in fh.readlines()[1:]:
            row_id, place_id = l.split(',')
            place_ids = place_id.split()
            r.append(int(row_id))
            p1.append(int(place_ids[0]))
            p2.append(int(place_ids[1]))
            p3.append(int(place_ids[2]))
        df1 = pd.DataFrame(
            {'row_id': r, 'pred1': p1, 'pred2': p2, 'pred3': p3})
        fh.close()
    # this should be triggered if train.csv is an input
    elif 'x' in df1.columns:
        df1['pred1'] = df1['place_id']
        df1['pred2'] = -99
        df1['pred3'] = -99

    if ('place_id' in df2.columns) and ('x' not in df2.columns):
        # this means it's in submission format, with space separators. parse
        # manually

        r = []
        p1 = []
        p2 = []
        p3 = []

        print("Manually parsing {}".format(filename2))
        fh = open(filename2, 'r')
        for l in fh.readlines()[1:]:
            row_id, place_id = l.split(',')
            place_ids = place_id.split()
            r.append(int(row_id))
            p1.append(int(place_ids[0]))
            p2.append(int(place_ids[1]))
            p3.append(int(place_ids[2]))
        df2 = pd.DataFrame({'row_id': r, 'pred1': p1,
                            'pred2': p2,
                            'pred3': p3})
        fh.close()

    df1.sort_values('row_id', inplace=True)
    df2.sort_values('row_id', inplace=True)

    print("merging dataframes")
    merged = pd.merge(df1, df2, on='row_id', suffixes=['_1', '_2'])

    N = merged.shape[0]

    mat = np.array(np.zeros((3, 3), dtype=np.float64))

    print("constructing matrix")
    for i in range(3):
        for j in range(3):
            # print((i, j, 'pred{}_1'.format(i+1), 'pred{}_2'.format(j+1)))
            mat[i, j] = (np.count_nonzero(
                merged['pred{}_1'.format(i+1)].values
                == merged['pred{}_2'.format(j+1)].values)) / N

    mat = np.round(mat, 5)
    print(mat)
    trace = np.trace(mat)
    offdiag = np.sum(mat) - trace
    print("Trace: {0:.4f}    Offdiag: {1:.4f}".format(trace, offdiag))
    return trace, offdiag


ENSEMBLE_MEMBERS = [
    # ----- This block of classifiers is in submission format,
    # that is a space separated list of place_ids

    ('ABHI', 'abhi.csv', 0.56951),
    ('HARTMANN', 'hartmann.csv', 0.55282),

    ('NN16', 'knn_16_run_A8_pred.csv', 0.545708),
    ('NN19', 'knn_19_run_A7_pred.csv', 0.546601),
    ('NN24', 'knn_24_run_A9_pred.csv', 0.546892),
    ('NN29', 'knn_29_run_A3_pred.csv', 0.548435),
    ('NN31', 'knn_31_run_A5_pred.csv', 0.548821),

    # --------- This block of classifiers has cell-wise map3 as a column -----

    ('ET1', 'extratrees_test_2016-06-26-20-34.csv', 0.529532),
    ('ET2', 'extratrees_test_2016-07-04-15-39.csv', 0.525208),

    ('NXGB1', 'naive_xgboost_test_2016-06-26-20-34.csv', 0.5676),
    ('NXGB2', 'naive_xgboost_test_2016-07-04-15-14.csv', 0.5665),
    ('NXGB3', 'run9_test.csv', 0.565445),

    ('RF1', 'random_forest_test_2016-06-26-20-34.csv', 0.550621),
    ('RF2', 'random_forest_test_2016-07-04-15-46.csv', 0.551428),

    # these runs are with XGBoost with NN features
    ('XGNN0', 'run0_test.csv', 0.555004),
    ('XGNN1', 'run1_test.csv', 0.559433),
    ('XGNN2', 'run2_test.csv', 0.563182),
    ('XGNN3', 'run3_test.csv', 0.564233),
    ('XGNN4', 'run4_test.csv', 0.566821),
    ('XGNN5', 'run5_test.csv', 0.554711),
    ('XGNN6', 'run6_test.csv', 0.559900),
    ('XGNN7', 'run7_test.csv', 0.564275),
    ('XGNN8', 'run8_test.csv', 0.562885),
    ('XGNN9', 'run11_test.csv', 0.564363),

    ('ETNN1', 'run10_test.csv', 0.549796),
    ('RFNN1', 'run12_test.csv', 0.549942),
]


ENSEMBLE_MEMBERS = pd.DataFrame(ENSEMBLE_MEMBERS, columns=[
    'key',
    'file',
    'map3'])
ENSEMBLE_MEMBERS.set_index('key', inplace=True)


def main():
    d = {}
    runs = ENSEMBLE_MEMBERS.index.values

    keys = []
    traces = []
    offdiagonals = []

    for x in runs:
        for y in runs:
            print((x, y))
            if x == y:
                continue
            if '{}_{}'.format(x, y) in d:
                continue
            tr, offdiag = agreement(ENSEMBLE_MEMBERS.ix[x].file,
                                    ENSEMBLE_MEMBERS.ix[y].file)
            d['{}_{}'.format(x, y)] = (tr, offdiag)
            d['{}_{}'.format(y, x)] = (tr, offdiag)
            print('{}_{} : {}, {}'.format(x, y, tr, offdiag))

    for x in runs:
        tr = []
        offdiag = []

        for k in d:
            if k.startswith(x+"_"):
                t = d[k]
                tr.append(t[0])
                offdiag.append(t[1])
        keys.append(x)
        traces.append(np.mean(tr))
        offdiagonals.append(np.mean(offdiag))
        print(x)
        print(tr)
        print(offdiag)

    new_df = pd.DataFrame({
        'key': keys,
        'traces': traces,
        'offdiagonals' : offdiagonals
        })

    new_df = pd.merge(new_df, ENSEMBLE_MEMBERS, left_on='key', right_index=True)

    print(new_df.to_json())
    fh = open('ensemble_weights.json', 'w')
    fh.write(new_df.to_json())
    fh.close()
    pass

if __name__ == '__main__':
    main()
