#!/usr/bin/env python

from pprint import pprint
import argparse
import hashlib
import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import sklearn.cross_validation
import sklearn.neighbors
import sklearn.preprocessing
import sqlalchemy
import sys
import uuid
import xgboost

# Read in connection information so it doesn't get committed to source control
with open('sql.url', 'r') as fh:
    SQL_CONNECTION_STRING = fh.read().strip()
conn = psycopg2.connect(SQL_CONNECTION_STRING)
with open('sql2.url', 'r') as fh:
    SQLALCHEMY_CONNECTION_STRING = fh.read().strip()
engine = sqlalchemy.engine.create_engine(SQLALCHEMY_CONNECTION_STRING)


def mapkprecision(truthvalues, predictions):
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


def select_train_test_data(bin_nx, bin_ny, bin_ix, bin_iy, train_margin=0.):
    "Select train and test data. "
    assert(train_margin >= 0. and train_margin <= 1.)

    epsilon = 1.0e-5

    X_RANGE = (-1., 1.)  # x has been rescaled from -1 to 1 in preprocessing
    Y_RANGE = (-1., 1.)  # y has been rescaled from -1 to 1 in preprocessing

    delta_X = (X_RANGE[1] - X_RANGE[0]) / float(bin_nx)
    delta_Y = (Y_RANGE[1] - Y_RANGE[0]) / float(bin_ny)

    # bin edges
    x_lower = bin_ix * delta_X + X_RANGE[0]
    x_upper = (bin_ix + 1)*delta_X + X_RANGE[0]
    if bin_ix == bin_nx-1:  # a few points are on the upper edges
        x_upper += epsilon

    y_lower = bin_iy * delta_Y + Y_RANGE[0]
    y_upper = (bin_iy + 1)*delta_Y + Y_RANGE[0]
    if bin_iy == bin_ny-1:
        y_upper += epsilon

    d_test = pd.read_sql('SELECT * from test_data WHERE '
                         'x >= %s and x < %s '
                         ' and y>= %s and y < %s'
                         ' ORDER BY time', engine,
                         params=(x_lower, x_upper, y_lower, y_upper))

    if train_margin > 0.:
        x_lower -= delta_X * train_margin
        x_upper += delta_X * train_margin
        y_lower -= delta_Y * train_margin
        y_upper += delta_Y * train_margin

    d_train = pd.read_sql('SELECT * from train_data WHERE '
                          'x >= %s and x < %s '
                          ' and y>= %s and y < %s'
                          ' ORDER BY time', engine,
                          params=(x_lower, x_upper, y_lower, y_upper))

    return d_train, d_test


def keep_fraction_cut(data, keep_fraction):
    """Cut the lowest frequency place_ids such that keep_fraction of the original
    events are retained.

    """

    vc = data.place_id.value_counts()
    vc = pd.DataFrame({'place_id': vc.index, 'frequency': vc.values})

    N_new = data.shape[0]
    cumul_fraction = np.cumsum(vc.frequency, dtype=np.float32)/N_new

    # any place ids after idx+1 are low frequency
    idx = np.searchsorted(cumul_fraction, keep_fraction)
    idx += 1

    vc = vc.iloc[:idx[0]]
    df = pd.merge(data, vc, on='place_id', how='inner')
    assert(float(df.shape[0]) / float(N_new) >= keep_fraction)
    return df


def unmarginify(data, bin_nx, bin_ny, bin_ix, bin_iy):
    "Trim points on the edges of a bin from a dataset."

    epsilon = 1.0e-5

    X_RANGE = (-1., 1.)  # x has been rescaled from -1 to 1 in preprocessing
    Y_RANGE = (-1., 1.)  # y has been rescaled from -1 to 1 in preprocessing

    delta_X = (X_RANGE[1] - X_RANGE[0]) / float(bin_nx)
    delta_Y = (Y_RANGE[1] - Y_RANGE[0]) / float(bin_ny)

    # bin edges
    x_lower = bin_ix * delta_X + X_RANGE[0]
    x_upper = (bin_ix + 1)*delta_X + X_RANGE[0]
    if bin_ix == bin_nx-1:  # a few points are on the upper edges
        x_upper += epsilon

    y_lower = bin_iy * delta_Y + Y_RANGE[0]
    y_upper = (bin_iy + 1)*delta_Y + Y_RANGE[0]
    if bin_iy == bin_ny-1:
        y_upper += epsilon

    data = data[(data['x'] >= x_lower) &
                (data['x'] < x_upper) &
                (data['y'] >= y_lower) &
                (data['y'] < y_upper)]
    return data


def make_kfold_cv_iter(data, kfold_k, keep_fraction,
                       bin_nx, bin_ny, bin_ix, bin_iy):
    '''Makes a k-fold iterator in time that exclude low frequency place_ids.

    Wrap the standard CV iterator, for each training fold, cut out low freq
    place_ids, but not for the testing folds. This is an inefficient
    implementation, generating lists of tuples, but it isn't a huge problem 
    yet. 

    TODO: Refactor this to make it more efficient. It should be possible to
    do this in the SQL database fairly efficiently.

    '''

    N = data.shape[0]
    # print(N)

    cviter = sklearn.cross_validation.KFold(N, kfold_k)
    new_indices = []

    for train_index, test_index in cviter:
        # for the training fold we want to remove low frequency place_id
        new_train_df = data.take(train_index).copy()
        new_train_df['row_index'] = np.array(train_index)
        new_train_df = keep_fraction_cut(new_train_df, keep_fraction)
        new_train_index_after_cuts = new_train_df.row_index.values.tolist()

        # for the testing fold we want to remove the events on the margins
        new_test_df = data.take(test_index).copy()
        new_test_df['row_index'] = np.array(test_index)
        new_test_df = unmarginify(new_test_df, bin_nx, bin_ny, bin_ix, bin_iy)
        new_test_index_after_cuts = new_test_df.row_index.values.tolist()

        new_indices.append((new_train_index_after_cuts, test_index))

    # was used for debuggging
    # for tr_i, te_i in new_indices:
    #     print((len(tr_i), len(te_i)))

    return new_indices


def naive_knn_stage2(row):
    # stage 1 should not include bayesian optimization
    assert(row.metaparams['bayes_optimize'] == False)

    # This classifier is based on 
    # https://www.kaggle.com/svpons/facebook-v-predicting-check-ins/grid-knn/code
    # version 3. 

    # To that effect, I do not change

    # print(row)

    df_train, df_test = select_train_test_data(
        row.bin_nx, row.bin_ny, row.bin_ix, row.bin_iy,
        row.metaparams['train_margin'])

    print(df_train.head())
    sys.exit(9)

    cviter = make_kfold_cv_iter(df_train, row.metaparams['kfold_k'],
                                row.metaparams['keep_fraction'],
                                row.bin_nx, row.bin_ny, row.bin_ix, row.bin_iy)

    cv_predictions = []

    map3s = []

    # CV iterator
    for train_idx, test_idx in cviter:

        le = sklearn.preprocessing.LabelEncoder()
        y_train_encoded = le.fit_transform(
            df_train.take(train_idx).place_id.values)

        # initialize and fit the train fold
        clf = sklearn.neighbors.KNeighborsClassifier(**row.hyperparams)
        clf.fit(
            df_train.take(train_idx)[row.metaparams['features']].as_matrix(),
            y_train_encoded
        )

        # predict on the test fold
        proba = clf.predict_proba(
            df_train.take(test_idx)[row.metaparams['features']].as_matrix())
        predictions = le.inverse_transform(
            np.argsort(proba, axis=1)[:, -5:][:, ::-1])
        # do an argsort, take five highest values, reverse the list so
        # in descending order, and then find their place_ids.

        # save the test_fold predictions and place_ids for later
        cv_predictions.append(pd.DataFrame({
            'train_row_id': df_train.take(test_idx).train_row_id.values,
            'place_id': df_train.take(test_idx).place_id.values,
            'predicted_1': predictions[:, 0],
            'predicted_2': predictions[:, 1],
            'predicted_3': predictions[:, 2],
            'predicted_4': predictions[:, 3],
            'predicted_5': predictions[:, 4],
        }))

        # append the MAP@3

        map_at_3 = mapkprecision(df_train.take(test_idx).place_id.values,
                                 predictions[:, :3])
        map3s.append(map_at_3)
        # print(map_at_3)

    # refit on the whole training dataset.
    le = sklearn.preprocessing.LabelEncoder()
    
    df_train = keep_fraction_cut(df_train, row.metaparams['keep_fraction'])
    y_train_encoded = le.fit_transform(df_train.place_id.values)
    clf = sklearn.neighbors.KNeighborsClassifier(**row.hyperparams)
    clf.fit(
        df_train[row.metaparams['features']].as_matrix(),
        y_train_encoded
    )

    # predict on the test dataset
    proba = clf.predict_proba(
        df_test[row.metaparams['features']].as_matrix()
    )
    test_predictions = le.inverse_transform(
        np.argsort(proba, axis=1)[:, -5:][:, ::-1])
    test_predictions = pd.DataFrame({
        'test_row_id': df_test.test_row_id.values,
        'predicted_1': test_predictions[:, 0],
        'predicted_2': test_predictions[:, 1],
        'predicted_3': test_predictions[:, 2],
        'predicted_4': test_predictions[:, 3],
        'predicted_5': test_predictions[:, 4],
    })

    # Pickling a NN classifier is a bad idea as it just stores the data points
    # in a ball tree (or some other tree) data structure.
    # So in this case, I'm going to fill NULLs
    id_str = None
    sha1sum = None

    print("NX={0:03d} NY={1:03d} ix={2:03d} iy={3:03d} MAP@3={4:.4f}".format(
        row.bin_nx, row.bin_ny, row.bin_ix, row.bin_iy, np.mean(map3s)))

    # enter into the database 
    # bit of a misnomer, but this should work fine for stage 2 as well.
    # enter_into_db_stage1(uuid=id_str, sha1sum=sha1sum, row=row,
    #                      train_cv_preds=pd.concat(cv_predictions),
    #                      test_preds=test_predictions,
    #                      map_3_folds=map3s
    #                      )
    return


def naive_xgboost_stage1(row):

    # stage 1 should not include bayesian optimization
    assert(row.metaparams['bayes_optimize'] == False)

    print(row)

    df_train, df_test = select_train_test_data(
        row.bin_nx, row.bin_ny, row.bin_ix, row.bin_iy,
        row.metaparams['train_margin'])

    cviter = make_kfold_cv_iter(df_train, row.metaparams['kfold_k'],
                                row.metaparams['keep_fraction'],
                                row.bin_nx, row.bin_ny, row.bin_ix, row.bin_iy)

    cv_predictions = []

    map3s = []

    # CV iterator
    for train_idx, test_idx in cviter:

        # initialize and fit the train fold
        clf = xgboost.XGBClassifier(**row.hyperparams)
        clf.fit(
            df_train.take(train_idx)[row.metaparams['features']].as_matrix(),
            df_train.take(train_idx).place_id.values
        )

        # predict on the test fold
        proba = clf.predict_proba(
            df_train.take(test_idx)[row.metaparams['features']].as_matrix())
        predictions = (clf.classes_.take(
            np.argsort(proba, axis=1)[:, -5:][:, ::-1]))
        # do an argsort, take five highest values, reverse the list so
        # in descending order, and then find their place_ids.

        # save the test_fold predictions and place_ids for later
        cv_predictions.append(pd.DataFrame({
            'train_row_id': df_train.take(test_idx).train_row_id.values,
            'place_id': df_train.take(test_idx).place_id.values,
            'predicted_1': predictions[:, 0],
            'predicted_2': predictions[:, 1],
            'predicted_3': predictions[:, 2],
            'predicted_4': predictions[:, 3],
            'predicted_5': predictions[:, 4],
        }))

        # append the MAP@3

        map_at_3 = mapkprecision(df_train.take(test_idx).place_id.values,
                                 predictions[:, :3])
        map3s.append(map_at_3)
        print(map_at_3)

    # refit on the whole training dataset.
    df_train = keep_fraction_cut(df_train, row.metaparams['keep_fraction'])
    clf = xgboost.XGBClassifier(**row.hyperparams)
    clf.fit(
        df_train[row.metaparams['features']].as_matrix(),
        df_train.place_id.values
    )

    # predict on the test dataset
    proba = clf.predict_proba(
        df_test[row.metaparams['features']].as_matrix()
    )
    test_predictions = (clf.classes_.take(
        np.argsort(proba, axis=1)[:, -5:][:, ::-1]))
    test_predictions = pd.DataFrame({
        'test_row_id': df_test.test_row_id.values,
        'predicted_1': test_predictions[:, 0],
        'predicted_2': test_predictions[:, 1],
        'predicted_3': test_predictions[:, 2],
        'predicted_4': test_predictions[:, 3],
        'predicted_5': test_predictions[:, 4],
    })

    # pickle the classifier for later
    id_str = str(uuid.uuid4())
    joblib.dump(clf, 'checkpoints/{}.pickle'.format(id_str),
                compress=9)
    with open('checkpoints/{}.pickle'.format(id_str), 'r') as fh:
        sha1sum = hashlib.sha1(fh.read()).hexdigest()

    # enter into the database
    enter_into_db_stage1(uuid=id_str, sha1sum=sha1sum, row=row,
                         train_cv_preds=pd.concat(cv_predictions),
                         test_preds=test_predictions,
                         map_3_folds=map3s
                         )
    return


def enter_into_db_stage1(uuid, sha1sum, row,
                         train_cv_preds,
                         test_preds,
                         map_3_folds
                         ):
    cur = conn.cursor()
    cur.execute("""INSERT into pickled_classifiers (pickle_uuid, sha1sum)
        VALUES (%s, %s) RETURNING pickle_id""",
                (uuid, sha1sum)
                )
    pickle_id = cur.fetchone()[0]

    # this seems redundant, but converts a list of np.float32 to list of floats
    map_3_folds = [float(x) for x in map_3_folds]

    cur.execute("""UPDATE cv_bin_results SET 
                       map_at_3=%s ,
                       map_at_3_folds=%s ,
                       pickle_id=%s 
                   WHERE
                       cv_bin_result_id = %s""",
                (float(np.mean(map_3_folds)),
                 map_3_folds,
                 pickle_id,
                 row.cv_bin_result_id
                 ))

    train_cv_preds['map3_contribution'] = (
        (train_cv_preds.place_id == train_cv_preds.predicted_1) +
        (1/2.) * (train_cv_preds.place_id == train_cv_preds.predicted_2) +
        (1/3.) * (train_cv_preds.place_id == train_cv_preds.predicted_3))
    # reorder columns
    train_cv_preds = train_cv_preds['''
     train_row_id place_id 
     predicted_1 predicted_2 predicted_3
     predicted_4 predicted_5 map3_contribution
     '''.split()]

    for idx, r in train_cv_preds.iterrows():
        cur.execute("""INSERT INTO cv_predicted_train(
            pickle_id, cv_bin_result_id, train_row_id, place_id, 
            predicted_1, predicted_2, predicted_3, predicted_4, predicted_5,
            map3_contribution) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);""",
                    (pickle_id, row.cv_bin_result_id,
                     r['train_row_id'], r['place_id'],
                     r['predicted_1'],
                     r['predicted_2'],
                     r['predicted_3'],
                     r['predicted_4'],
                     r['predicted_5'],
                     r['map3_contribution'])
                    )

    test_preds = test_preds['''
     test_row_id
     predicted_1 predicted_2 predicted_3
     predicted_4 predicted_5
     '''.split()]

    for idx, r in test_preds.iterrows():
        cur.execute("""INSERT INTO predicted_test(
            pickle_id, cv_bin_result_id, test_row_id,
            predicted_1, predicted_2, predicted_3, predicted_4, predicted_5)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s);""",
                    (pickle_id, row.cv_bin_result_id,
                     r['test_row_id'],
                     r['predicted_1'],
                     r['predicted_2'],
                     r['predicted_3'],
                     r['predicted_4'],
                     r['predicted_5'])
                    )
    conn.commit()

    pass


def compute(jobs_df):
    dispatch_table = {
        'naive xgboost': naive_xgboost_stage1,
        'naive knn': naive_knn_stage2
    }

    for index, row in jobs_df.iterrows():
        dispatch_key = row['custom_classifier_type'].lower()
        dispatch_table[dispatch_key](row=row)

    pass


def query_for_jobs(stage, njobs, ijob):
    "Query the database, get jobs to execute"

    d = pd.read_sql('SELECT * from cv_bin_results WHERE '
                    "(pipeline_stage = %(stage)s) and (map_at_3 IS NULL)"
                    ' and (cv_bin_result_id %% %(njobs)s = %(ijob)s);', engine,
                    params=locals())
    if d.shape[0] == 0:
        print("No jobs available for stage={} njobs={} ijob={}".format(
            stage, njobs, ijob))
        sys.exit(0)

    return d


def main():
    'Main function'
    args = parse_args()
    jobs_df = query_for_jobs(**args)

    compute(jobs_df)

    pass


def parse_args():
    '''Parse the command line arguments, print help if needed.'''

    parser = argparse.ArgumentParser("Pull jobs from database, fit the data.")
    parser.add_argument('--stage', action='store', metavar='STAGE',
                        type=int, default=1,
                        help="Run pipeline stage STAGE")
    parser.add_argument('-N', '--njobs', action='store', metavar='N',
                        type=int, default=1,
                        help="Number of compute jobs")
    parser.add_argument('-i', '--ijob', action='store', metavar='i',
                        type=int, default=0,
                        help="Run job i")

    # vars returns a dictionary, which is far easier to work with
    return vars(parser.parse_args())


if __name__ == '__main__':
    main()
