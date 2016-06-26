#!/usr/bin/env python

import numpy as np
import pandas as pd
import scipy.stats as sps

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(10)


def preprocess():
    """Read input csv files, process them, adding features, and write back out.
    """

    logging.info('Reading csv files from disk.')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    logging.info('Calculating frequency.')
    vc = (train.place_id.value_counts())
    vc = pd.DataFrame(vc)
    vc.columns = ['frequency']

    logging.info("Calculating time features")

    train['hour'] = ((train['time']//60)%24+1 ) / 24.
    train['weekday'] = ((train['time']//1440)%7+1 )/ 7.
    train['day'] = ((train['time']//1440)%30+1 ) / 30.
    train['month'] = ((train['time']//43200)%12+1  )/ 12.
    train['year'] = ((train['time']//525600)+1 ) / 2.

    test['hour'] = ((test['time']//60)%24+1 ) / 24.
    test['weekday'] = ((test['time']//1440)%7+1 )/ 7.
    test['day'] = ((test['time']//1440)%30+1 ) / 30.
    test['month'] = ((test['time']//43200)%12+1  )/ 12.
    test['year'] = ((test['time']//525600)+1 ) / 2.

    print(sps.describe(train.year))
    print(sps.describe(test.year))

    logging.info("Rescaling x and y.")
    # rescale data
    train.x = train.x / 5. - 1.
    train.y = train.y / 5. - 1.
    test.x = test.x / 5. - 1
    test.y = test.y / 5. - 1

    logging.info("Generate space features x*y and x/y")
    epsilon = 1.0e-5

    logging.info("Calculating median.")
    k = train[['x', 'y', 'place_id']]
    k = k.groupby('place_id')
    med = k.median()
    # these two functions are unreasonably slow, and I'm not sure why.
    # perhaps it's doing a sort and possibly averaging behind the scenes?
    logging.info("Calculating lower quartile.")
    lower_quartile = k.quantile(0.25)
    logging.info("Calculating upper quartile.")
    upper_quartile = k.quantile(0.75)
    logging.info("Calculating IQR.")
    iqr = upper_quartile - lower_quartile

    med.columns = ['x_med', 'y_med']
    iqr.columns = ['x_iqr', 'y_iqr']

    logging.info("Merging places table")
    places = pd.merge(med, iqr, how='inner', left_index=True,
                      right_index=True)
    places = pd.merge(places, vc, how='inner', left_index=True,
                      right_index=True)

    logging.info("Dumping to csv: places")
    places.to_csv('places_processed.csv', index_label='place_id')

    logging.info("Dumping to csv: train")
    train.drop('row_id', axis=1, inplace=True)
    train.to_csv('train_processed.csv', index_label='train_row_id')

    logging.info("Dumping to csv: test")
    test.drop('row_id', axis=1, inplace=True)
    test.to_csv('test_processed.csv', index_label='test_row_id')
    logging.info("All done. ")


def main():
    logging.warning("This script can use up to 16GB memory. If it"
        " crashes, please check that you have enough memory.")
    preprocess()


if __name__ == '__main__':
    main()
