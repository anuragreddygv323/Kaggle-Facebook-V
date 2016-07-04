#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys

df = pd.read_csv(sys.argv[1])
df.drop('map3', axis=1, inplace=True)
df = df['row_id pred1 pred2 pred3'.split()]
df.sort_values('row_id', inplace=True)

with open(sys.argv[2], 'w') as fh:
    fh.write('row_id,place_id\n')
    for idx, row in df.iterrows():
        fh.write("{},{} {} {}\n".format(
            row['row_id'], row['pred1'],
            row['pred2'], row['pred3']))

                                
