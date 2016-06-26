#!/bin/bash

wget 'http://s3.amazonaws.com/rspublic/kaggle-facebook/test.csv.gz'
wget 'http://s3.amazonaws.com/rspublic/kaggle-facebook/train.csv.gz'
wget 'http://s3.amazonaws.com/rspublic/kaggle-facebook/sample_submission.csv.gz'

echo "expanding data"
ls *.csv.gz |xargs -P 3 -n1 gunzip -v
