# Entry Readme
# Ravi Shekhar

This document provides a tutorial on how to run this software and reproduce the 
results from scratch. For a writeup of the data science, please see 
`writeup.md`. If you want to skip to the last step (ensembling) please consider
loading the database dump and running the ensembler. 

Requires
- Python 2.7 (should run in Python 3 with `2to3`)
- Pandas
- Numpy
- Scikit-Learn
- XGBoost library and python module
- PostgreSQL (9.4+ -- I use json features)
- psycopg2
- SQLAlchemy


## Step 0: Download data

```bash
./fetch_data.sh
```

## Step 1: Create PostgreSQL database

I recommend you do this on a reasonably powerful computer with an SSD disk. All 
the indexes that make the database fast take up a fair bit of space too, around
12GB, plus all the space for storing the output predictions. A database of 100GB
is not unreasonable after ensembling many predictors. I used a Sandy Bridge 
quad core i5 4.0 GHz with a 500GB SSD and 16GB memory, and it was not a 
bottleneck.

This is how I created the database in the psql utility. A password in the code
isn't good practice, but it's not a huge problem since there's an additional
layer of ssh security enforced.

```
create database challenge;
create user shekhar with password 'PasswordHERE';
grant all privileges on database challenge to shekhar;
```

## Step 2: Setup passwordless ssh access

The code will need to SSH from the compute nodes to the database node under
the same account used for the database. Generate a key, add it to your 
authorized_keys file on the database node.


## Step 3: Setup Connection string

```bash
cp sql.url.example sql.url
cp db.hostname.example db.hostname
```

Edit the sql.url to point to your database. See SQLAlchemy docs for more 
details. Also edit db.hostname to point to the the database account on your 
database server. In my case, it's shekhar@dbnode.internal.


## Step 4: Preprocess and Generate Features

Take the input data, grid it (select x,y subsets), generate new features, 
and generate `place_id` features. Spit the results back out as new csv files.

```bash
python preprocess.py
```

You should now have `places_processed.csv`, `train_processed.csv`, and 
`test_processed.csv`. 

## Step 5: Populate database

Use the psql utility with the username and password to populate the database
with the schema defined in setup.sql.

```bash
psql -U shekhar -d challenge < setup.sql
```

## Step 6: Follow instructions in AnalysisPipeline.ipynb

It's important that the instructions in AnalysisPipeline be followed in order,
and not accidentally re-executed. To that effect, all *dangerous* commands are
blocked off with an `if False:`, which must be manually changed before executing
the cell.
