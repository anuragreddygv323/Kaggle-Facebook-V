#!/usr/bin/env python


## This notebook heavily borrows code from 
#### https://github.com/kylemcdonald/Parametric-t-SNE


import os, os.path
import numpy as np
import pandas as pd
import sklearn.preprocessing

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')
d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm')
                           for mn in train_set.time.values)
train_set['hour'] = (d_times.hour+ d_times.minute/60)
train_set['weekday'] = d_times.weekday 
train_set['month'] = d_times.month 
train_set['year'] = (d_times.year - 2013)

d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm')
                           for mn in test_set.time.values)
test_set['hour'] = (d_times.hour+ d_times.minute/60)
test_set['weekday'] = d_times.weekday 
test_set['month'] = d_times.month 
test_set['year'] = (d_times.year - 2013)

print(train_set.head())

train_set.x /= 12.
train_set.y /= 12.
train_set.hour /= 24.
train_set.weekday /= 7.
train_set.month /= 12.
train_set.year /=2.
train_set.accuracy /= 1500.

test_set.x /= 12.
test_set.y /= 12.
test_set.hour /= 24.
test_set.weekday /= 7.
test_set.month /= 12.
test_set.year /=2.
test_set.accuracy /= 1500.

print(train_set.head())

batch_size = 1000

# This code cell was pulled from notebook indicated by github link at top

def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(np.multiply(D, P)) / sumP
    P = P / sumP
    return H, P

def x2p(X, u=15, tol=1e-4, print_iter=500, max_tries=50, verbose=0):
    # Initialize some variables
    n = X.shape[0]                     # number of instances
    P = np.zeros((n, n))               # empty probability matrix
    beta = np.ones(n)                  # empty precision vector
    logU = np.log(u)                   # log of perplexity (= entropy)
    
    # Compute pairwise distances
    if verbose > 0: print('Computing pairwise distances...')
    sum_X = np.sum(np.square(X), axis=1)
    # note: translating sum_X' from matlab to numpy means using reshape to add a dimension
    D = sum_X + sum_X[:,None] + -2 * X.dot(X.T)

    # Run over all datapoints
    if verbose > 0: print('Computing P-values...')
    for i in range(n):
        
        if verbose > 1 and print_iter and i % print_iter == 0:
            print('Computed P-values {} of {} datapoints...'.format(i, n))
        
        # Set minimum and maximum values for precision
        betamin = float('-inf')
        betamax = float('+inf')
        
        # Compute the Gaussian kernel and entropy for the current precision
        indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        Di = D[i, indices]
        H, thisP = Hbeta(Di, beta[i])
        
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:
            
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            
            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        
        # Set the final row of P
        P[i, indices] = thisP
        
    if verbose > 0: 
        print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
        print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
        print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))
    
    return P, beta

def compute_joint_probabilities(samples, batch_size=batch_size, d=2, perplexity=30, tol=1e-5, verbose=0):
    v = d - 1
    
    # Initialize some variables
    n = samples.shape[0]
    batch_size = min(batch_size, n)
    
    # Precompute joint probabilities for all batches
    if verbose > 0: print('Precomputing P-values...')
    batch_count = int(n / batch_size)
    P = np.zeros((batch_count, batch_size, batch_size))
    for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):   
        curX = samples[start:start+batch_size]                   # select batch
        P[i], beta = x2p(curX, perplexity, tol, verbose=verbose) # compute affinities using fixed perplexity
        P[i][np.isnan(P[i])] = 0                                 # make sure we don't have NaN's
        P[i] = (P[i] + P[i].T) # / 2                             # make symmetric
        P[i] = P[i] / P[i].sum()                                 # obtain estimation of joint probabilities
        P[i] = np.maximum(P[i], np.finfo(P[i].dtype).eps)

    return P

# P is the joint probabilities for this batch (Keras loss functions call this y_true)
# activations is the low-dimensional output (Keras loss functions call this y_pred)
def tsne(P, activations):
#     d = K.shape(activations)[1]
    d = 2 # TODO: should set this automatically, but the above is very slow for some reason
    n = batch_size # TODO: should set this automatically
    v = d - 1.
    eps = K.variable(10e-15) # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)
    sum_act = K.sum(K.square(activations), axis=1)
    Q = K.reshape(sum_act, [-1, 1]) + -2 * K.dot(activations, K.transpose(activations))
    Q = (sum_act + Q) / v
    Q = K.pow(1 + Q, -(v + 1) / 2)
    Q *= K.variable(1 - np.eye(n))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C

model = Sequential()
model.add(Dense(500, activation='linear', input_shape=(7,)))
model.add(LeakyReLU(0.3))
model.add(BatchNormalization())
model.add(Dense(500, activation='linear'))
model.add(LeakyReLU(0.3))
model.add(BatchNormalization())
model.add(Dense(500, activation='linear'))
model.add(LeakyReLU(0.3))
model.add(Dense(2))
sgd = SGD(lr=0.1, momentum=0.05, nesterov=True)
model.compile(loss=tsne, optimizer=sgd)
json_string = model.to_json()
open('checkpoints/keras_tsne.json', 'w').write(json_string)

rs = np.random.RandomState(7)
for i in range(1000):
    X_train = train_set['x y accuracy hour weekday month year'.split()].sample(200000, random_state=rs).as_matrix()
    if os.path.isfile('checkpoints/{0:03d}.npy'.format(i)):
        P = np.load('checkpoints/{0:03d}.npy'.format(i))
    else:
        P = compute_joint_probabilities(X_train, batch_size=batch_size, verbose=0)
        np.save('checkpoints/{0:03d}.npy'.format(i), P)
    
    Y_train = P.reshape(X_train.shape[0], -1)

    if i > 1:
        nb_epoch = 30
    else:
        nb_epoch = 100
    model.fit(X_train, Y_train, batch_size=batch_size, shuffle=True, nb_epoch=100)
    model.save_weights('checkpoints/{0:03d}.h5'.format(i))


