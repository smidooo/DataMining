# -*- coding: utf-8 -*-
"""
@author: Casablanca
"""

import numpy as np
import time


C_ALPHA = 0.0005
C_BETA1 = 0.99
C_BETA2 = 0.9993
C_COMPONENTS = 20000
C_EPSILON = 1e-8
C_GAMMA = 10
C_RANDOM_STATE = 1
C_LEARN=500
start = time.time()


def transform(X):
    # this is a simplified rbf sampler
    # initialization
    random_state = np.random.RandomState(C_RANDOM_STATE)
    weights = (np.sqrt(2 * C_GAMMA) *
               random_state.normal(size=(X.shape[1], C_COMPONENTS)))
    offset = random_state.uniform(0, 2 * np.pi, size=C_COMPONENTS)

    # projection
    projection = np.dot(X, weights) + offset
    np.cos(projection, projection)
    projection *= np.sqrt(2.) / np.sqrt(C_COMPONENTS)

    return projection


def project_L2(w, a):

    return w * min(1, 1 / (np.sqrt(a) * np.linalg.norm(w, 2)))


def mapper(key, value):
    # key: None
    # value: one line of input file

    features = np.genfromtxt(value, delimiter=' ').T
    y = features[:1].T
    X = features[1:].T
    X = transform(X)

    # sgd = SGDClassifier(loss='hinge', alpha=C_ALPHA).fit(X, y)
    # w = sgd.coef_
    # print('shape', np.shape(w))
    # w = w[0, :]


    assert X.shape[0] == y.shape[0]
    w = np.zeros(X.shape[1])
    # Adam
    m = np.ones(X.shape[1])
    v = np.ones(X.shape[1])
    for t in range(X.shape[0]):
        if y[t] * np.dot(w, X[t, :]) < 1:
            eta = C_LEARN / np.sqrt((t + 1.)) #learning rate np.sqrt(
            m = C_BETA1 * m + (1. - C_BETA1) * -y[t] * X[t, :]
            m_ = m / (1. - C_BETA1 ** (t + 1.))
            v = C_BETA2 * v + (1. - C_BETA2) * (-y[t] * X[t, :]) ** 2
            v_ = v / (1. - C_BETA2 ** (t + 1.))
            w -= eta * m_ / np.sqrt(v_ + C_EPSILON)
            #w += eta * y[t] * X[t, :]
            w = project_L2(w, C_ALPHA)
    # assert x.shape[0] == y.shape[0]
    # w = np.zeros(x.shape[1])
    # for t in range(x.shape[0]):
    #     if y[t] * np.dot(w, x[t, :]) < 1:
    #         eta = 1 / np.sqrt(t + 1)
    #         w += eta * y[t] * x[t, :]
    #         w = project_L2(w, self.a)



    end = time.time()
    print('Mapper elapsed time (min)', round((end - start) / 60., 2))

    yield "key", w


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    w = np.array(values).mean(axis=0)
    w_output = w.T

    end = time.time()
    print('Reducer elapsed time (min)', round((end - start) / 60., 2))

    yield w_output
