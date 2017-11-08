# -*- coding: utf-8 -*-
"""
@author: Casablanca
"""

import numpy as np
import time
from sklearn.linear_model import SGDClassifier

C_ALPHA = 1e-4
C_BETA1 = 0.34
C_BETA2 = 0.999999999
C_EPSILON = 1e-8

C_COMPONENTS = 10000
C_GAMMA = 20
C_RANDOM_STATE = 1


start = time.time()


class rbf_simple():

    def fit(self, X):
        random_state = np.random.RandomState(C_RANDOM_STATE)
        self.weights = (np.sqrt(2 * C_GAMMA) *
                        random_state.normal(size=(X.shape[1], C_COMPONENTS)))
        self.offset = random_state.uniform(0, 2 * np.pi, size=C_COMPONENTS)
        return self

    def transform(self, X):
        projection = np.dot(X, self.weights)
        projection += self.offset
        np.cos(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(C_COMPONENTS)

        return projection


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    print('X', np.shape(X))
    rbf = rbf_simple().fit(X)
    X = rbf.transform(X)
    print('X', np.shape(X))

    return X


def project_L2(w, a):
    """Project to L2-ball, as presented in the lecture."""
    return w * min(1, 1 / (np.sqrt(a) * np.linalg.norm(w, 2)))


def mapper(key, value):
    # key: None
    # value: one line of input file

    features = np.genfromtxt(value, delimiter=' ').T
    y = features[:1].T
    X = features[1:].T
    X = transform(X)

    sgd = SGDClassifier(loss='hinge', alpha=C_ALPHA).fit(X, y)
    w = sgd.coef_
    print('shape', np.shape(w))
    w = w[0, :]

    """
    assert X.shape[0] == y.shape[0]
    w = np.zeros(X.shape[1])
    # Adam
    m = np.ones(X.shape[1])
    v = np.ones(X.shape[1])
    for t in range(X.shape[0]):
        if y[t] * np.dot(w, X[t, :]) < 1:
            eta = 1. / np.sqrt((t + 1.)) #learning rate
            m = C_BETA1 * m + (1. - C_BETA1) * -y[t] * X[t, :]
            m_ = m / (1. - C_BETA1 ** (t + 1.))
            v = C_BETA2 * v + (1. - C_BETA2) * (-y[t] * X[t, :]) ** 2
            v_ = v / (1. - C_BETA2 ** (t + 1.))
            w -= eta * m_ / np.sqrt(v_ + C_EPSILON)
            w = project_L2(w, C_ALPHA)
    """

    end = time.time()
    print('Mapper elapsed time (min)', round((end - start)/60., 2))

    yield "key", w


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    w = np.array(values).mean(axis=0)
    w_output = w.T

    end = time.time()
    print('Reducer elapsed time (min)', round((end - start)/60., 2))

    yield w_output
