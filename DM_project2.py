# -*- coding: utf-8 -*-
"""
@author: Casablanca
"""

import numpy as np
import time
#from sklearn.kernel_approximation import Nystroem

from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import SGDClassifier

C_ALPHA = 1e-4
C_BETA1 = 0.34
C_BETA2 = 0.999999999
C_EPSILON = 1e-8

C_components = 10000
C_gamma = 20


start = time.time()


class RBF_Sampler():
    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.
    It implements a variant of Random Kitchen Sinks.[1]
    Read more in the :ref:`User Guide <rbf_kernel_approx>`.
    Parameters
    ----------
    gamma : float
        Parameter of RBF kernel: exp(-gamma * x^2)
    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.
    [1] "Weighted Sums of Random Kitchen Sinks: Replacing
    minimization with randomization in learning" by A. Rahimi and
    Benjamin Recht.
    (http://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)
    """

    def __init__(self, gamma=1., n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.
        Samples random projection according to n_features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the transformer.
        """

        X = check_array(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]

        self.random_weights_ = (np.sqrt(2 * self.gamma) * random_state.normal(
            size=(n_features, self.n_components)))

        self.random_offset_ = random_state.uniform(0, 2 * np.pi,
                                                   size=self.n_components)
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, 'random_weights_')

        X = check_array(X, accept_sparse='csr')
        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(self.n_components)

        return projection


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # normalize
    print 'X', np.shape(X)
    #X = Nystroem(n_components = C_components, gamma = C_gamma, random_state = 1).fit_transform(X)
    rbf = RBFSampler(n_components = C_components, gamma = C_gamma, random_state = 1).fit(X)
    X = rbf.transform(X)
    print 'X', np.shape(X)

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


    sgd = SGDClassifier(loss = 'hinge', alpha = C_ALPHA).fit(X,y)

    w= sgd.coef_
    print 'shape', np.shape(w)
    w = w[0,:]
    
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
    print 'Mapper elapsed time (min)', round((end - start)/60.,2)

    yield "key", w


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    w = np.array(values).mean(axis=0)
    w_output = w.T

    end = time.time()
    print 'Reducer elapsed time (min)', round((end - start)/60.,2)

    yield w_output
