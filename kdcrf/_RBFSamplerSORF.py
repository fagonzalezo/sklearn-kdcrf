"""
Class for RBF Sampler with Orthogonal Random Features
"""
import warnings

import numpy as np
import scipy.stats as stats
from scipy.linalg import hadamard

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils.validation import check_non_negative, _deprecate_positional_args


class RBFSamplerSORF(TransformerMixin, BaseEstimator):
    """Approximates feature map of an RBF kernel by Structured Orthogonal Random Features
    of its Fourier transform.

    It implements a variant of Structured Orthogonal Random Features.[1]

    Read more in the :ref:`User Guide <rbf_kernel_approx>`.

    Parameters
    ----------
    gamma : float
        Parameter of RBF kernel: exp(-gamma * x^2)

    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.

    n_blocks : int
        Number of blocks for computing the dot product in the
        Kronecker-product spaces.

    random_state : int, RandomState instance or None, optional (default=None)
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    random_offset_ : ndarray of shape (n_components,), dtype=float64
        Random offset used to compute the projection in the `n_components`
        dimensions of the feature space.

    random_weights_ : ndarray of shape (n_features, n_components),\
        dtype=float64
        Random projection directions drawn from the Fourier transform
        of the RBF kernel.


    Examples
    --------
    >>> from sklearn.kernel_approximation import RBFSampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> rbf_feature = RBFSamplerSORF(gamma=1, random_state=1)
    >>> X_features = rbf_feature.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=5, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=5)
    >>> clf.score(X_features, y)
    1.0

    Notes
    -----
    "Note that these Hn matrices are defined only for n a power of 2, but if needed, one can always adjust data by
    padding with 0s to enable the use of ‘the next larger’ H, doubling the number of dimensions in the worst case."
    See "Orthogonal Random Features" by Felix, X et al.

    [1] "Orthogonal Random Features" by Felix, X et al.
    (https://arxiv.org/pdf/1610.09072)
    """

    @_deprecate_positional_args
    def __init__(self, *, gamma=1., n_components=100, n_blocks=3, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.n_blocks = n_blocks
        assert n_blocks > 0

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

        X = self._validate_data(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]

        # order_matrix = int(n_features if n_features == 1 or n_features == 2 else np.power(2, np.floor(np.log2(n_features-1))+1))
        # stack_random_weights = []
        # for j in range(round(self.n_components / n_features)+1):
        #     fwht_matrix = np.ones(shape=(order_matrix, order_matrix))
        #     for i in range(0, self.n_blocks):
        #         D = np.random.choice(a=(-1, 1), size=order_matrix, replace=True)
        #         D_diag = np.diag(D)
        #         if i == 0:
        #             fwht_matrix = self.fwht(D_diag)
        #         else:
        #             fwht_matrix = np.dot(fwht_matrix, self.fwht(D_diag))
        #     stack_random_weights.append(fwht_matrix)

        order_matrix = int(n_features if n_features == 1 or n_features == 2 else np.power(2, np.floor(np.log2(n_features - 1)) + 1))
        S_matrix = hadamard(n=order_matrix)
        stack_random_weights = []
        for j in range(round(self.n_components / n_features) + 1):
            random_weights_ = np.ones(shape=S_matrix.shape)
            for i in range(0, self.n_blocks):
                D = np.random.choice(a=(-1, 1), size=order_matrix, replace=True)
                D_diag = np.diag(D)

                if i == 0:
                    random_weights_ = np.dot(S_matrix, D_diag)
                else:
                    random_weights_ = np.dot(random_weights_, np.dot(S_matrix, D_diag))
            stack_random_weights.append(random_weights_)

        self.random_weights_ = np.sqrt(n_features) * np.sqrt(2 * self.gamma) * np.hstack(stack_random_weights)[:n_features, :self.n_components]
        self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)

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
        check_is_fitted(self)

        X = check_array(X, accept_sparse='csr')
        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(self.n_components)
        return projection


    def fwht(self, matrix):
        """ Simple implementation of FWHT"""
        bit = length = len(matrix)

        for _ in range(int(np.log2(length))):
            bit >>= 1
            for i in range(length):
                if i & bit == 0:
                    j = i | bit
                    temp = matrix[i]  # this copies by value
                    matrix[i] += matrix[j]
                    matrix[j] = temp - matrix[j]

        return matrix / np.sqrt(length)
