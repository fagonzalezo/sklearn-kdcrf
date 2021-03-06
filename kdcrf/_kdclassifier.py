"""
Class for Kernel Density Classification with Random Features
"""
import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel


class KDClassifierRF(ClassifierMixin, BaseEstimator):
    """ Kernel Density Classification with Random Features

    Use KDE to estimate the likelihood for each class.

    Parameters
    ----------
    approx : {'exact', 'rff', 'rff+'}, default='rff'
        Algorithm for kernel approximation.
    normalize : bool, default=True
        Whether to normalize the RF transformed vectors. 
    gamma : float, default=1.
        Parameter of RBF kernel: exp(-gamma * x^2)
    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
    random_state : int, RandomState instance or None, optional (default=None)
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, approx='rff', normalize=True,
                 gamma=1., n_components=100,
                 random_state=None, sampler=None, square_dot=False):
        assert approx in ['rff+', 'rff', 'lrff', 'lrff+', 'dmrff', 'orf', 'lorf', 'exact']
        self.approx = approx
        self.normalize = normalize
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.rbf_sampler_ = sampler
        self.square_dot = square_dot

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        super().set_params(**params)
        # Check if sampler is defined
        self.check_if_sampler_is_defined()
        if self.approx in ['rff', 'rff+', 'lrff', 'lrff+','dmrff']:
            for param in ["gamma", "n_components", "random_state"]:
                if param in params.keys():
                    self.rbf_sampler_.set_params(**{param: params[param]})

        return self

    def check_if_sampler_is_defined(self):
        if self.approx in ['rff', 'rff+', 'lrff', 'lrff+', 'dmrff'] and self.rbf_sampler_ is None:
            self.rbf_sampler_ = RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=self.random_state)

    def fit(self, X, y):
        """Fits the classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Check if sampler is defined
        self.check_if_sampler_is_defined()

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.Xtrain_ = {}

        if self.approx == 'exact':
            for label in self.classes_:
                self.Xtrain_[label] = X[y == label]
        elif self.approx in ['rff', 'rff+', 'lrff', 'lrff+', 'dmrff']:
            Xt = self.rbf_sampler_.fit_transform(X)
            if self.normalize:
                norms = np.linalg.norm(Xt, axis=1)
                Xt = Xt / norms[:, np.newaxis]
            if self.approx in ['rff', 'rff+']:
                for label in self.classes_:
                    self.Xtrain_[label] = Xt[y == label]
            elif self.approx == 'dmrff':
                self.rff_mean = {}
                for label in self.classes_:
                    dm = np.einsum('...i, ...j->ij', Xt[y == label], 
                                   np.conj(Xt[y == label]), optimize='optimal')
                    w, v = np.linalg.eigh(dm)
                    # self.rff_mean[label] = np.linalg.cholesky(dmp)
                    wp = w[w>0]
                    num_p = wp.shape[0]
                    self.rff_mean[label] = np.matmul(v[:, -num_p:],np.diag(np.sqrt(wp)))
            else:
                self.rff_mean = {}
                # mean calculation of rff for each class
                for label in self.classes_:
                    self.rff_mean[label] = np.mean(Xt[y == label], axis=0)
        else:
            raise Exception(f"Invalid approximation method:{self.approx}")

        # Returns the classifier
        return self

    def predict(self, X):
        """ Performs classification.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        P = self.predict_proba(X)
        idxs = np.argmax(P, axis=1)
        return self.classes_[idxs]

    def predict_proba(self, X):
        """
        Return probability estimates for the test vectors X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        P : array-like (n_samples, n_classes)
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        """
        check_is_fitted(self)
        K = {}
        if self.approx == 'exact':
            for label in self.classes_:
                K[label] = rbf_kernel(X, self.Xtrain_[label], gamma=self.gamma)
        elif self.approx in ['rff', 'rff+', 'lrff', 'lrff+', 'dmrff']:
            Xt = self.rbf_sampler_.transform(X)
            if self.normalize == True:
                norms = np.linalg.norm(Xt, axis=1)
                Xt = Xt / norms[:, np.newaxis]
            for label in self.classes_:
                if self.approx == 'rff' or self.approx == 'rff+':
                    K[label] = np.matmul(Xt, self.Xtrain_[label].T)

                if self.approx == 'lrff' or self.approx == 'lrff+':
                    K[label] = np.matmul(Xt, self.rff_mean[label].T)
                if self.approx in ('rff', 'rff+', 'lrff', 'lrff+') and self.square_dot:
                    K[label] = np.power(K[label], 2)
                if self.approx == 'rff+' or self.approx == 'lrff+':
                    # K[label] = np.abs(K[label])
                    K[label] = np.maximum(K[label], 0.0000001)
                if self.approx == 'dmrff':
                    proj = np.matmul(Xt, self.rff_mean[label])
                    #K[label] = np.einsum('...i,...i', proj, np.conj(proj))
                    K[label] = np.linalg.norm(proj, axis=1) ** 2
        else:
            raise Exception(f"Invalid approximation method:{self.approx}")
        eps = 0.00001
        if self.approx in ['lrff', 'lrff+', 'dmrff']:
            sums = np.stack([K[label] for label in self.classes_], axis=1)
            sums = sums - sums.min() if sums.min()<0 else sums
            probs = sums / (np.sum(sums, axis=1)+eps)[:, np.newaxis]
        else:
            sums = np.stack([np.sum(K[label], axis=1) for label in self.classes_], axis=1)
            sums = sums - sums.min() if sums.min()<0 else sums
            probs = sums / (np.sum(sums, axis=1)+eps)[:, np.newaxis]
        return probs


class KDEstimatorRF(BaseEstimator):
    """ Kernel Density with Random Features
    TODO: Implement Kernel Density Estimation using RF 
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    """

    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.ones(X.shape[0], dtype=np.int64)


class TemplateTransformer(TransformerMixin, BaseEstimator):
    """ An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return np.sqrt(X)
