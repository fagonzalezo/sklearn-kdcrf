import pytest
import numpy as np

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from .._kdclassifier import KDClassifierRF


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_KDClassifierRF(data):
    X, y = data
    clf = KDClassifierRF()
    assert hasattr(clf, 'approx')
    assert hasattr(clf, 'normalize')
    assert hasattr(clf, 'gamma')
    assert hasattr(clf, 'n_components')

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'Xtrain_')
    if clf.approx == 'rff':
        assert hasattr(clf, 'rbf_sampler_')
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
