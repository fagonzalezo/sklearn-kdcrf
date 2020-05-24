import pytest

from sklearn.utils.estimator_checks import check_estimator

from .._kdclassifier import KDClassifierRF

@pytest.mark.parametrize(
    "Estimator", [KDClassifierRF]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
