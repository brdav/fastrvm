import numpy as np
import numpy.testing as npt
from sklearn.datasets import make_regression
from sklearn.utils.estimator_checks import check_estimator

from fastrvm import RVR


def _make_regression_data(seed=0):
    res = make_regression(
        n_samples=50, n_features=5, n_informative=3, noise=0.1, random_state=seed
    )
    # make_regression may return (X, y) or (X, y, coef) depending on sklearn version
    if isinstance(res, tuple) and len(res) == 3:
        X = res[0]
        y = res[1]
    else:
        X = res[0]
        y = res[1]
    return X, y


def test_sklearn():
    check_estimator(RVR())


def test_rvr_fit_intercept_false():
    X, y = _make_regression_data(seed=0)
    rvr = RVR(kernel="rbf", fit_intercept=False, max_iter=10000, n_jobs=1)
    rvr.fit(X, y)

    # check fit_status_ to ensure convergence
    assert rvr.fit_status_ == 0

    assert rvr.n_iter_ == 256
    assert rvr.n_relevance_ == 21
    npt.assert_array_equal(
        rvr.relevance_[:11],
        np.asarray([1, 6, 7, 9, 10, 12, 16, 17, 19, 21, 22]),
    )
    npt.assert_almost_equal(
        rvr.dual_coef_.ravel()[:3],
        [48.0830672023, -24.7453919352, 94.3074349441],
        decimal=6,
    )


def test_rvr_fit_intercept_true():
    X, y = _make_regression_data(seed=0)
    rvr = RVR(kernel="rbf", fit_intercept=True, max_iter=10000, n_jobs=1)
    rvr.fit(X, y)

    # check fit_status_ to ensure convergence
    assert rvr.fit_status_ == 0

    assert rvr.n_iter_ == 256
    assert rvr.n_relevance_ == 21
    npt.assert_array_equal(rvr.relevance_[:6], np.asarray([1, 6, 7, 9, 10, 12]))
    npt.assert_array_almost_equal(rvr.intercept_, np.asarray([15.33254665]))
