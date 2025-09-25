import numpy as np
import numpy.testing as npt
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

from fastrvm import RVC


def test_sklearn():
    check_estimator(RVC())


def test_rvc_binary_fit_intercept_false():
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=1,
    )
    rvc = RVC(kernel="rbf", fit_intercept=False, max_iter=10000, n_jobs=1)
    rvc.fit(X, y)

    # check fit_status_ to ensure convergence
    assert rvc.fit_status_ == 0

    npt.assert_array_equal(rvc.n_iter_, np.asarray([108]))
    npt.assert_array_equal(rvc.n_relevance_, np.asarray([6]))
    npt.assert_array_equal(rvc.relevance_, np.asarray([0, 14, 15, 19, 23, 46]))
    npt.assert_array_almost_equal(
        rvc.dual_coef_.ravel()[:3],
        np.asarray([-1.0331734247, -1.136671976, -5.2816171146]),
        decimal=6,
    )


def test_rvc_binary_fit_intercept_true():
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=1,
    )
    rvc = RVC(kernel="rbf", fit_intercept=True, max_iter=10000, n_jobs=1)
    rvc.fit(X, y)

    # check fit_status_ to ensure convergence
    assert rvc.fit_status_ == 0

    npt.assert_array_equal(rvc.n_iter_, np.asarray([38]))
    npt.assert_array_equal(rvc.n_relevance_, np.asarray([5]))
    npt.assert_array_equal(rvc.relevance_, np.asarray([15, 19, 36, 51, 54]))
    npt.assert_array_almost_equal(rvc.intercept_, np.asarray([-1.65365439]), decimal=6)


def test_rvc_multiclass_fit_intercept_false():
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=2,
    )
    rvc = RVC(kernel="rbf", fit_intercept=False, max_iter=10000, n_jobs=1)
    rvc.fit(X, y)

    # check fit_status_ to ensure convergence
    assert rvc.fit_status_ == 0

    npt.assert_array_equal(rvc.n_iter_, np.asarray([70, 114, 40]))
    npt.assert_array_equal(rvc.n_relevance_, np.asarray([6, 5, 5]))
    npt.assert_array_equal(
        rvc.relevance_,
        np.asarray([7, 9, 23, 29, 32, 33, 38, 40, 52, 61, 68, 69, 72, 76]),
    )
    assert rvc.dual_coef_.shape == (3, 14)


def test_rvc_multiclass_fit_intercept_true():
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=2,
    )
    rvc = RVC(kernel="rbf", fit_intercept=True, max_iter=10000, n_jobs=1)
    rvc.fit(X, y)

    # check fit_status_ to ensure convergence
    assert rvc.fit_status_ == 0

    npt.assert_array_equal(rvc.n_iter_, np.asarray([20, 27, 45]))
    npt.assert_array_equal(rvc.n_relevance_, np.asarray([4, 4, 6]))
    npt.assert_array_equal(
        rvc.relevance_, np.asarray([21, 22, 23, 32, 38, 51, 52, 56, 63, 66, 72, 73])
    )
    assert rvc.dual_coef_.shape == (3, 12)
    assert rvc.intercept_.shape == (3,)
