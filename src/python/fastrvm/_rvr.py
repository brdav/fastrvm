# Copyright 2025 brdav

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _sparsebayes_bindings


class RVR(RegressorMixin, BaseEstimator):
    """Relevance Vector Regression (RVR).

    A scikit-learn style estimator that wraps a C++ SparseBayes implementation.

    Example:
        sb = RVR()
        sb.fit(X, y)

    Attributes:
        n_features_in_ (int): Number of features seen during fit.
        relevance_ (ndarray of shape (n_relevance,)): Indices of relevance vectors selected by the model.
        relevance_vectors_ (ndarray of shape (n_relevance, n_features)): Relevance vectors (subset of training X).
        dual_coef_ (ndarray of shape (1, n_relevance)): Dual coefficients for the relevance vectors.
        coef_ (ndarray of shape (1, n_features)): Weights assigned to the features when using a linear kernel.
        intercept_ (float): Intercept term (0.0 if fit_intercept=False).
        covariance_ (ndarray of shape (n_relevance, n_relevance)): Posterior covariance of the weights for relevance vectors.
        alpha_ (ndarray of shape (1, n_relevance)): Precision (inverse variance) for the weights.
        beta_ (float): Noise precision (inverse noise variance).
        n_relevance_ (int): Number of relevance vectors.
        n_iter_ (int): Number of iterations run by the SparseBayes solver.
        fit_status_ (int): Status code returned by the SparseBayes solver.
        scores_ (ndarray of shape (n_iter,)): Log marginal likelihood values at each iteration.
    """

    def __init__(
        self,
        kernel: (
            Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]
            | Callable[[NDArray, NDArray], NDArray]
        ) = "rbf",
        degree: int = 3,
        gamma: Literal["scale", "auto"] | float = "scale",
        coef0: float = 0.0,
        fit_intercept: bool = False,
        max_iter: int = 10000,
        verbose: bool = False,
        noise_fixed: bool = False,
        noise_std_init: float | None = None,
        n_jobs: int | None = None,
        prioritize_addition: bool = False,
        prioritize_deletion: bool = True,
    ):
        """Initialize an RVR estimator.

        Args:
            kernel: Kernel specification passed to sklearn.metrics.pairwise.pairwise_kernels.
                Allowed strings: "linear", "poly", "rbf", "sigmoid", "precomputed",
                or a callable kernel(X, Y) -> ndarray.
            degree: Degree for the polynomial kernel (if used).
            gamma: Kernel coefficient for rbf/poly/sigmoid. If "scale" or "auto"
                the value is computed from training data. Can also be a float.
            coef0: Independent term in kernel function. Used for poly/sigmoid.
            fit_intercept: If True, an additional bias basis is used and an
                intercept is learned.
            max_iter: Maximum number of iterations for the SparseBayes solver.
            verbose: Verbosity flag forwarded to the backend.
            noise_fixed: If True, the observation noise precision is kept fixed.
            noise_std_init: Initial noise standard deviation passed to the backend.
            n_jobs: Number of CPU cores to use for kernel computations.
            prioritize_addition: Backend heuristic flag.
            prioritize_deletion: Backend heuristic flag.
        """

        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.verbose = verbose
        self.noise_fixed = noise_fixed
        self.noise_std_init = noise_std_init
        self.n_jobs = n_jobs
        self.prioritize_addition = prioritize_addition
        self.prioritize_deletion = prioritize_deletion

    def fit(self, X: NDArray, y: NDArray) -> "RVR":
        """Fit the RVR model to training data.

        Args:
            X: Training input array of shape (n_samples, n_features).
            y: Target values of shape (n_samples,) or (n_samples, 1).

        Returns:
            self: Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        X = cast(NDArray, X)
        y = cast(NDArray, y)

        # Compute and store gamma value if needed
        self._gamma = self._get_gamma(X)

        # Precompute the kernel matrix
        Phi = pairwise_kernels(
            X,
            metric=self.kernel,
            filter_params=True,
            n_jobs=self.n_jobs,
            gamma=self._gamma,
            degree=self.degree,
            coef0=self.coef0,
        )  # shape (n_samples, n_samples)

        # Call the C++ backend
        _sparse_bayes = _sparsebayes_bindings.SparseBayes(
            _sparsebayes_bindings.Likelihood.Gaussian,
            self.max_iter,
            self.fit_intercept,
            self.verbose,
            self.prioritize_addition,
            self.prioritize_deletion,
            self.noise_fixed,
            self.noise_std_init,
        )
        # It takes in Phi and y, where Phi is the design matrix
        # (one column per basis function) and y is the target values.
        try:
            result = _sparse_bayes.inference(Phi, y)
        except Exception as e:
            raise RuntimeError("SparseBayes backend failed during inference") from e

        self.n_iter_ = result["n_iter"]
        self.fit_status_ = result["status"]
        self.scores_ = result["log_marginal_likelihood_trace"]
        self.beta_ = result["beta"]

        mean = result["mean"]
        covariance = result["covariance"]
        relevant_idx = result["relevant_idx"]
        alpha = result["alpha"]

        if self.fit_intercept:
            self.relevance_ = relevant_idx[:-1].ravel()
            self.covariance_ = covariance[:-1, :-1]
            self.dual_coef_ = mean[:-1].reshape(1, -1)
            self.intercept_ = float(mean[-1])
            self.alpha_ = alpha[:-1].reshape(1, -1)
        else:
            self.relevance_ = relevant_idx.ravel()
            self.covariance_ = covariance
            self.dual_coef_ = mean.reshape(1, -1)
            self.intercept_ = 0.0
            self.alpha_ = alpha.reshape(1, -1)

        self.relevance_vectors_ = X[self.relevance_]
        self.n_relevance_ = len(self.relevance_)

        return self

    def predict(
        self, X: NDArray, return_std: bool = False
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Predict using the trained relevance vector model.

        Args:
            X: Input samples, shape (n_samples, n_features).
            return_std: If True, also return the predictive standard deviation for
                each sample. Defaults to False.

        Returns:
            If ``return_std`` is False (default), returns a 1-D array of shape
            (n_samples,) with predicted values. If True, returns a tuple
            ``(y_pred, y_std)`` where ``y_std`` is a 1-D array with predictive
            standard deviations.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = cast(NDArray, X)

        Phi = pairwise_kernels(
            X,
            self.relevance_vectors_,
            metric=self.kernel,
            filter_params=True,
            n_jobs=self.n_jobs,
            gamma=self._gamma,
            degree=self.degree,
            coef0=self.coef0,
        )

        y_pred = (Phi @ self.dual_coef_.T).ravel()

        if self.fit_intercept:
            y_pred += self.intercept_

        if return_std:
            y_std = np.sqrt(
                np.einsum("ij,ij->i", Phi @ self.covariance_, Phi) + 1.0 / self.beta_
            )
            return y_pred, y_std

        return y_pred

    def score(self, X: NDArray, y: NDArray) -> float:
        """Return the coefficient of determination R^2 of the prediction.

        Args:
            X: Test samples.
            y: True values for X.

        Returns:
            R^2 score as a float.
        """
        check_is_fitted(self)
        X, y = validate_data(self, X, y, reset=False)
        X = cast(NDArray, X)
        y = cast(NDArray, y)

        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def _get_gamma(self, X: NDArray) -> float:
        """Return the numeric gamma value used by kernel functions.

        Args:
            X: Training input array used to compute data-dependent gamma when
                ``self.gamma`` is "scale" or "auto". Expected shape is
                (n_samples, n_features).

        Returns:
            Float value to be passed to pairwise kernel functions.
        """
        if self.gamma == "scale":
            return 1.0 / (X.shape[1] * X.var()) if X.var() > 0 else 1.0
        elif self.gamma == "auto":
            return 1.0 / X.shape[1]
        else:
            assert isinstance(self.gamma, float)
            return self.gamma

    @property
    def coef_(self) -> NDArray:
        """Weights assigned to the features when ``kernel="linear"``.

        Returns:
            Array of shape (n_features,) with the weights in the primal space.
        """
        check_is_fitted(self)

        if self.kernel != "linear":
            raise AttributeError("coef_ is only available when using a linear kernel")

        return self.dual_coef_ @ self.relevance_vectors_
