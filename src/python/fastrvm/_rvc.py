# Copyright 2025 brdav

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import warnings
from typing import Callable, Literal, cast

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _sparsebayes_bindings


class RVC(ClassifierMixin, BaseEstimator):
    """Relevance Vector Classification (RVC).

    A scikit-learn style estimator that wraps a C++ SparseBayes implementation.

    Example:
        sb = RVC()
        sb.fit(X, y)

    Attributes:
        n_features_in_ (int): Number of features seen during fit.
        relevance_ (ndarray): Indices of relevance vectors selected by the model.
        relevance_vectors_ (ndarray): Relevance vectors (subset of training X).
        dual_coef_ (ndarray): Dual coefficients for the relevance vectors.
        coef_ (ndarray): Weights assigned to the features when using a linear kernel.
        intercept_ (ndarray): Intercept term (0.0 if fit_intercept=False).
        alpha_ (ndarray): Precision (inverse variance) for the weights.
        n_relevance_ (ndarray): Number of relevance vectors.
        n_iter_ (ndarray): Number of iterations run by the SparseBayes solver.
        fit_status_ (int): Status code returned by the SparseBayes solver.
        scores_ (ndarray): Log marginal likelihood values at each iteration. Only available for binary classifier.
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
        n_jobs: int | None = None,
        prioritize_addition: bool = False,
        prioritize_deletion: bool = True,
    ):
        """Initialize an RVC estimator.

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
            n_jobs: Number of CPU cores to use for kernel computations and multiclass classification.
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
        self.n_jobs = n_jobs
        self.prioritize_addition = prioritize_addition
        self.prioritize_deletion = prioritize_deletion

    def fit(self, X: NDArray, y: NDArray) -> "RVC":
        """Fit the RVC model to training data.

        Args:
            X: Training input array of shape (n_samples, n_features).
            y: Target values of shape (n_samples,) or (n_samples, 1).

        Returns:
            self: Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        X = cast(NDArray, X)
        y = cast(NDArray, y)

        check_classification_targets(y)
        self.classes_ = np.unique(y)

        if len(self.classes_) < 2:
            raise ValueError("RVC requires more than one class in the data")

        self._backend_failure = False

        # Multiclass: one-vs-rest (train one binary RVC per class)
        if len(self.classes_) > 2 and not getattr(self, "_no_multiclass", False):
            self._label_binarizer = LabelBinarizer()
            Y = self._label_binarizer.fit_transform(y)
            # Ensure Y is 2-d (LabelBinarizer returns 1-d for binary)
            if Y.ndim == 1:
                Y = Y[:, None]

            # Train one cloned estimator per class. We clone `self` and set
            # a flag so the clone doesn't try to create its own multiclass
            # wrappers when its `fit` is called.
            def _train_one(estimator, X_train, y_binary):
                est = clone(estimator)
                # Prevent clone from doing OvR again
                setattr(est, "_no_multiclass", True)
                est.fit(X_train, y_binary)
                return est

            self._estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_train_one)(self, X, Y[:, i]) for i in range(Y.shape[1])
            )

            # Exit early if any estimator failed in the backend
            if any(est._backend_failure for est in self._estimators):
                self._backend_failure = True
                return self

            # Aggregate fitted attributes from the per-class estimators
            self._aggregate_estimators(X)
            return self

        self._gamma = self._get_gamma(X)

        # Compute the kernel matrix for the binary (non-multiclass) case
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
            _sparsebayes_bindings.Likelihood.Bernoulli,
            self.max_iter,
            self.fit_intercept,
            self.verbose,
            self.prioritize_addition,
            self.prioritize_deletion,
        )
        # It expects Phi and y_zero_one, where:
        # - Phi is the design matrix (one column per basis function)
        # - y_zero_one are binary target values (0 or 1)
        y_zero_one = np.where(y == self.classes_[0], 0, 1)
        try:
            result = _sparse_bayes.inference(Phi, y_zero_one)
        except Exception as e:
            warnings.warn(f"SparseBayes backend failed during inference: {e}")
            self._backend_failure = True
            return self

        self.n_iter_ = np.array([result["n_iter"]])
        self.fit_status_ = result["status"]
        self.scores_ = result["log_marginal_likelihood_trace"]
        self.alpha_ = result["alpha"].T  # shape (1, n_relevance)

        mean = result["mean"]
        relevant_idx = result["relevant_idx"]
        alpha = result["alpha"]

        if self.fit_intercept:
            self.relevance_ = relevant_idx[:-1].ravel()
            self.dual_coef_ = mean[:-1].T
            self.intercept_ = mean[-1]
            self.alpha_ = alpha[:-1].T
        else:
            self.relevance_ = relevant_idx.ravel()
            self.dual_coef_ = mean.T
            self.intercept_ = np.array([0.0])
            self.alpha_ = alpha.T

        self.relevance_vectors_ = X[self.relevance_]
        self.n_relevance_ = np.array([len(self.relevance_)])

        return self

    def decision_function(self, X: NDArray) -> NDArray:
        """Compute the decision function (logits) for the samples in X.

        Args:
            X: Input samples, shape (n_samples, n_features).

        Returns:
            1-D array of shape (n_samples,) with decision function values.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        # Multiclass (OvR): delegate to trained binary estimators
        if len(self.classes_) > 2:
            scores = np.column_stack(
                [est.decision_function(X) for est in self._estimators]
            )
            return scores

        # Binary / single estimator path
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

        logits = (Phi @ self.dual_coef_.T).ravel()

        if self.fit_intercept:
            logits += self.intercept_

        return logits

    def predict(self, X: NDArray) -> NDArray:
        """Predict using the trained relevance vector model. For the
        one-class setting, the model returns 0 or 1.

        Args:
            X: Input samples, shape (n_samples, n_features).

        Returns:
            1-D array of shape (n_samples,) with predicted values.
        """
        logits = self.decision_function(X)

        if len(self.classes_) > 2:
            idx = np.argmax(logits, axis=1)
            return self._label_binarizer.classes_[idx]

        y_pred = np.where(logits > 0, self.classes_[1], self.classes_[0])
        return y_pred

    def predict_proba(self, X: NDArray) -> NDArray:
        """Predict class probabilities using the trained relevance vector model.

        Args:
            X: Input samples, shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.
        """
        logits = self.decision_function(X)

        if len(self.classes_) > 2:
            probs = softmax(logits, axis=1)
            return probs

        probs = expit(logits)
        return np.column_stack((1 - probs, probs))

    def score(self, X: NDArray, y: NDArray) -> float:
        """Return the accuracy of the prediction.

        Args:
            X: Test samples.
            y: True values for X.

        Returns:
            Accuracy as a float.
        """
        check_is_fitted(self)
        X, y = validate_data(self, X, y, reset=False)

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

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

    def _aggregate_estimators(self, X: NDArray) -> None:
        """Aggregate per-class estimators into top-level attributes.

        This builds a global union of relevance indices, aligns per-class
        dual coefficients and alphas into matrices (dense), and extracts
        the corresponding relevance_vectors_ from X. Designed to be
        called after self._estimators is populated.
        """
        self.n_iter_ = np.concatenate([est.n_iter_ for est in self._estimators])
        self.fit_status_ = int(any(est.fit_status_ for est in self._estimators))

        # Aggregate relevance indices
        rels = [est.relevance_ for est in self._estimators]
        if all(r.size == 0 for r in rels):
            self.relevance_ = np.empty((0,), dtype=int)
        else:
            self.relevance_ = np.unique(
                np.concatenate([r for r in rels if r.size > 0])
            ).ravel()

        n_classes = len(self._estimators)
        n_rel_global = len(self.relevance_)

        pos_map = {int(v): i for i, v in enumerate(self.relevance_)}

        dual_mat = np.zeros((n_classes, n_rel_global), dtype=float)
        alpha_mat = np.zeros((n_classes, n_rel_global), dtype=float)

        for i, est in enumerate(self._estimators):
            if est.relevance_.size == 0:
                continue
            positions = [pos_map[int(v)] for v in est.relevance_]

            coef = est.dual_coef_.ravel()
            if coef.size == 0:
                continue
            if coef.size != len(positions):
                L = min(coef.size, len(positions))
                coef = coef[:L]
                positions = positions[:L]

            dual_mat[i, positions] = coef

            a = est.alpha_.ravel()
            if a.size == coef.size:
                alpha_mat[i, positions] = a
            elif a.size == 1:
                alpha_mat[i, positions] = a.item()

        self.dual_coef_ = dual_mat
        self.alpha_ = alpha_mat

        self.intercept_ = np.concatenate([est.intercept_ for est in self._estimators])
        self.n_relevance_ = np.array([len(est.relevance_) for est in self._estimators])

        if len(self.relevance_) > 0:
            self.relevance_vectors_ = X[self.relevance_]
        else:
            self.relevance_vectors_ = np.empty((0, X.shape[1]))

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
