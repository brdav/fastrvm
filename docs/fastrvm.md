
# fastrvm — Reference

This short reference documents the public Python API exposed by the package: the
`RVR` (regression) and `RVC` (probabilistic classification) estimators. Both
implement a scikit-learn compatible estimator interface and wrap a high‑performance
C++ core implementing Tipping's sparse Bayesian learning algorithm.

The doc below is concise; consult the runtime docstrings (e.g. `help(RVR)`) for
the exact parameter defaults and full explanations.

## Shared constructor parameters

The two classes share several constructor parameters (check class docstrings
for exact defaults):

- `kernel`: 'linear' | 'poly' | 'rbf' | 'sigmoid' | 'precomputed' or a callable.
- `gamma`: float or `'scale'`/`'auto'` (data-dependent kernel coefficient).
- `degree`: int (polynomial degree for `poly`).
- `coef0`: float (independent term for `poly`/`sigmoid`).
- `fit_intercept`: bool (whether to learn an intercept/bias).
- `max_iter`: int (max greedy solver iterations).
- `verbose`: bool/int (backend verbosity).
- `n_jobs`: int | None (parallel kernel computations; `-1` = all cores).

Note: if `kernel='precomputed'` you must pass a Gram/design matrix to `fit`.

## RVR (Relevance Vector Regression)

Brief: regression estimator assuming a Gaussian likelihood.

Constructor highlights (see `src/python/fastrvm/_rvr.py`):

- `kernel`, `degree`, `gamma`, `coef0`, `fit_intercept`, `max_iter`,
  `verbose`, `noise_fixed`, `noise_std_init`, `n_jobs`, `prioritize_addition`,
  `prioritize_deletion`.

`noise_std_init` can be set if the noise standard deviation is known, and its optimization can be disabled with `noise_fixed=True`.

Public methods

- `fit(X, y)` — X: (n_samples, n_features) or precomputed design matrix; y: (n_samples,) or (n_samples, 1).
- `predict(X, return_std=False)` — returns y_pred (n_samples,) or (y_pred, y_std)
  where y_std is (n_samples,) when `return_std=True`.
- `score(X, y)` — R2 (float).

Fitted attributes (names, types, shapes)

- `n_features_in_` : int
- `relevance_` : ndarray[int], shape (n_relevance,) — indices into the training set.
- `relevance_vectors_` : ndarray[float], shape (n_relevance, n_features) — subset of training X.
- `dual_coef_` : ndarray[float], shape (1, n_relevance) — dual coefficients for selected bases.
- `coef_` : ndarray[float], shape (n_features,) — primal weights (only for linear kernel).
- `intercept_` : float (scalar) — intercept (0.0 if fit_intercept=False).
- `covariance_` : ndarray[float], shape (n_relevance, n_relevance) — posterior covariance of weights.
- `alpha_` : ndarray[float], shape (1, n_relevance) — precision(s) for the weights.
- `beta_` : float — observation noise precision (inverse variance).
- `n_relevance_` : int — number of relevance vectors (len(relevance_)).
- `n_iter_` : int — iterations performed by the backend.
- `fit_status_` : int — backend status code (0 typically = success).
- `scores_` : ndarray[float], shape (n_iter,) — log marginal likelihood trace.

## RVC (Relevance Vector Classification)

Brief: probabilistic classifier implemented via Bernoulli (binary) or
one-vs-rest (multiclass) binary classifiers.

Constructor highlights (see `src/python/fastrvm/_rvc.py`):

- `kernel`, `degree`, `gamma`, `coef0`, `fit_intercept`, `max_iter`,
  `verbose`, `n_jobs`, `prioritize_addition`, `prioritize_deletion`.

Public methods

- `fit(X, y)` — X: (n_samples, n_features) or precomputed; y: (n_samples,)
  The estimator uses one-vs-rest internally for multiclass problems.
- `decision_function(X)` — returns logits:
  - binary: 1-D array (n_samples,)
  - multiclass: 2-D array (n_samples, n_classes)
- `predict(X)` — class labels (n_samples,)
- `predict_proba(X)` — probabilities (n_samples, n_classes)
- `score(X, y)` — accuracy (float).

Fitted attributes (names, types, shapes)

- `classes_` : ndarray — unique class labels learned during `fit`.
- `relevance_` : ndarray[int], shape (n_relevance,) — union of relevance indices across classifiers (global set).
- `relevance_vectors_` : ndarray[float], shape (n_relevance, n_features) — corresponding training vectors.
- `dual_coef_` : ndarray[float], shape (n_classifiers, n_relevance)
  - For binary classification `n_classifiers == 1` and `dual_coef_.shape == (1, n_relevance)`.
  - For multiclass (OvR) `n_classifiers == n_classes`.
- `coef_` : ndarray[float], shape (n_classifiers, n_features) — primal weights when `kernel='linear'`.
- `intercept_` : ndarray[float], shape (n_classifiers,) — intercept for each classifier (binary case returns array([0.0]) if no intercept).
- `alpha_` : ndarray[float], shape (n_classifiers, n_relevance) — per-class precision(s) for the weights.
- `n_relevance_` : ndarray[int], shape (n_classifiers,) — number of relevance vectors per classifier (binary: array([k]) ).
- `n_iter_` : ndarray[int], shape (n_classifiers,) — iterations per classifier (binary: array([n_iter]) ).
- `fit_status_` : int — aggregated status (0 = success for all, non-zero if any classifier failed).
- `scores_` : ndarray[float] — log marginal likelihood trace (available only for binary classifiers).

Notes

- In the multiclass path the estimator clones itself and fits separate binary
  estimators in parallel. It is possible to do multiclass classification with RVMs
  using a multinomial likelihood, however, the size of the covariance matrix scales with
  n_classes, leading to very expensive matrix inversion.
