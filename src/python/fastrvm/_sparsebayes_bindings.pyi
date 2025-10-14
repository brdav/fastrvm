# Copyright 2025 brdav

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import enum
from typing import Optional, TypedDict

import numpy as np
from numpy.typing import NDArray

__all__ = ["Likelihood", "SparseBayes"]

class Likelihood(enum.Enum):
    """Likelihood enum exported from the C++ bindings."""

    Gaussian: "Likelihood"
    Bernoulli: "Likelihood"

class SparseBayesResult(TypedDict):
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    relevant_idx: NDArray[np.int_]
    alpha: NDArray[np.float64]
    beta: float
    n_iter: int
    status: int
    log_marginal_likelihood_trace: NDArray[np.float64]

class SparseBayes:
    """Binding wrapper around the C++ SparseBayes core.

    The real implementation is in a compiled extension module. This stub
    provides type information for editors and type checkers.
    """

    def __init__(
        self,
        likelihood: Likelihood = Likelihood.Gaussian,
        iterations: int = 1000,
        use_bias: bool = False,
        verbose: bool = False,
        prioritize_addition: bool = False,
        prioritize_deletion: bool = True,
        fixed_noise: bool = False,
        noise_std: Optional[float] = None,
    ) -> None: ...
    def inference(self, basis: NDArray, targets: NDArray) -> SparseBayesResult: ...
