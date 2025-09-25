// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef LAPACE_HPP
#define LAPACE_HPP

#include <armadillo>

#include "sparse_bayes.hpp"

namespace sparse_bayes {

/// Minimal result for Laplace posterior mode finder: returns posterior mean
/// and the upper-triangular Cholesky factor of the Hessian at the mode.
struct LaplaceModeResult {
  arma::vec mu;  ///< Posterior mean at the mode
  arma::mat U;   ///< Upper-triangular Cholesky factor of the Hessian
};

/// Find the posterior mode (Laplace approximation) for non-Gaussian
/// likelihoods. Returns posterior mean (`mu`) and the Cholesky factor (`U`).
LaplaceModeResult LaplacePosteriorMode(const arma::mat &basis,
                                       const arma::vec &targets,
                                       const arma::vec &mu_init,
                                       const arma::vec &alpha,
                                       Likelihood likelihood);

}  // namespace sparse_bayes

#endif  // LAPACE_HPP
