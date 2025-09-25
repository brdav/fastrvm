#pragma once

#include <armadillo>

#include "sparse_bayes.hpp"

namespace sparse_bayes {

// Minimal result for Laplace posterior mode finder: returns posterior mean
// and the upper-triangular Cholesky factor of the Hessian at the mode.
struct LaplaceModeResult {
  arma::vec mu;
  arma::mat U;  // upper-triangular Cholesky factor of the Hessian
};

// Find the posterior mode (Laplace approximation) for non-Gaussian
// likelihoods. This function only returns the posterior mean (mu) and the
// upper-triangular Cholesky factor U.
LaplaceModeResult laplace_posterior_mode(const arma::mat &basis,
                                         const arma::vec &targets,
                                         const arma::vec &mu_init,
                                         const arma::vec &alpha,
                                         Likelihood likelihood);

}  // namespace sparse_bayes
