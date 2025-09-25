// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "laplace.hpp"

#include <ensmallen.hpp>

#include "logistic.hpp"
#include "sparse_bayes.hpp"

namespace sparse_bayes {

LaplaceModeResult LaplacePosteriorMode(const arma::mat &basis,
                                       const arma::vec &targets,
                                       const arma::vec &mu_init,
                                       const arma::vec &alpha,
                                       Likelihood likelihood) {
  LaplaceModeResult out;

  if (likelihood == Likelihood::kBernoulli) {
    LogisticRegressionWithPriorFunction f(basis, targets, alpha);

    ens::L_BFGS lbfgs;
    arma::mat params = arma::mat(mu_init);
    lbfgs.Optimize(f, params);

    out.mu = arma::vec(params);

    // Compute Hessian's Cholesky factor
    arma::vec logits = basis * out.mu;
    arma::vec b_vec = arma::exp(
        -arma::abs(logits) - 2.0 * arma::log1p(arma::exp(-arma::abs(logits))));

    arma::mat basis_b = basis.each_col() % b_vec;
    arma::mat H = basis_b.t() * basis + arma::diagmat(alpha);

    out.U = arma::chol(H);

  } else {
    throw std::invalid_argument("LaplacePosteriorMode got unknown likelihood");
  }

  return out;
}

}  // namespace sparse_bayes
