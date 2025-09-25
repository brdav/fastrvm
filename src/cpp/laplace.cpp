
#include "laplace.hpp"

#include <ensmallen.hpp>

#include "logistic.hpp"
#include "sparse_bayes.hpp"

namespace sparse_bayes {

LaplaceModeResult laplace_posterior_mode(const arma::mat &basis,
                                         const arma::vec &targets,
                                         const arma::vec &mu_init,
                                         const arma::vec &alpha,
                                         Likelihood likelihood) {
  LaplaceModeResult out;
  if (likelihood == Likelihood::Bernoulli) {
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
    throw std::invalid_argument(
        "laplace_posterior_mode called for unknown likelihood");
  }

  return out;
}

}  // namespace sparse_bayes
