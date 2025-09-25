#include <armadillo>

// Numerically stable sigmoid function
static arma::vec sigmoid(const arma::vec &z) {
  arma::vec result(z.n_elem);
  arma::uvec pos = arma::find(z >= 0);
  arma::uvec neg = arma::find(z < 0);
  if (pos.n_elem > 0) {
    arma::vec zp = z.elem(pos);
    result.elem(pos) = 1.0 / (1.0 + arma::exp(-zp));
  }
  if (neg.n_elem > 0) {
    arma::vec zn = z.elem(neg);
    arma::vec expzn = arma::exp(zn);
    result.elem(neg) = expzn / (1.0 + expzn);
  }
  return result;
}

class LogisticRegressionWithPriorFunction {
 public:
  LogisticRegressionWithPriorFunction(const arma::mat &X, const arma::vec &t,
                                      const arma::vec &alpha)
      : X_(X),
        t_mat_(arma::mat(t)),         // N x 1
        alpha_mat_(arma::mat(alpha))  // M x 1
  {}

  // Return the objective function for model parameters w, while
  // computing its gradient and storing it in g.
  double EvaluateWithGradient(const arma::mat &w, arma::mat &g) {
    // Compute and store some quantities
    arma::mat logits = X_ * w;
    arma::mat alpha_w = alpha_mat_ % w;

    // Compute the gradient
    arma::mat y = sigmoid(logits);
    g = X_.t() * (y - t_mat_) + alpha_w;

    // Negative log-likelihood using a numerically-stable formulation
    arma::vec softplus(logits.n_elem);
    arma::uvec pos = arma::find(logits > 0);
    arma::uvec neg = arma::find(logits <= 0);
    if (pos.n_elem > 0) {
      arma::vec zp = logits.elem(pos);
      // softplus(z) = z + log1p(exp(-z)) for z>0
      softplus.elem(pos) = zp + arma::log1p(arma::exp(-zp));
    }
    if (neg.n_elem > 0) {
      arma::vec zn = logits.elem(neg);
      // softplus(z) = log1p(exp(z)) for z<=0
      softplus.elem(neg) = arma::log1p(arma::exp(zn));
    }
    return arma::accu(softplus - t_mat_ % logits) + arma::dot(w, alpha_w) / 2.0;
  }

 private:
  // The data matrix and target vector.
  const arma::mat &X_;   // N x M
  arma::mat t_mat_;      // N
  arma::mat alpha_mat_;  // M
};
