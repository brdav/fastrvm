#include "sparse_bayes.hpp"

namespace sparse_bayes {

// Re-estimate a basis function's alpha
void SparseBayes::reestimate_basis(ModelState &state, arma::uword j,
                                   double new_alpha) {
  arma::vec s_j = state.Sigma.col(j);
  double delta_inv = 1.0 / (new_alpha - state.alpha(j));
  double sig_jj = state.Sigma(j, j);
  double kappa = 1.0 / (sig_jj + delta_inv);
  arma::vec tmp = kappa * s_j;
  arma::vec delta_weights = -state.weights(j) * tmp;

  if (likelihood_ == Likelihood::Gaussian) {
    arma::vec prod = state.b_basis_Phi * s_j;
    state.S_in = state.S_in + kappa * arma::square(prod);
    state.Q_in = state.Q_in - state.b_basis_Phi * delta_weights;
  }

  state.alpha(j) = new_alpha;
  state.weights = state.weights + delta_weights;
  state.Sigma = state.Sigma - tmp * s_j.t();
}

// Add a basis function 'nu' into the active set
void SparseBayes::add_basis_function(ModelState &state, arma::uword nu,
                                     double new_alpha, const arma::mat &basis,
                                     const arma::vec &Phi_vec) {
  arma::vec basis_Phi_vec;
  arma::vec b_Phi;
  if (likelihood_ == Likelihood::Gaussian) {
    basis_Phi_vec = basis.t() * Phi_vec;
    b_Phi = state.beta * Phi_vec;
  } else {
    b_Phi = Phi_vec % state.b_vec;
  }

  arma::uword M = state.Phi.n_cols;
  arma::vec tmp = ((b_Phi.t() * state.Phi) * state.Sigma).t();
  double s_ii = 1.0 / (new_alpha + state.S_in(nu));
  arma::vec s_i = -s_ii * tmp;                        // length M
  arma::mat Sigma_sub = state.Sigma - s_i * tmp.t();  // MxM

  double mu_i = s_ii * state.Q_in(nu);
  arma::vec top = -mu_i * tmp;
  arma::vec delta_weights = arma::join_vert(top, arma::vec{mu_i});

  if (likelihood_ == Likelihood::Gaussian) {
    state.basis_Phi.insert_cols(state.basis_Phi.n_cols, basis_Phi_vec);

    arma::vec mCi = state.beta * basis_Phi_vec - state.b_basis_Phi * tmp;
    state.S_in = state.S_in - s_ii * arma::square(mCi);
    state.Q_in = state.Q_in - mu_i * mCi;
  }

  state.Sigma = arma::mat(M + 1, M + 1);
  state.Sigma.submat(0, 0, M - 1, M - 1) = Sigma_sub;
  state.Sigma.submat(0, M, M - 1, M) = s_i;
  state.Sigma.submat(M, 0, M, M - 1) = s_i.t();
  state.Sigma(M, M) = s_ii;

  state.weights.insert_rows(state.weights.n_rows, arma::vec{0.0});
  state.weights = state.weights + delta_weights;

  state.alpha.insert_rows(state.alpha.n_elem, arma::vec{new_alpha});
  state.Phi.insert_cols(state.Phi.n_cols, Phi_vec);
  state.used_basis_idx.insert_rows(state.used_basis_idx.n_rows, arma::uvec{nu});
}

// Delete a basis function at interior index j
void SparseBayes::delete_basis_function(ModelState &state, arma::uword j) {
  double s_jj = state.Sigma(j, j);
  arma::vec s_j = state.Sigma.col(j);
  arma::vec tmp = s_j / s_jj;

  if (likelihood_ == Likelihood::Gaussian) {
    state.basis_Phi.shed_col(j);

    double mu_j = state.weights(j);
    arma::vec jPm = state.b_basis_Phi * s_j;
    state.S_in = state.S_in + arma::square(jPm) / s_jj;
    state.Q_in = state.Q_in + jPm * (mu_j / s_jj);
  }

  state.Sigma = state.Sigma - tmp * s_j.t();
  state.Sigma.shed_row(j);
  state.Sigma.shed_col(j);

  state.weights = state.weights - state.weights(j) * tmp;
  ;
  state.weights.shed_row(j);

  state.Phi.shed_col(j);
  state.alpha.shed_row(j);
  state.used_basis_idx.shed_row(j);
}

// Dispatch helper: perform the chosen action and update counters.
void SparseBayes::perform_action(ModelState &state, Action selected_action,
                                 arma::uword j, arma::uword nu,
                                 double new_alpha, const arma::mat &basis,
                                 const arma::vec &Phi_vec, int &add_count,
                                 int &delete_count, int &update_count) {
  if (selected_action == Action::Reestimate) {
    reestimate_basis(state, j, new_alpha);
    update_count++;
  } else if (selected_action == Action::Add) {
    add_basis_function(state, nu, new_alpha, basis, Phi_vec);
    add_count++;
  } else if (selected_action == Action::Delete) {
    delete_basis_function(state, j);
    delete_count++;
  }
}

}  // namespace sparse_bayes
