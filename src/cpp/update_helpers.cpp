#include "sparse_bayes.hpp"

namespace sparse_bayes {

// Update statistics after an action. For non-Gaussian likelihoods this may
// recompute full statistics; for Gaussian we update via cheaper formulas.
void SparseBayes::update_after_action(
    ModelState &state, Action selected_action, const arma::mat &basis,
    const arma::vec &targets_in, const arma::mat &basis_targets, double &logML,
    arma::vec &Gamma, double &delta_log_marginal, int &ll_update_count) {
  if (selected_action == Action::Reestimate || selected_action == Action::Add ||
      selected_action == Action::Delete) {
    if (likelihood_ == Likelihood::Gaussian) {
      state.S_out = state.S_in;
      state.Q_out = state.Q_in;
      arma::vec tmp =
          state.alpha / (state.alpha - state.S_in.elem(state.used_basis_idx));
      state.S_out.elem(state.used_basis_idx) =
          state.S_in.elem(state.used_basis_idx) % tmp;
      state.Q_out.elem(state.used_basis_idx) =
          state.Q_in.elem(state.used_basis_idx) % tmp;
      state.relevance_factor = arma::square(state.Q_out) - state.S_out;

      Gamma = arma::ones<arma::vec>(state.alpha.n_elem) -
              state.alpha % arma::diagvec(state.Sigma);

      // Recompute basis_Phi/b_basis_Phi
      state.b_basis_Phi = state.beta * state.basis_Phi;

      if (delta_log_marginal < 0) {
        printf("WARNING: decrease in log likelihood! (%.6f)\n",
               delta_log_marginal);
      }
    } else {
      FullStatisticsResult full_stat = compute_full_statistics(
          basis, state.Phi, targets_in, state.used_basis_idx, state.basis_Phi,
          basis_targets, state.weights, state.alpha, state.beta);
      state.weights = full_stat.weights;
      state.Sigma = full_stat.Sigma;
      state.S_in = full_stat.S_in;
      state.Q_in = full_stat.Q_in;
      state.S_out = full_stat.S_out;
      state.Q_out = full_stat.Q_out;
      state.relevance_factor = full_stat.relevance_factor;
      state.b_basis_Phi = full_stat.b_basis_Phi;
      state.b_vec = full_stat.b_vec;

      Gamma = full_stat.Gamma;
      // we update logML below
      delta_log_marginal = full_stat.logML - logML;
    }
    logML = logML + delta_log_marginal;
    ll_update_count++;
  }
}

// Update beta (Gaussian noise precision) when appropriate. If beta changes
// significantly, recompute full statistics and possibly defer termination by
// modifying selected_action.
void SparseBayes::maybe_update_beta(ModelState &state, const arma::mat &basis,
                                    const arma::vec &targets_in,
                                    const arma::mat &basis_targets,
                                    double &logML, arma::vec &Gamma,
                                    int &ll_update_count, int iter,
                                    Action &selected_action) {
  if (likelihood_ != Likelihood::Gaussian)
    throw std::logic_error(
        "maybe_update_beta called for non-Gaussian likelihood");

  // If fixed_noise_ is true, never update beta
  if (fixed_noise_) {
    return;
  }

  // Determine whether we should attempt a beta update this iteration
  if (!(selected_action == Action::Terminate || iter < BETA_WARMUP_ITER ||
        (iter + 1) % BETA_UPDATE_FREQUENCY == 0)) {
    return;
  }

  arma::uword N = basis.n_rows;

  double new_beta = state.beta;
  arma::vec y = state.Phi * state.weights;
  arma::vec e = targets_in - y;
  double ED = arma::dot(e, e);

  new_beta = (static_cast<double>(N) - arma::sum(Gamma)) / ED;
  // Bound the change in beta to avoid instability
  new_beta = std::min(new_beta, BETA_MAX_FACTOR / arma::var(targets_in));
  double delta_log_beta = std::log(new_beta) - std::log(state.beta);

  state.beta = new_beta;

  if (std::abs(delta_log_beta) > MIN_DELTA_LOG_BETA) {
    FullStatisticsResult full_stat = compute_full_statistics(
        basis, state.Phi, targets_in, state.used_basis_idx, state.basis_Phi,
        basis_targets, state.weights, state.alpha, state.beta);

    state.weights = full_stat.weights;
    state.Sigma = full_stat.Sigma;
    state.S_in = full_stat.S_in;
    state.Q_in = full_stat.Q_in;
    state.S_out = full_stat.S_out;
    state.Q_out = full_stat.Q_out;
    state.relevance_factor = full_stat.relevance_factor;
    state.b_basis_Phi = full_stat.b_basis_Phi;

    logML = full_stat.logML;
    Gamma = full_stat.Gamma;

    ll_update_count++;

    if (selected_action == Action::Terminate) {
      // If we were going to terminate, but beta changed significantly, continue
      selected_action = Action::NoiseOnly;
      if (verbose_) printf("Noise update (termination deferred)\n");
    }
  }
}

}  // namespace sparse_bayes
