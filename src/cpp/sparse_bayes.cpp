
#include "sparse_bayes.hpp"

#include <algorithm>
#include <ensmallen.hpp>
#include <iostream>
#include <limits>
#include <optional>

#include "laplace.hpp"
#include "logistic.hpp"

namespace sparse_bayes {

SparseBayes::FullStatisticsResult SparseBayes::compute_full_statistics(
    const arma::mat &basis, const arma::mat &Phi, const arma::vec &targets,
    const arma::uvec &used_basis_idx, const arma::mat &basis_Phi,
    const arma::vec &basis_targets, const arma::vec &weights,
    const arma::vec &alpha, double beta) {
  FullStatisticsResult out;

  arma::uword N = basis.n_rows;
  arma::uword M = Phi.n_cols;

  arma::mat U, Ui;
  double data_likelihood;
  arma::vec y, e;
  arma::vec b_vec;
  if (likelihood_ == Likelihood::Gaussian) {
    // Build Hessian
    arma::mat H = Phi.t() * Phi * beta + arma::diagmat(alpha);

    // Cholesky decomposition (upper triangular)
    U = arma::chol(H);
    Ui = arma::inv(arma::trimatu(U));
    out.Sigma = Ui * Ui.t();

    // Posterior mean
    out.weights = (out.Sigma * (Phi.t() * targets)) * beta;

    // Data error and likelihood
    y = Phi * out.weights;
    e = targets - y;
    double ED = arma::dot(e, e);
    data_likelihood =
        (static_cast<double>(N) * std::log(beta) - beta * ED) / 2.0;

    // Compute Q ("quality" factor) and S ("sparsity" factor) values
    out.b_basis_Phi = beta * basis_Phi;
    out.S_in = beta - arma::sum(arma::square(out.b_basis_Phi * Ui), 1);
    out.Q_in = beta * (basis_targets - basis_Phi * out.weights);
  } else if (likelihood_ == Likelihood::Bernoulli) {
    // Use Laplace approximation on the active basis (Phi) to find posterior
    // mode The free function returns mu and the upper-triangular Cholesky
    // factor U;
    LaplaceModeResult mode =
        laplace_posterior_mode(Phi, targets, weights, alpha, likelihood_);
    out.weights = mode.mu;
    U = mode.U;

    // Compute Ui and Sigma
    Ui = arma::inv(arma::trimatu(U));
    out.Sigma = Ui * Ui.t();

    // For Bernoulli compute b_vec, predictions y and data likelihood
    arma::vec logits = Phi * out.weights;
    arma::vec b_local = arma::exp(
        -arma::abs(logits) - 2.0 * arma::log1p(arma::exp(-arma::abs(logits))));
    b_vec = b_local;
    y = sigmoid(logits);

    arma::vec softplus(logits.n_elem);
    arma::uvec pos = arma::find(logits > 0);
    arma::uvec neg = arma::find(logits <= 0);
    if (pos.n_elem > 0) {
      arma::vec zp = logits.elem(pos);
      softplus.elem(pos) = zp + arma::log1p(arma::exp(-zp));
    }
    if (neg.n_elem > 0) {
      arma::vec zn = logits.elem(neg);
      softplus.elem(neg) = arma::log1p(arma::exp(zn));
    }
    data_likelihood = -arma::accu(softplus - targets % logits);

    out.b_vec = b_local;

    e = targets - y;

    // Compute Q ("quality" factor) and S ("sparsity" factor) values
    out.b_basis_Phi =
        basis.t() * (Phi % (out.b_vec * arma::ones<arma::rowvec>(M)));
    out.S_in = (out.b_vec.t() * arma::square(basis)).t() -
               arma::sum(arma::square(out.b_basis_Phi * Ui), 1);
    out.Q_in = basis.t() * e;
  }

  out.S_out = out.S_in;
  out.Q_out = out.Q_in;

  arma::vec S_in_used = out.S_in.elem(used_basis_idx);
  arma::vec Q_in_used = out.Q_in.elem(used_basis_idx);
  arma::vec S_out_used = (alpha % S_in_used) / (alpha - S_in_used);
  arma::vec Q_out_used = (alpha % Q_in_used) / (alpha - S_in_used);
  out.S_out.elem(used_basis_idx) = S_out_used;
  out.Q_out.elem(used_basis_idx) = Q_out_used;

  out.relevance_factor = arma::square(out.Q_out) - out.S_out;

  // Log marginal likelihood
  out.logML =
      data_likelihood - arma::dot(arma::square(out.weights), alpha) / 2.0 +
      arma::sum(arma::log(alpha)) / 2.0 - arma::sum(arma::log(U.diag()));

  // Well-determinedness (Gamma)
  arma::vec diagC = arma::sum(arma::square(Ui), 1);
  out.Gamma = arma::ones<arma::vec>(alpha.n_elem) - alpha % diagC;

  return out;
}

void SparseBayes::inference(const arma::mat &basis_in,
                            const arma::vec &targets_in) {
  // Make local, mutable copy because pre-processing modifies basis
  arma::mat basis = basis_in;

  if (use_bias_) {
    // Add bias column to basis as last column
    basis.insert_cols(basis.n_cols, arma::ones<arma::vec>(basis.n_rows));
  }
  // Pre-process basis: scale columns to unit norm
  arma::vec basis_scales;
  basis_scales.set_size(basis.n_cols);
  for (arma::uword c = 0; c < basis.n_cols; ++c) {
    double nrm = arma::norm(basis.col(c), 2);
    if (nrm == 0.0) nrm = 1.0;
    basis.col(c) /= nrm;
    basis_scales(c) = nrm;
  }

  // Precompute some quantities
  arma::mat basis_targets = basis.t() * targets_in;

  // Initialize the model
  double logML;
  arma::vec Gamma;
  ModelState state = initialize(basis, basis_targets, targets_in, logML, Gamma);

  arma::uword N = basis.n_rows;
  arma::uword M = state.Phi.n_cols;
  arma::uword M_full = basis.n_cols;

  int add_count = 0;
  int delete_count = 0;
  int update_count = 0;
  int ll_update_count = 0;
  int align_defer_count = 0;

  arma::vec log_ml_trace = arma::zeros<arma::vec>(this->iterations_);

  int iter = 0;
  while (iter < this->iterations_) {
    arma::vec DeltaML = arma::zeros<arma::vec>(M_full);
    arma::ivec action_vec =
        arma::ones<arma::ivec>(M_full) * static_cast<int>(Action::None);

    // Evaluate candidate changes using helper functions
    evaluate_reestimates(state, DeltaML, action_vec);
    evaluate_deletions(state, DeltaML, action_vec);
    evaluate_additions(state, DeltaML, action_vec);

    // Fix bias column (if present)
    if (use_bias_) {
      arma::uword bias_idx = M_full - 1;
      DeltaML(bias_idx) = 0.0;
      action_vec(bias_idx) = static_cast<int>(Action::None);
    }
    bool any_to_add = arma::any(action_vec == static_cast<int>(Action::Add));
    bool any_to_del = arma::any(action_vec == static_cast<int>(Action::Delete));

    if ((any_to_add && this->prioritize_addition_) ||
        (any_to_del && this->prioritize_deletion_)) {
      // We will not perform re-estimation in this iteration
      DeltaML
          .elem(arma::find(action_vec == static_cast<int>(Action::Reestimate)))
          .zeros();
      // If one priority is set, enforce it
      if (any_to_add && this->prioritize_addition_ &&
          ~this->prioritize_deletion_) {
        DeltaML.elem(arma::find(action_vec == static_cast<int>(Action::Delete)))
            .zeros();
      } else if (any_to_del && this->prioritize_deletion_ &&
                 ~this->prioritize_addition_) {
        DeltaML.elem(arma::find(action_vec == static_cast<int>(Action::Add)))
            .zeros();
      }
    }

    arma::uword nu = DeltaML.index_max();
    Action selected_action = static_cast<Action>(action_vec(nu));
    double delta_log_marginal = DeltaML(nu);

    arma::vec Phi_vec = basis.col(nu);
    double new_alpha =
        state.S_out(nu) * state.S_out(nu) / state.relevance_factor(nu);

    arma::uword j;  // index into used_basis_idx, Phi, alpha, weights
    if (selected_action == Action::Reestimate ||
        selected_action == Action::Delete) {
      arma::uvec loc = arma::find(state.used_basis_idx == nu);
      if (loc.n_elem == 1) {
        j = loc(0);
      } else {
        // Set j to NaN to indicate an error if we try to use it
        j = arma::datum::nan;
      }
    }

    // Check for termination conditions
    if ((delta_log_marginal <= 0) ||
        (selected_action == Action::Reestimate &&
         std::abs(std::log(new_alpha) - std::log(state.alpha(j))) <
             MIN_DELTA_LOG_ALPHA &&
         ~any_to_del)) {
      selected_action = Action::Terminate;
    }
    // Run alignment checks and handle deferrals/reinstatements
    process_alignment_checks(state, selected_action, nu, Phi_vec,
                             align_defer_count);

    // Perform the selected action and then update statistics
    perform_action(state, selected_action, j, nu, new_alpha, basis, Phi_vec,
                   add_count, delete_count, update_count);

    // Update M after any shape changes
    M = state.Phi.n_cols;

    update_after_action(state, selected_action, basis, targets_in,
                        basis_targets, logML, Gamma, delta_log_marginal,
                        ll_update_count);

    // Beta update (Gaussian case only) moved to helper
    if (likelihood_ == Likelihood::Gaussian) {
      maybe_update_beta(state, basis, targets_in, basis_targets, logML, Gamma,
                        ll_update_count, iter, selected_action);
    }

    if (verbose_) {
      if (likelihood_ == Likelihood::Gaussian) {
        printf(
            "Iter %3d: M=%3d, L=%.6f, Gamma=%.2f, s=%.3f, act=%d on %d (%g)\n",
            iter, M, logML / N, arma::sum(Gamma), std::sqrt(1 / state.beta),
            static_cast<int>(selected_action), nu, delta_log_marginal);
      } else {
        printf("Iter %3d: M=%3d, L=%.6f, Gamma=%.2f, act=%d on %d (%g)\n", iter,
               M, logML / N, arma::sum(Gamma),
               static_cast<int>(selected_action), nu, delta_log_marginal);
      }
    }

    log_ml_trace(iter) = logML;

    if (selected_action == Action::Terminate) {
      if (verbose_) {
        printf("Stopping at iteration %d (max delta log marginal = %g)\n", iter,
               delta_log_marginal);
      }
      break;  // Terminate
    }

    iter++;
  }

  int status = 0;
  if (iter == this->iterations_) {
    if (verbose_) {
      printf("Reached maximum iterations (%d)\n", this->iterations_);
    }
    status = 1;
  }
  if (M == 0) {
    if (verbose_) {
      printf("No basis functions selected\n");
    }
    status = 1;
  }

  if (verbose_) {
    printf("Action summary: Added %d, Deleted %d, Re-estimated %d\n", add_count,
           delete_count, update_count);

    if (align_defer_count > 0) {
      printf("Alignment deferrals: %d\n", align_defer_count);
    }

    printf("Total of %d log likelihood updates\n", ll_update_count);
  }

  // Store outputs on the instance (class members)

  // sort the relevant indices and get the permutation indices
  this->relevant_idx_ =
      arma::sort(state.used_basis_idx);  // sorted Used directly
  arma::uvec index =
      arma::sort_index(state.used_basis_idx);  // permutation indices
  arma::vec scales_used = basis_scales.elem(state.used_basis_idx.elem(index));

  // posterior mean of the weights
  this->mean_ = state.weights.elem(index) / scales_used;

  // posterior covariance of the weights
  arma::mat Sigma_sorted = state.Sigma.submat(index, index);
  arma::mat Dinv = arma::diagmat(1.0 / scales_used);
  this->covariance_ = Dinv * Sigma_sorted * Dinv;

  // precision of the weights prior
  this->alpha_ =
      state.alpha.elem(index) /
      arma::square(basis_scales.elem(state.used_basis_idx.elem(index)));

  // noise precision (only for Gaussian case)
  this->beta_ = state.beta;

  // monitoring values
  this->n_iter_ = iter + 1;
  this->status_ = status;

  // log marginal likelihood trace
  if (iter == this->iterations_) {
    this->log_marginal_likelihood_trace_ = log_ml_trace;
  } else {
    this->log_marginal_likelihood_trace_ = log_ml_trace.subvec(0, iter);
  }
}

SparseBayes::ModelState SparseBayes::initialize(const arma::mat &basis,
                                                const arma::vec &basis_targets,
                                                const arma::vec &targets,
                                                double &logML,
                                                arma::vec &Gamma) {
  // Constants determining defaults
  const double GAUSSIAN_SNR_INIT = 0.1;
  const double INIT_ALPHA_MAX = 1e3;
  const double INIT_ALPHA_MIN = 1e-3;
  const double ALPHA_ZERO = std::numeric_limits<double>::epsilon();

  // Default beta for non-Gaussian: use NaN sentinel for unset
  double beta = std::numeric_limits<double>::quiet_NaN();
  if (likelihood_ == Likelihood::Gaussian) {
    if (this->fixed_noise_ && !this->noise_std_.has_value()) {
      throw std::invalid_argument(
          "Error: noise_std must be set when fixed_noise is true");
    }
    if (this->noise_std_.has_value()) {
      double noise_std = noise_std_.value();
      if (noise_std <= 0.0) {
        throw std::invalid_argument("Error: noise_std must be positive");
      }
      beta = 1.0 / (noise_std * noise_std);
    } else {
      // Initialize beta inversely proportional to signal level
      double stdt = std::max(1e-6, arma::stddev(targets));
      beta = 1.0 / std::pow(stdt * GAUSSIAN_SNR_INIT, 2);
    }
  }

  // Targets pseudo-linearised
  arma::vec targets_pseudo_linear = targets;
  if (likelihood_ == Likelihood::Bernoulli) {
    targets_pseudo_linear = (2.0 * targets - 1.0);
  }

  // Determine initial used_basis_idx set: if available, the bias is used in the
  // initial set. Otherwise pick the basis column with largest projection onto
  // targets_pseudo_linear.
  arma::uvec used_basis_idx;
  if (use_bias_) {
    used_basis_idx = arma::uvec{basis.n_cols - 1};
  } else {
    // Pick column with max absolute projection
    arma::vec proj = arma::abs(basis.t() * targets_pseudo_linear);
    arma::uword idx = proj.index_max();
    used_basis_idx = arma::uvec{idx};
  }
  /////////////////////////////////////////////////////////
  if (verbose_) {
    printf("Initial basis function(s): ");
    for (arma::uword i = 0; i < used_basis_idx.n_elem; ++i) {
      printf("%d ", used_basis_idx(i));
    }
    printf("\n");
  }
  /////////////////////////////////////////////////////////
  arma::mat Phi = basis.cols(used_basis_idx);
  arma::mat basis_Phi;
  if (likelihood_ == Likelihood::Gaussian) {
    basis_Phi = basis.t() * Phi;
  }

  // Weights initialization
  arma::vec weights;
  if (likelihood_ == Likelihood::Gaussian) {
    // Weights left empty (computed later)
  } else if (likelihood_ == Likelihood::Bernoulli) {
    // Initialize weights based on log-odds
    arma::vec log_out = (targets_pseudo_linear * 0.9 + 1.0) / 2.0;
    arma::vec v = arma::log(log_out / (1.0 - log_out));
    weights = arma::solve(Phi, v);
  }

  // Initialize hyperparameters alpha
  arma::vec alpha;
  if (use_bias_) {
    // Set bias alpha to zero (free basis function)
    alpha = arma::vec{ALPHA_ZERO};
  } else {
    if (likelihood_ == Likelihood::Gaussian) {
      arma::vec p = arma::diagvec(Phi.t() * Phi) * beta;
      arma::vec q = (Phi.t() * targets) * beta;
      alpha = arma::square(p) / (arma::square(q) - p);

      // Throw if all alphas are negative at initialization
      // TODO: better handling of this case?
      if (arma::all(alpha < 0)) {
        throw std::runtime_error(
            "Error: no relevant basis function at initialization");
      }

      // Catch negative alpha --> to be pruned
      for (arma::uword i = 0; i < alpha.n_elem; ++i) {
        if (alpha(i) < 0) {
          alpha(i) = INIT_ALPHA_MAX;
        }
      }
    } else if (likelihood_ == Likelihood::Bernoulli) {
      // Prevent division by zero in case of zero weights
      arma::vec safe_w = weights;
      arma::uvec zero_idx = arma::find(safe_w == 0);
      if (zero_idx.n_elem > 0) safe_w.elem(zero_idx).ones();

      alpha = 1.0 / arma::square(safe_w);
      alpha = arma::clamp(alpha, INIT_ALPHA_MIN, INIT_ALPHA_MAX);
    }
  }
  if (verbose_ && (Phi.n_cols == 1)) {
    printf("Initial alpha(s): ");
    for (arma::uword i = 0; i < alpha.n_elem; ++i) {
      printf("%f ", alpha(i));
    }
    printf("\n");
  }

  FullStatisticsResult full_stat =
      compute_full_statistics(basis, Phi, targets, used_basis_idx, basis_Phi,
                              basis_targets, weights, alpha, beta);

  // Fill logging values:
  logML = full_stat.logML;
  Gamma = full_stat.Gamma;

  ModelState state;
  state.alpha = alpha;
  state.Sigma = full_stat.Sigma;
  state.weights = full_stat.weights;
  state.S_in = full_stat.S_in;
  state.Q_in = full_stat.Q_in;
  state.S_out = full_stat.S_out;
  state.Q_out = full_stat.Q_out;
  state.basis_Phi = basis_Phi;
  state.b_basis_Phi = full_stat.b_basis_Phi;
  state.Phi = Phi;
  state.beta = beta;
  state.b_vec = full_stat.b_vec;
  state.used_basis_idx = used_basis_idx;
  state.relevance_factor = full_stat.relevance_factor;

  return state;
}

}  // namespace sparse_bayes
