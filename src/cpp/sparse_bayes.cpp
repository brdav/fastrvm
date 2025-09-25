// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sparse_bayes.hpp"

#include <algorithm>
#include <ensmallen.hpp>
#include <iostream>
#include <limits>
#include <optional>

#include "laplace.hpp"
#include "logistic.hpp"

namespace sparse_bayes {

void SparseBayes::Inference(const arma::mat& basis_in,
                            const arma::vec& targets_in) {
  // Make local, mutable copy because pre-processing modifies basis.
  arma::mat basis = basis_in;

  if (use_bias_) {
    // Add bias column to basis as last column.
    basis.insert_cols(basis.n_cols, arma::ones<arma::vec>(basis.n_rows));
  }

  // Pre-process basis: scale columns to unit norm.
  arma::vec basis_scales(basis.n_cols);
  for (arma::uword c = 0; c < basis.n_cols; ++c) {
    double nrm = arma::norm(basis.col(c), 2);
    if (nrm == 0.0) {
      nrm = 1.0;
    }
    basis.col(c) /= nrm;
    basis_scales(c) = nrm;
  }

  // Precompute some quantities.
  arma::mat basis_targets = basis.t() * targets_in;

  // Initialize the model.
  double logML;
  arma::vec Gamma;
  ModelState state = Initialize(basis, basis_targets, targets_in, logML, Gamma);

  arma::uword N = basis.n_rows;
  arma::uword M = state.Phi.n_cols;
  arma::uword M_full = basis.n_cols;

  int add_count = 0;
  int delete_count = 0;
  int update_count = 0;
  int ll_update_count = 0;
  int align_defer_count = 0;

  arma::vec log_ml_trace = arma::zeros<arma::vec>(iterations_);

  int iter = 0;
  while (iter < iterations_) {
    arma::vec DeltaML = arma::zeros<arma::vec>(M_full);
    arma::ivec action_vec =
        arma::ones<arma::ivec>(M_full) * static_cast<int>(Action::kNone);

    // Evaluate candidate changes using helper functions.
    EvaluateReestimates(state, DeltaML, action_vec);
    EvaluateDeletions(state, DeltaML, action_vec);
    EvaluateAdditions(state, DeltaML, action_vec);

    // Fix bias column (if present).
    if (use_bias_) {
      arma::uword bias_idx = M_full - 1;
      DeltaML(bias_idx) = 0.0;
      action_vec(bias_idx) = static_cast<int>(Action::kNone);
    }
    bool any_to_add = arma::any(action_vec == static_cast<int>(Action::kAdd));
    bool any_to_del =
        arma::any(action_vec == static_cast<int>(Action::kDelete));

    if ((any_to_add && prioritize_addition_) ||
        (any_to_del && prioritize_deletion_)) {
      // We will not perform re-estimation in this iteration.
      DeltaML
          .elem(arma::find(action_vec == static_cast<int>(Action::kReestimate)))
          .zeros();
      if (any_to_add && prioritize_addition_ && !prioritize_deletion_) {
        DeltaML
            .elem(arma::find(action_vec == static_cast<int>(Action::kDelete)))
            .zeros();
      } else if (any_to_del && prioritize_deletion_ && !prioritize_addition_) {
        DeltaML.elem(arma::find(action_vec == static_cast<int>(Action::kAdd)))
            .zeros();
      }
    }

    arma::uword nu = DeltaML.index_max();
    Action selected_action = static_cast<Action>(action_vec(nu));
    double delta_log_marginal = DeltaML(nu);

    arma::vec Phi_vec = basis.col(nu);
    double new_alpha =
        state.S_out(nu) * state.S_out(nu) / state.relevance_factor(nu);

    arma::uword j;  // Index into used_basis_idx, Phi, alpha, weights.
    if (selected_action == Action::kReestimate ||
        selected_action == Action::kDelete) {
      arma::uvec loc = arma::find(state.used_basis_idx == nu);
      if (loc.n_elem == 1) {
        j = loc(0);
      } else {
        j = arma::datum::nan;  // Error sentinel.
      }
    }

    // Check for termination conditions.
    if ((delta_log_marginal <= 0) ||
        (selected_action == Action::kReestimate &&
         std::abs(std::log(new_alpha) - std::log(state.alpha(j))) <
             kMinDeltaLogAlpha &&
         !any_to_del)) {
      selected_action = Action::kTerminate;
    }

    // Run alignment checks and handle deferrals/reinstatements.
    ProcessAlignmentChecks(state, selected_action, nu, Phi_vec,
                           align_defer_count);

    // Perform the selected action and update statistics.
    PerformAction(state, selected_action, j, nu, new_alpha, basis, Phi_vec,
                  add_count, delete_count, update_count);

    // Update M after any shape changes.
    M = state.Phi.n_cols;

    UpdateAfterAction(state, selected_action, basis, targets_in, basis_targets,
                      logML, Gamma, delta_log_marginal, ll_update_count);

    // Beta update (Gaussian case only).
    if (likelihood_ == Likelihood::kGaussian) {
      MaybeUpdateBeta(state, basis, targets_in, basis_targets, logML, Gamma,
                      ll_update_count, iter, selected_action);
    }

    if (verbose_ && (iter % 10 == 0)) {
      if (likelihood_ == Likelihood::kGaussian) {
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

    if (selected_action == Action::kTerminate) {
      if (verbose_) {
        printf("Stopping at iteration %d (max delta log marginal = %g)\n", iter,
               delta_log_marginal);
      }
      break;
    }

    iter++;
  }

  int status = 0;
  if (iter == iterations_) {
    if (verbose_) {
      printf("Reached maximum iterations (%d)\n", iterations_);
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

  // Store outputs on the instance.
  relevant_idx_ = arma::sort(state.used_basis_idx);
  arma::uvec index = arma::sort_index(state.used_basis_idx);
  arma::vec scales_used = basis_scales.elem(state.used_basis_idx.elem(index));

  mean_ = state.weights.elem(index) / scales_used;

  arma::mat Sigma_sorted = state.Sigma.submat(index, index);
  arma::mat Dinv = arma::diagmat(1.0 / scales_used);
  covariance_ = Dinv * Sigma_sorted * Dinv;

  alpha_ = state.alpha.elem(index) /
           arma::square(basis_scales.elem(state.used_basis_idx.elem(index)));

  beta_ = state.beta;

  n_iter_ = iter + 1;
  status_ = status;

  if (iter == iterations_) {
    log_marginal_likelihood_trace_ = log_ml_trace;
  } else {
    log_marginal_likelihood_trace_ = log_ml_trace.subvec(0, iter);
  }
}

SparseBayes::ModelState SparseBayes::Initialize(const arma::mat& basis,
                                                const arma::vec& basis_targets,
                                                const arma::vec& targets,
                                                double& logML,
                                                arma::vec& Gamma) {
  constexpr double kGaussianSnrInit = 0.1;
  constexpr double kInitAlphaMax = 1e3;
  constexpr double kInitAlphaMin = 1e-3;
  constexpr double kAlphaZero = std::numeric_limits<double>::epsilon();

  double beta = std::numeric_limits<double>::quiet_NaN();
  if (likelihood_ == Likelihood::kGaussian) {
    if (fixed_noise_ && !noise_std_.has_value()) {
      throw std::invalid_argument(
          "Error: noise_std must be set when fixed_noise is true");
    }
    if (noise_std_.has_value()) {
      double noise_std = noise_std_.value();
      if (noise_std <= 0.0) {
        throw std::invalid_argument("Error: noise_std must be positive");
      }
      beta = 1.0 / (noise_std * noise_std);
    } else {
      double stdt = std::max(1e-6, arma::stddev(targets));
      beta = 1.0 / std::pow(stdt * kGaussianSnrInit, 2);
    }
  }

  arma::vec targets_pseudo_linear = targets;
  if (likelihood_ == Likelihood::kBernoulli) {
    targets_pseudo_linear = (2.0 * targets - 1.0);
  }

  arma::uvec used_basis_idx;
  if (use_bias_) {
    used_basis_idx = arma::uvec{basis.n_cols - 1};
  } else {
    arma::vec proj = arma::abs(basis.t() * targets_pseudo_linear);
    arma::uword idx = proj.index_max();
    used_basis_idx = arma::uvec{idx};
  }

  if (verbose_) {
    printf("Initial basis function(s): ");
    for (arma::uword i = 0; i < used_basis_idx.n_elem; ++i) {
      printf("%d ", used_basis_idx(i));
    }
    printf("\n");
  }

  arma::mat Phi = basis.cols(used_basis_idx);
  arma::mat basis_Phi;
  if (likelihood_ == Likelihood::kGaussian) {
    basis_Phi = basis.t() * Phi;
  }

  arma::vec weights;
  if (likelihood_ == Likelihood::kBernoulli) {
    arma::vec log_out = (targets_pseudo_linear * 0.9 + 1.0) / 2.0;
    arma::vec v = arma::log(log_out / (1.0 - log_out));
    weights = arma::solve(Phi, v);
  }

  arma::vec alpha;
  if (use_bias_) {
    alpha = arma::vec{kAlphaZero};
  } else {
    if (likelihood_ == Likelihood::kGaussian) {
      arma::vec p = arma::diagvec(Phi.t() * Phi) * beta;
      arma::vec q = (Phi.t() * targets) * beta;
      alpha = arma::square(p) / (arma::square(q) - p);
      if (arma::all(alpha < 0)) {
        throw std::runtime_error(
            "Error: no relevant basis function at initialization");
      }
      for (arma::uword i = 0; i < alpha.n_elem; ++i) {
        if (alpha(i) < 0) {
          alpha(i) = kInitAlphaMax;
        }
      }
    } else if (likelihood_ == Likelihood::kBernoulli) {
      arma::vec safe_w = weights;
      arma::uvec zero_idx = arma::find(safe_w == 0);
      if (zero_idx.n_elem > 0) {
        safe_w.elem(zero_idx).ones();
      }
      alpha = 1.0 / arma::square(safe_w);
      alpha = arma::clamp(alpha, kInitAlphaMin, kInitAlphaMax);
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
      ComputeFullStatistics(basis, Phi, targets, used_basis_idx, basis_Phi,
                            basis_targets, weights, alpha, beta);

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

SparseBayes::FullStatisticsResult SparseBayes::ComputeFullStatistics(
    const arma::mat& basis, const arma::mat& Phi, const arma::vec& targets,
    const arma::uvec& used_basis_idx, const arma::mat& basis_Phi,
    const arma::vec& basis_targets, const arma::vec& weights,
    const arma::vec& alpha, double beta) {
  FullStatisticsResult out;

  arma::uword N = basis.n_rows;
  arma::uword M = Phi.n_cols;

  arma::mat U, Ui;
  double data_likelihood;
  arma::vec y, e;
  arma::vec b_vec;

  if (likelihood_ == Likelihood::kGaussian) {
    arma::mat H = Phi.t() * Phi * beta + arma::diagmat(alpha);
    U = arma::chol(H);
    Ui = arma::inv(arma::trimatu(U));
    out.Sigma = Ui * Ui.t();

    out.weights = (out.Sigma * (Phi.t() * targets)) * beta;

    y = Phi * out.weights;
    e = targets - y;
    double ED = arma::dot(e, e);
    data_likelihood =
        (static_cast<double>(N) * std::log(beta) - beta * ED) / 2.0;

    out.b_basis_Phi = beta * basis_Phi;
    out.S_in = beta - arma::sum(arma::square(out.b_basis_Phi * Ui), 1);
    out.Q_in = beta * (basis_targets - basis_Phi * out.weights);
  } else if (likelihood_ == Likelihood::kBernoulli) {
    LaplaceModeResult mode =
        LaplacePosteriorMode(Phi, targets, weights, alpha, likelihood_);
    out.weights = mode.mu;
    U = mode.U;

    Ui = arma::inv(arma::trimatu(U));
    out.Sigma = Ui * Ui.t();

    arma::vec logits = Phi * out.weights;
    arma::vec b_local = arma::exp(
        -arma::abs(logits) - 2.0 * arma::log1p(arma::exp(-arma::abs(logits))));
    b_vec = b_local;
    y = Sigmoid(logits);

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

  out.logML =
      data_likelihood - arma::dot(arma::square(out.weights), alpha) / 2.0 +
      arma::sum(arma::log(alpha)) / 2.0 - arma::sum(arma::log(U.diag()));

  arma::vec diagC = arma::sum(arma::square(Ui), 1);
  out.Gamma = arma::ones<arma::vec>(alpha.n_elem) - alpha % diagC;

  return out;
}

}  // namespace sparse_bayes
