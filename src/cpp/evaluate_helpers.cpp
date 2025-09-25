// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sparse_bayes.hpp"

namespace sparse_bayes {

// Evaluate re-estimation candidates and populate DeltaML for re-estimates.
void SparseBayes::EvaluateReestimates(ModelState &state, arma::vec &DeltaML,
                                      arma::ivec &action_vec) {
  arma::vec used_factor = state.relevance_factor.elem(state.used_basis_idx);
  arma::uvec i_pos = arma::find(used_factor > kZeroFactor);

  if (i_pos.n_elem > 0) {
    arma::uvec reest_idx = state.used_basis_idx.elem(i_pos);
    arma::vec new_alpha = arma::square(state.S_out.elem(reest_idx)) /
                          state.relevance_factor.elem(reest_idx);

    arma::vec delta = (1.0 / new_alpha) - (1.0 / state.alpha.elem(i_pos));
    arma::vec Q_idx = state.Q_in.elem(reest_idx);
    arma::vec S_idx = state.S_in.elem(reest_idx);

    arma::vec numer = delta % arma::square(Q_idx);
    arma::vec denom = delta % S_idx + 1.0;
    arma::vec term = numer / denom - arma::log(1.0 + S_idx % delta);

    DeltaML.elem(reest_idx) = term / 2.0;
    action_vec.elem(reest_idx).fill(static_cast<int>(Action::kReestimate));
  }
}

// Evaluate deletion candidates and set action_vec entries for deletions.
void SparseBayes::EvaluateDeletions(ModelState &state, arma::vec &DeltaML,
                                    arma::ivec &action_vec) {
  arma::vec used_factor = state.relevance_factor.elem(state.used_basis_idx);
  arma::uvec i_neg = arma::find(used_factor <= kZeroFactor);

  if (i_neg.n_elem > 0 && state.used_basis_idx.n_elem > 1) {
    arma::uvec del_idx = state.used_basis_idx.elem(i_neg);
    arma::vec Q_idx = state.Q_out.elem(del_idx);
    arma::vec S_idx = state.S_out.elem(del_idx);
    arma::vec alpha_used = state.alpha.elem(i_neg);

    arma::vec term = (arma::square(Q_idx) / (S_idx + alpha_used)) -
                     arma::log(S_idx / alpha_used + 1.0);

    DeltaML.elem(del_idx) = -term / 2.0;
    action_vec.elem(del_idx).fill(static_cast<int>(Action::kDelete));
  }
}

// Evaluate additions: set DeltaML and action_vec for candidate additions.
void SparseBayes::EvaluateAdditions(ModelState &state, arma::vec &DeltaML,
                                    arma::ivec &action_vec) {
  arma::uvec add_mask =
      arma::conv_to<arma::uvec>::from(state.relevance_factor > kZeroFactor);

  add_mask.elem(state.used_basis_idx).zeros();

  if (state.aligned_out.n_elem > 0) {
    add_mask.elem(state.aligned_out).zeros();
  }

  arma::uvec add_idx = arma::find(add_mask);
  if (add_idx.n_elem > 0) {
    arma::vec Q_idx = state.Q_in.elem(add_idx);
    arma::vec S_idx = state.S_in.elem(add_idx);

    arma::vec quot = arma::square(Q_idx) / S_idx;
    arma::vec delta = (quot - 1.0 - arma::log(quot)) / 2.0;

    DeltaML.elem(add_idx) = delta;
    action_vec.elem(add_idx).fill(static_cast<int>(Action::kAdd));
  }
}

}  // namespace sparse_bayes
