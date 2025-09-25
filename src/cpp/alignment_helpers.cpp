#include "sparse_bayes.hpp"

namespace sparse_bayes {

// Handle alignment checks and deferrals for candidate additions/deletions.
void SparseBayes::process_alignment_checks(ModelState &state,
                                           Action &selected_action,
                                           arma::uword nu,
                                           const arma::vec &Phi_vec,
                                           int &align_defer_count) {
  // If we're considering adding a basis, check alignment with active bases and
  // defer the addition if highly aligned.
  if (selected_action == Action::Add) {
    arma::rowvec p = Phi_vec.t() * state.Phi;  // 1 x M
    arma::uvec aligned_idx = arma::find(arma::abs(p) > ALIGNMENT_MAX);
    if (aligned_idx.n_elem > 0) {
      // Record all aligned pairs and skip this addition for now
      for (arma::uword k = 0; k < aligned_idx.n_elem; ++k) {
        arma::uword act_pos = aligned_idx(k);
        state.aligned_in.insert_rows(state.aligned_in.n_rows,
                                     arma::uvec{state.used_basis_idx(act_pos)});
        state.aligned_out.insert_rows(state.aligned_out.n_rows, arma::uvec{nu});
      }
      // skip performing the addition now
      selected_action = Action::AlignmentSkip;
      align_defer_count++;
    }
  }

  // If we're deleting a basis, any candidates that were deferred because of
  // alignment with it should be reconsidered (reinstated).
  if (selected_action == Action::Delete) {
    arma::uvec matches = arma::find(state.aligned_in == nu);
    if (matches.n_elem > 0) {
      // Remove matched entries from aligned_in/out
      for (arma::sword m = matches.n_elem - 1; m >= 0; --m) {
        arma::uword idx_to_remove = matches(m);
        state.aligned_in.shed_row(idx_to_remove);
        state.aligned_out.shed_row(idx_to_remove);
        if (verbose_) printf("Alignment reinstatement of %d\n", idx_to_remove);
      }
    }
  }
}

}  // namespace sparse_bayes
