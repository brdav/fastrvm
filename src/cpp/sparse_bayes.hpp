// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef SPARSE_BAYES_HPP
#define SPARSE_BAYES_HPP

#include <armadillo>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sparse_bayes {

// Supported likelihood types.
enum class Likelihood {
  kGaussian,
  kBernoulli,
};

// Actions performed by the main optimization loop.
enum class Action : int {
  kNone = 0,
  kReestimate = 1,
  kAdd = 2,
  kDelete = 3,
  kTerminate = 4,
  kNoiseOnly = 5,
  kAlignmentSkip = 6,
};

class SparseBayes {
 public:
  SparseBayes(Likelihood likelihood = Likelihood::kGaussian,
              int iterations = 1000, bool use_bias = false,
              bool verbose = false, bool prioritize_addition = false,
              bool prioritize_deletion = true, bool fixed_noise = false,
              std::optional<double> noise_std = std::nullopt)
      : likelihood_(likelihood),
        iterations_(iterations),
        use_bias_(use_bias),
        verbose_(verbose),
        prioritize_addition_(prioritize_addition),
        prioritize_deletion_(prioritize_deletion),
        fixed_noise_(fixed_noise),
        noise_std_(noise_std) {}

  virtual ~SparseBayes() = default;

  // Run the optimization using the supplied basis and targets.
  void Inference(const arma::mat& basis, const arma::vec& targets);

  // Accessors for model parameters after fitting.
  const arma::uvec& relevant_idx() const { return relevant_idx_; }
  const arma::vec& mean() const { return mean_; }
  const arma::mat& covariance() const { return covariance_; }
  const arma::vec& alpha() const { return alpha_; }
  double beta() const { return beta_; }
  int n_iter() const { return n_iter_; }
  int status() const { return status_; }
  const arma::vec& log_marginal_likelihood_trace() const {
    return log_marginal_likelihood_trace_;
  }

 private:
  // Forward declarations for types used in function signatures.
  struct ModelState;

  // Initialize model variables and do basic basis preprocessing.
  ModelState Initialize(const arma::mat& basis, const arma::vec& basis_targets,
                        const arma::vec& targets, double& logML,
                        arma::vec& Gamma);

  // Re-estimate a basis function's alpha.
  void ReestimateBasis(ModelState& state, arma::uword j, double new_alpha);

  // Add a basis function into the active set.
  void AddBasisFunction(ModelState& state, arma::uword nu, double new_alpha,
                        const arma::mat& basis, const arma::vec& phi_vec);

  // Delete a basis function at interior index j.
  void DeleteBasisFunction(ModelState& state, arma::uword j);

  // Result container for full-statistics computation.
  struct FullStatisticsResult {
    arma::vec weights;
    arma::mat Sigma;
    arma::vec S_in;
    arma::vec Q_in;
    arma::vec S_out;
    arma::vec Q_out;
    arma::vec relevance_factor;
    arma::mat b_basis_Phi;
    arma::vec b_vec;
    double logML;     // Log marginal likelihood.
    arma::vec Gamma;  // Well-determinedness.
  };

  // Mutable model state used across helper functions to avoid long parameter
  // lists and make ownership explicit.
  struct ModelState {
    arma::vec alpha;
    arma::mat Sigma;
    arma::vec weights;
    arma::vec S_in;
    arma::vec Q_in;
    arma::vec S_out;
    arma::vec Q_out;
    arma::mat basis_Phi;
    arma::mat b_basis_Phi;
    arma::mat Phi;
    double beta;
    arma::vec b_vec;  // Non-Gaussian per-datum weights.
    arma::uvec used_basis_idx;
    arma::vec relevance_factor;
    arma::uvec aligned_in;
    arma::uvec aligned_out;
  };

  FullStatisticsResult ComputeFullStatistics(
      const arma::mat& basis, const arma::mat& Phi, const arma::vec& targets,
      const arma::uvec& used_basis_idx, const arma::mat& basis_Phi,
      const arma::vec& basis_targets, const arma::vec& weights,
      const arma::vec& alpha, double beta);

  // Helpers to split the main optimization loop into smaller pieces.
  void EvaluateReestimates(ModelState& state, arma::vec& DeltaML,
                           arma::ivec& action);
  void EvaluateDeletions(ModelState& state, arma::vec& DeltaML,
                         arma::ivec& action);
  void EvaluateAdditions(ModelState& state, arma::vec& DeltaML,
                         arma::ivec& action);

  // Perform an action (reestimate/add/delete) and update local counters.
  void PerformAction(ModelState& state, Action selected_action, arma::uword j,
                     arma::uword nu, double new_alpha, const arma::mat& basis,
                     const arma::vec& phi_vec, int& add_count,
                     int& delete_count, int& update_count);

  // Update statistics (possibly recomputing full stats) after an action and
  // update log marginal likelihood and Gamma as needed.
  void UpdateAfterAction(ModelState& state, Action selected_action,
                         const arma::mat& basis, const arma::vec& targets_in,
                         const arma::mat& basis_targets, double& logML,
                         arma::vec& Gamma, double& delta_log_marginal,
                         int& ll_update_count);

  // Update beta (Gaussian noise precision) when appropriate. This may recompute
  // full statistics and adjust `selected_action` to defer termination if the
  // beta update is large.
  void MaybeUpdateBeta(ModelState& state, const arma::mat& basis,
                       const arma::vec& targets_in,
                       const arma::mat& basis_targets, double& logML,
                       arma::vec& Gamma, int& ll_update_count, int iter,
                       Action& selected_action);

  // Handle alignment checks and deferrals for candidate additions/deletions.
  // This will record aligned pairs and possibly defer an addition by modifying
  // selected_action and aligned_in/aligned_out in state.
  void ProcessAlignmentChecks(ModelState& state, Action& selected_action,
                              arma::uword nu, const arma::vec& phi_vec,
                              int& align_defer_count);

  // Likelihood for this instance.
  Likelihood likelihood_;

  // Maximum number of iterations.
  int iterations_;

  // Whether to print progress information.
  bool verbose_;

  // Whether to include a bias (intercept) term.
  bool use_bias_;

  // If true, the noise precision (beta) is fixed and not updated during
  // fitting.
  bool fixed_noise_;

  // Optional initial noise standard deviation for Gaussian likelihood.
  std::optional<double> noise_std_;

  // Control flags for prioritizing operations during the main loop.
  bool prioritize_addition_;
  bool prioritize_deletion_;

  // Control parameters.
  static constexpr double kZeroFactor = 1e-12;
  static constexpr double kMinDeltaLogAlpha = 1e-3;
  static constexpr double kMinDeltaLogBeta = 1e-6;
  static constexpr int kBetaWarmupIter = 10;
  static constexpr int kBetaUpdateFrequency = 5;
  static constexpr double kBetaMaxFactor = 1e6;
  static constexpr double kAlignmentMax = 1 - 1e-3;

  // Indices of selected basis functions.
  arma::uvec relevant_idx_;

  // Posterior mean of the model weights.
  arma::vec mean_;

  // Posterior covariance of the model weights.
  arma::mat covariance_;

  // Precisions of the (Gaussian) weights priors.
  arma::vec alpha_;

  // Noise precision for regression; not used for classification.
  double beta_;

  // Number of iterations completed.
  int n_iter_;

  // Status code: 0 = success, 1 = max iterations reached.
  int status_;

  // Log marginal likelihood trace over iterations.
  arma::vec log_marginal_likelihood_trace_;
};

}  // namespace sparse_bayes

#endif  // SPARSE_BAYES_HPP
