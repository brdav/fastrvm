// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <carma>

#include "sparse_bayes.hpp"

namespace py = pybind11;

namespace sparse_bayes {

PYBIND11_MODULE(_sparsebayes_bindings, m) {
  m.doc() = "SparseBayes C++ core bindings";

  py::enum_<Likelihood>(m, "Likelihood")
      .value("Gaussian", Likelihood::kGaussian)
      .value("Bernoulli", Likelihood::kBernoulli)
      .export_values();

  py::class_<SparseBayes>(m, "SparseBayes")
      .def(py::init([](Likelihood likelihood, int iterations, bool use_bias,
                       bool verbose, bool prioritize_addition,
                       bool prioritize_deletion, bool fixed_noise,
                       std::optional<double> noise_std) {
             return new SparseBayes(likelihood, iterations, use_bias, verbose,
                                    prioritize_addition, prioritize_deletion,
                                    fixed_noise, noise_std);
           }),
           py::arg("likelihood") = Likelihood::kGaussian,
           py::arg("iterations") = 1000, py::arg("use_bias") = false,
           py::arg("verbose") = false, py::arg("prioritize_addition") = false,
           py::arg("prioritize_deletion") = true,
           py::arg("fixed_noise") = false, py::arg("noise_std") = std::nullopt)
      .def("inference", [](SparseBayes &self, const py::array_t<double> &basis,
                           const py::array_t<double> &targets) {
        // Convert incoming numpy arrays to Armadillo via carma
        arma::mat BASIS = carma::arr_to_mat(basis);
        arma::vec TARGET = carma::arr_to_col(targets);

        self.Inference(BASIS, TARGET);

        py::dict result;
        result["mean"] = carma::col_to_arr(self.mean());
        result["covariance"] = carma::mat_to_arr(self.covariance());
        result["relevant_idx"] = carma::col_to_arr(self.relevant_idx());
        result["alpha"] = carma::col_to_arr(self.alpha());
        result["beta"] = self.beta();
        result["n_iter"] = self.n_iter();
        result["status"] = self.status();
        result["log_marginal_likelihood_trace"] =
            carma::col_to_arr(self.log_marginal_likelihood_trace());

        return result;
      });
}

}  // namespace sparse_bayes
