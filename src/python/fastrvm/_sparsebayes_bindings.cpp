// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <armadillo>
#include <cstring>

#include "sparse_bayes.hpp"

namespace py = pybind11;

namespace sparse_bayes {

// Helper namespace for NumPy <-> Armadillo conversions (copy-based)
namespace sbx {
inline arma::mat numpy_to_arma_mat_copy(const py::array &arr_in,
                                        const char *name) {
  // Force cast to double (copies if dtype != float64) then proceed
  py::array arr = py::array::ensure(arr_in);
  py::array_t<double, py::array::c_style | py::array::forcecast> arrd =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          arr);
  if (!arrd) {
    throw std::runtime_error(std::string(name) +
                             " must be convertible to float64");
  }
  py::buffer_info info = arrd.request();
  if (info.ndim != 2) {
    throw std::runtime_error(std::string(name) + " must be 2D");
  }
  arma::mat M(info.shape[0], info.shape[1]);
  const bool fortran =
      (info.strides[0] == (py::ssize_t)sizeof(double) &&
       info.strides[1] == (py::ssize_t)(sizeof(double) * info.shape[0]));
  double *dst = M.memptr();
  if (fortran) {
    std::memcpy(dst, info.ptr, sizeof(double) * info.shape[0] * info.shape[1]);
  } else {
    const char *base = static_cast<const char *>(info.ptr);
    for (py::ssize_t j = 0; j < info.shape[1]; ++j) {
      for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
        const double *src = reinterpret_cast<const double *>(
            base + i * info.strides[0] + j * info.strides[1]);
        dst[i + j * info.shape[0]] = *src;  // column-major storage
      }
    }
  }
  return M;
}
inline arma::vec numpy_to_arma_vec_copy(const py::array &arr_in,
                                        const char *name) {
  py::array arr = py::array::ensure(arr_in);
  py::array_t<double, py::array::c_style | py::array::forcecast> arrd =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          arr);
  if (!arrd) {
    throw std::runtime_error(std::string(name) +
                             " must be convertible to float64");
  }
  py::buffer_info info = arrd.request();
  if (info.ndim != 1) {
    throw std::runtime_error(std::string(name) + " must be 1D");
  }
  arma::vec v(info.shape[0]);
  double *dst = v.memptr();
  if (info.strides[0] == (py::ssize_t)sizeof(double)) {
    std::memcpy(dst, info.ptr, sizeof(double) * info.shape[0]);
  } else {
    const char *base = static_cast<const char *>(info.ptr);
    for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
      const double *src =
          reinterpret_cast<const double *>(base + i * info.strides[0]);
      dst[i] = *src;
    }
  }
  return v;
}
inline py::array arma_mat_to_numpy_copy(const arma::mat &M) {
  py::array out(py::buffer_info(
      /*ptr=*/nullptr,
      /*itemsize=*/sizeof(double),
      /*format=*/py::format_descriptor<double>::format(),
      /*ndim=*/2,
      /*shape=*/
      std::vector<py::ssize_t>{(py::ssize_t)M.n_rows, (py::ssize_t)M.n_cols},
      /*strides=*/
      std::vector<py::ssize_t>{(py::ssize_t)sizeof(double),
                               (py::ssize_t)(sizeof(double) * M.n_rows)}));
  auto buf = out.request();
  std::memcpy(buf.ptr, M.memptr(), sizeof(double) * M.n_rows * M.n_cols);
  return out;
}
inline py::array arma_vec_to_numpy_copy(const arma::vec &v) {
  py::array out(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::format(), 1,
      {(py::ssize_t)v.n_elem}, {(py::ssize_t)sizeof(double)}));
  std::memcpy(out.request().ptr, v.memptr(), sizeof(double) * v.n_elem);
  return out;
}
inline py::array arma_uvec_to_numpy_int64_copy(const arma::uvec &v) {
  // Always expose as int64 for Python indexing consistency.
  using I64 = long long;  // assuming 64-bit platform
  py::array out(py::buffer_info(
      nullptr, sizeof(I64), py::format_descriptor<I64>::format(), 1,
      {(py::ssize_t)v.n_elem}, {(py::ssize_t)sizeof(I64)}));
  auto buf = out.request();
  I64 *dst = static_cast<I64 *>(buf.ptr);
  for (arma::uword i = 0; i < v.n_elem; ++i) {
    dst[i] = static_cast<I64>(v[i]);
  }
  return out;
}
}  // namespace sbx

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
      .def("inference", [](SparseBayes &self, const py::array &basis,
                           const py::array &targets) {
        arma::mat BASIS = sbx::numpy_to_arma_mat_copy(basis, "basis");
        arma::vec TARGET = sbx::numpy_to_arma_vec_copy(targets, "targets");

        self.Inference(BASIS, TARGET);

        py::dict result;
        result["mean"] = sbx::arma_vec_to_numpy_copy(self.mean());
        result["covariance"] = sbx::arma_mat_to_numpy_copy(self.covariance());
        result["relevant_idx"] =
            sbx::arma_uvec_to_numpy_int64_copy(self.relevant_idx());
        result["alpha"] = sbx::arma_vec_to_numpy_copy(self.alpha());
        result["beta"] = self.beta();
        result["n_iter"] = self.n_iter();
        result["status"] = self.status();
        result["log_marginal_likelihood_trace"] =
            sbx::arma_vec_to_numpy_copy(self.log_marginal_likelihood_trace());

        return result;
      });
}

}  // namespace sparse_bayes
