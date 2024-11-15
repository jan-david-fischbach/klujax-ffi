// version: 0.2.10
// author: Floris Laporte, Jan David Fischbach

#include <iostream>
using namespace std;
#include <vector>

#include <klu.h>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

void coo_to_csc_analyze(int n_col, int n_nz, int *Ai, int *Aj, int *Bi, int *Bp,
                        int *Bk) {
  // TODO: rename n_nz to Anz 

  // compute number of non-zero entries per row of A
  for (int n = 0; n < n_nz; n++) {
    Bp[Aj[n]] += 1;
  }

  // cumsum the n_nz per row to get Bp
  int cumsum = 0;
  int temp = 0;
  for (int j = 0; j < n_col; j++) {
    temp = Bp[j];
    Bp[j] = cumsum;
    cumsum += temp;
  }

  // write Ai, Ax into Bi, Bk
  int col = 0;
  int dest = 0;
  for (int n = 0; n < n_nz; n++) {
    col = Aj[n];
    dest = Bp[col];
    Bi[dest] = Ai[n];
    Bk[dest] = n;
    Bp[col] += 1;
  }

  int last = 0;
  for (int i = 0; i <= n_col; i++) {
    temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }
}

void solve_f64_impl(
    int n_col, int n_lhs, int n_rhs, int Anz, int *Ai, int *Aj, 
    double *Ax, double *b, double *result
  ){
  // copy b into result
  for (int i = 0; i < n_lhs * n_col * n_rhs; i++) {
    result[i] = b[i];
  }

  // get COO -> CSC transformation information
  int *Bk = new int[Anz](); // Ax -> Bx transformation indices
  int *Bi = new int[Anz]();
  int *Bp = new int[n_col + 1]();

  coo_to_csc_analyze(n_col, Anz, Ai, Aj, Bi, Bp, Bk);

  // initialize KLU for given sparsity pattern
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  klu_common Common;
  klu_defaults(&Common);
  Symbolic = klu_analyze(n_col, Bp, Bi, &Common);

  // solve for other elements in batch:
  // NOTE: same sparsity pattern for each element in batch assumed
  double *Bx = new double[Anz]();
  for (int i = 0; i < n_lhs; i++) {
    int m = i * Anz;
    int n = i * n_rhs * n_col;

    // convert COO Ax to CSC Bx
    for (int k = 0; k < Anz; k++) {
      Bx[k] = Ax[m + Bk[k]];
    }

    // solve using KLU
    Numeric = klu_factor(Bp, Bi, Bx, Symbolic, &Common);
    klu_solve(Symbolic, Numeric, n_col, n_rhs, &result[n], &Common);
  }

  // clean up
  klu_free_symbolic(&Symbolic, &Common);
  klu_free_numeric(&Numeric, &Common);
  delete[] Bk;
  delete[] Bi;
  delete[] Bp;
  delete[] Bx;
}

void coo_mul_vec_f64(
    int n_col, int n_lhs, int n_rhs, int Anz, int *Ai, int *Aj, 
    double *Ax, double *b, double *result
  ){
    
  // initialize empty result
  for (int i = 0; i < n_lhs * n_col * n_rhs; i++) {
    result[i] = 0.0;
  }

  // fill result
  for (int i = 0; i < n_lhs; i++) {
    int m = i * Anz;
    int n = i * n_rhs * n_col;
    for (int j = 0; j < n_rhs; j++) {
      for (int k = 0; k < Anz; k++) {
        result[n + Ai[k] + j * n_col] += Ax[m + k] * b[n + Aj[k] + j * n_col];
      }
    }
  }
}

ffi::Error solve_f64_wrapped(
  int n_col,
  int n_lhs,
  int n_rhs,
  int n_nz,
  ffi::Buffer<ffi::S32> Ai,
  ffi::Buffer<ffi::S32> Aj, //TODO: should be uint as not negative 
  ffi::Buffer<ffi::F64> Ax,
  ffi::Buffer<ffi::F64> b,
  ffi::Result<ffi::Buffer<ffi::F64>> res) 
{
  solve_f64_impl(n_col, n_lhs, n_rhs, n_nz, 
    Ai.typed_data(), 
    Aj.typed_data(), 
    Ax.typed_data(), 
    b.typed_data(), 
    res->typed_data());
  return ffi::Error::Success();
}

// Wrap `solve_f64` and specify the interface to XLA. If you need to declare
// this handler in a header, you can use the `XLA_FFI_DECLASE_HANDLER_SYMBOL`
// macro: `XLA_FFI_DECLASE_HANDLER_SYMBOL(solve_f64)`.
XLA_FFI_DEFINE_HANDLER_SYMBOL(solve_f64, solve_f64_wrapped,
                              ffi::Ffi::Bind()
                                  .Attr<int>("n_col")
                                  .Attr<int>("n_lhs")
                                  .Attr<int>("n_rhs")
                                  .Attr<int>("n_nz")
                                  .Arg<ffi::Buffer<ffi::S32>>()  // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()  // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()  // Ax
                                  .Arg<ffi::Buffer<ffi::F64>>()  // b
                                  .Ret<ffi::Buffer<ffi::F64>>()  // res
);

template <typename T>
nb::capsule EncapsulateFfiHandler(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(_klujax, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["solve_f64"] = EncapsulateFfiHandler(solve_f64);
    return registrations;
  });
}