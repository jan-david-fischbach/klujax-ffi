// version: 0.2.10
// author: Floris Laporte, Jan David Fischbach

#include <iostream>
using namespace std;
#include <vector>

#include <klu.h>

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

void solve_f64(
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

// main() is where program execution begins.
int main() {
  int n_col = 5;
  int n_lhs = 1; //Batch 1
  int n_nz = 5;
  int n_rhs = 1; //Batch 2

  double Ax[n_nz];
  std::fill_n(Ax, n_nz, 2);
  double b[n_col*n_rhs];
  std::fill_n(b, n_col*n_rhs, 3);

  int Ai[n_nz] = {0, 1, 2, 3, 4};
  int Aj[n_nz] = {0, 3, 2, 1, 4};

  double result[n_col*n_rhs];

  solve_f64(
    n_col, n_lhs, n_rhs, n_nz, Ai, Aj, 
    Ax, b, result
  );
  
  cout << "Hello World: You are mine" << endl; // prints Hello World

  cout << result[0] << endl;
  return 0;
}