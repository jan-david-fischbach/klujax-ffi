from functools import partial
import numpy as np

import jax
import jax.extend as jex
import jax.numpy as jnp

from klujax_ffi import _klujax

for name, target in _klujax.registrations().items():
  jex.ffi.register_ffi_target(name, target)

def solve_f64(n_col, n_lhs, n_rhs, n_nz, Ai, Aj, Ax, b):

  # In this case, the output of our FFI function is just a single array with the
  # same shape and dtype as the input.
  out_type = jax.ShapeDtypeStruct(b.shape, b.dtype)

  # Note that here we're use `numpy` (not `jax.numpy`) to specify a dtype for
  # the attribute `eps`. Our FFI function expects this to have the C++ `float`
  # type (which corresponds to numpy's `float32` type), and it must be a
  # static parameter (i.e. not a JAX array).
  return jex.ffi.ffi_call(
    # The target name must be the same string as we used to register the target
    # above in `register_ffi_target`
    "solve_f64",
    out_type,
    vmap_method="broadcast_all",
  )(Ai, Aj, Ax, b, n_col=n_col, n_lhs=n_lhs, n_rhs=n_rhs, n_nz=n_nz)
