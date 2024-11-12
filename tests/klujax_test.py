from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu

from klujax_ffi import klujax

jax.config.parse_flags_with_absl()


class KlujaxTests(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("Unsupported platform")

  def test_basic(self):
    n_col=5
    n_lhs=1
    n_rhs=1
    n_nz=5
    Aj = jnp.array([0,1,2,3,4], dtype=jnp.int32)
    Ai = jnp.array([0,3,2,1,4], dtype=jnp.int32)
    Ax = jnp.array([0,3,2,1,4], dtype=jnp.float64)
    b  = jnp.array([1,1,1,1,1], dtype=jnp.float64)
    res = klujax.solve_f64(n_col, n_lhs, n_rhs, n_nz, Ai, Aj, Ax, b)
    print(res)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
