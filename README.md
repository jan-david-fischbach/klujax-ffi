# Klujax with jax FFI

Currently only f64 is implemented and no introspection on the array shapes is performed. As such the arguments including sizes have to be prepared by the user.

Make sure to 
```
make suitesparse
```

before 
```
pip install .
```

The (one) test I have implemented so far seems to give back b instead of the solution vector... Maybe I am missing something here?
