# Cholesky Decomposition
## Result 256 x 256 Matrix

| Language | Time, ms | Comment |
|----------|---------|---------|
| scipy |    0.34 | LAPACK |
| SWIG - CPP - Eigen |   0.91 |  |
| Numba - Numpy |    1.66 | |
| SWIG - CPP |   2.16 |  |
| Cython - Numpy |    44.7 |  |
| Python - Numpy |   71.4 |  |
| javascript - node |   294 | ref |
| Python - Native |   319 |  |

### Swig - Eigen
To build a Swig Eigen module, eigen.i and the most part of wrapper description are borrowed from [rdeits/swig-eigen-numpy: A demonstration of a SWIG wrapper for a C++ library containing Eigen matrix types for use with Python and NumPy](https://github.com/rdeits/swig-eigen-numpy)
