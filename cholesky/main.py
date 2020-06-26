# coding: utf-8
import pytest 
import numpy as np
import warnings

np.random.seed(331)
warnings.filterwarnings("ignore")

from scipy.linalg import cholesky
from math import sqrt
from numba import jit
from swig.SwigMod import cholesky_swig
from swig_eigen.swig_eigen_mod import cholesky_swig_eigen

def gen_sym_matrix(M):
    A = np.random.rand(M, M).astype(np.float64)
    return np.matmul(A, A.T)+np.eye(M)

def cholesky_native_python(A, **kwarg):
    M = len(A)
    L = [[0.0] * len(A) for _ in range(len(A))]
    for i in range(M):
        for j in range(0, i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if(i==j):
                L[i][j] = sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s)/L[j][j]
    return L

def cholesky_numpy(A, **kwarg):
    M = A.shape[0]
    L = np.zeros_like(A, dtype=np.float64)

    for i in range(M):
        for j in range(0, i+1):
            s = np.dot(L[i,:j],L[j,:j])
            L[i, j] = np.sqrt(A[i,i]-s) if(i==j) else ((A[i,j]-s)/L[j,j])
    return L

@jit(nopython=True)
def cholesky_numpy_numba(A):
    M = A.shape[0]
    L = np.zeros_like(A, dtype=np.float64)

    for i in range(M):
        for j in range(0, i+1):
            s = np.dot(L[i,:j],L[j,:j])
            L[i, j] = np.sqrt(A[i,i]-s) if(i==j) else ((A[i,j]-s)/L[j,j])
    return L

A128 =  gen_sym_matrix(128)
A256 = gen_sym_matrix(256)
A2048 = gen_sym_matrix(2048)

# Scipy - LAPACK
#def test_scipyA128(benchmark):
#    benchmark(cholesky, A128)
def test_scipyA256(benchmark):
    benchmark(cholesky, A256)
#def test_scipyA2048(benchmark):
#    benchmark(cholesky, A2048)

# Native Python
# [Cholesky decomposition - Rosetta Code](https://rosettacode.org/wiki/Cholesky_decomposition#Python3.X_version_using_extra_Python_idioms)
#def test_nativeA128(benchmark):
#    benchmark(cholesky_native_python, A128)
def test_nativeA256(benchmark):
    benchmark(cholesky_native_python, A256)

# Python - Numpy
#def test_numpyA128(benchmark):
#    benchmark(cholesky_numpy, A128)
def test_numpyA256(benchmark):
    benchmark(cholesky_numpy, A256)

# Numba - Numpy
#def test_numbaA128(benchmark):
#    benchmark(cholesky_numpy_numba, A128)
def test_numbaA256(benchmark):
    benchmark(cholesky_numpy_numba, A256)

# Swig 
#def test_swigA128(benchmark):
#    benchmark(cholesky_swig, A128)
def test_swigA256(benchmark):
    benchmark(cholesky_swig, A256)

# Swig - Eigen
#def test_swig_eigenA128(benchmark):
#    benchmark(cholesky_swig_eigen, A128)
def test_swig_eigenA256(benchmark):
    benchmark(cholesky_swig_eigen, A256)

if __name__ == "__main__":
    pytest.main(['-v', __file__])
