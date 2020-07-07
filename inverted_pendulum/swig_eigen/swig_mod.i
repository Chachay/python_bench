// [rdeits/swig-eigen-numpy: A demonstration of a SWIG wrapper for a C++ library containing Eigen matrix types for use with Python and NumPy](https://github.com/rdeits/swig-eigen-numpy)
%module swig_eigen_mod 

%{
#define SWIG_FILE_WITH_INIT
#include <Python.h>
#include "swig_mod.hpp"
%}

%include <typemaps.i>
%include <std_vector.i>
%include "eigen.i"

%template(vectorMatrixXd) std::vector<Eigen::MatrixXd>;
%template(vectorVectorXd) std::vector<Eigen::VectorXd>;

%eigen_typemaps(Eigen::VectorXd)
%eigen_typemaps(Eigen::MatrixXd)
%eigen_typemaps(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>)

%include "swig_mod.hpp"

