#ifndef __SWIG_CPP
#define __SWIG_CPP

#include "swig_mod.hpp"
#include <Eigen/Cholesky>

Eigen::MatrixXd cholesky_swig_eigen(const Eigen::MatrixXd &M) {
    Eigen::LLT<Eigen::MatrixXd> lltOfM(M);
  return lltOfM.matrixL();
}
#endif
