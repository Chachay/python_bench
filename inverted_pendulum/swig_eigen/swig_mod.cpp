#ifndef __SWIG_CPP
#define __SWIG_CPP

#include "swig_mod.hpp"
#include <math.h>

Eigen::VectorXd calc_rhe(const Eigen::VectorXd &q) {
  Eigen::VectorXd qd(4); 
  const double l = .8;
  const double M = 1.;
  const double m =.1;
  const double g = 9.8;

  qd << q[2], 
      q[3], 
      (g*m*sin(2*q[1])/2 + l*m*pow(q[3],2)*sin(q[1]))/(M + m*pow(sin(q[1]),2)),
      -(g*(M + m)*sin(q[1]) + (l*m*pow(q[3],2)*sin(q[1]))*cos(q[1]))/(l*(M + m*pow(sin(q[1]),2)));
  return qd;
}

Eigen::VectorXd RK4_eigen(const Eigen::VectorXd &q, const double dT) {
  Eigen::VectorXd k1, k2, k3, k4;
  k1 = calc_rhe(q);
  k2 = calc_rhe(q+k1*dT/2);
  k3 = calc_rhe(q+k2*dT/2);
  k4 = calc_rhe(q+k3*dT/2);
  return q+dT*(k1/6+k2/3+k3/3+k4/6);
}
#endif
