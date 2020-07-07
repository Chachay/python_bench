#include <exception>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

Eigen::VectorXd RK4_eigen(const Eigen::VectorXd &q, const double dT);
