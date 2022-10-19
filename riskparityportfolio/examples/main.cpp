#include "../vanilla.h"
#include <iostream>

using namespace Eigen;
using namespace std;

int main() {
  Eigen::MatrixXd cov_mat(3, 3);
  Eigen::VectorXd budget(3);
  cov_mat << 1.0000,0.0015,-0.0119,
             0.0015,1.0000,-0.0308,
            -0.0119,-0.0308,1.0000;
  budget << 0.1594,0.0126,0.8280;

  std::cout << risk_parity_portfolio_ccd_spinu(cov_mat, budget);
  return 0;
}
