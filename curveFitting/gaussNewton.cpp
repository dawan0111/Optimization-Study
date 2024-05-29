#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <matplotlibcpp.h>
#include <random>

#define RANDOM_SEED 1500

namespace plt = matplotlibcpp;

void draw_data_point(const std::vector<double> &x_data,
                     const std::vector<double> &y_data);
void draw_graph(std::vector<double> &x_data, double a, double b, double c,
                std::string visual);

int main(int argc, char **argv) {
  std::mt19937_64 rng1(RANDOM_SEED);

  double ar = 1.0, br = 2.0, cr = 1.0;  // real
  double ae = 2.0, be = -1.0, ce = 5.0; // estimate
  int N = 100;
  double w_sigma = 5.0;
  double inv_sigma = 1.0 / w_sigma;

  std::vector<double> x_data, y_data, y_real_data;
  std::vector<double> iter_data, a_data, b_data, c_data, cost_data;
  std::uniform_real_distribution xyDist(w_sigma * -1, w_sigma);

  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    double y = exp(ar * x * x + br * x + cr);
    x_data.push_back(x);
    y_real_data.push_back(y);
    y_data.push_back(y + xyDist(rng1));
  }

  plt::figure_size(1200, 800);

  int iterations = 100;
  double cost = 0, lastCost = 0;

  // MLE를 통해 information matrix를 고려
  // 최소제곱법 형태로 나타내어 information matrix가 없는 형태로도 가능
  for (int iter = 0; iter < iterations; iter++) {

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    cost = 0;

    for (int i = 0; i < N; i++) {
      double xi = x_data[i], yi = y_data[i];
      double error = yi - exp(ae * xi * xi + be * xi + ce);
      Eigen::Vector3d J;
      J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce); // de/da
      J[1] = -xi * exp(ae * xi * xi + be * xi + ce);      // de/db
      J[2] = -exp(ae * xi * xi + be * xi + ce);           // de/dc

      H += inv_sigma * inv_sigma * J * J.transpose();
      b += -inv_sigma * inv_sigma * error * J;

      cost += error * error;
    }
    Eigen::Vector3d dx = H.ldlt().solve(b); // Cholesky decomposition
    if (std::isnan(dx[0])) {
      std::cout << "result is nan!" << std::endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {
      std::cout << "cost: " << cost << ">= last cost: " << lastCost
                << ", break." << std::endl;
      break;
    }

    ae += dx[0];
    be += dx[1];
    ce += dx[2];

    lastCost = cost;

    std::cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose()
              << "\t\testimated params: " << ae << "," << be << "," << ce
              << std::endl;

    iter_data.push_back(iter);
    cost_data.push_back(cost);
    a_data.push_back(ae);
    b_data.push_back(be);
    c_data.push_back(ce);

    plt::clf();

    plt::subplot2grid(4, 2, 0, 0, 3, 1);
    plt::title("Iteration #" + std::to_string(iter));
    plt::xlabel("x axis");
    plt::ylabel("y axis");
    plt::xlim(0.0, 1.0);
    plt::ylim(0.0, 50.0);
    plt::plot(x_data, y_data, "bo");
    plt::plot(x_data, y_real_data, "b");
    draw_graph(x_data, ae, be, ce, "r");

    plt::subplot2grid(4, 2, 0, 1);
    plt::title("a");
    plt::plot(iter_data, a_data, "ro-");
    plt::subplot2grid(4, 2, 1, 1);
    plt::title("b");
    plt::plot(iter_data, b_data, "go-");
    plt::subplot2grid(4, 2, 2, 1);
    plt::title("c");
    plt::plot(iter_data, c_data, "bo-");
    plt::subplot2grid(4, 2, 3, 0, 1, 4);
    plt::title("cost");
    plt::plot(iter_data, cost_data, "bo-");

    plt::pause(2);
  }

  std::cout << "estimated abc = " << ae << ", " << be << ", " << ce
            << std::endl;

  plt::show();
  return 0;
}

void draw_data_point(const std::vector<double> &x_data,
                     const std::vector<double> &y_data) {
  plt::plot(x_data, y_data, "bo");
}

void draw_graph(std::vector<double> &x_data, double a, double b, double c,
                std::string visual) {
  std::vector<double> y_data;
  y_data.reserve(x_data.size());
  for (int i = 0; i < x_data.size(); ++i) {
    auto x = x_data[i];
    auto y = exp(a * x * x + b * x + c);
    y_data.push_back(y);
  }

  plt::plot(x_data, y_data, visual);
}