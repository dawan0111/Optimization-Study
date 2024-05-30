#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <iostream>
#include <matplotlibcpp.h>
#include <random>

#define RANDOM_SEED 1500

using namespace std;
namespace plt = matplotlibcpp;

void draw_graph(std::vector<double> &x_data, double a, double b, double c,
                std::string visual);

class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual void setToOriginImpl() override { _estimate << 0, 0, 0; }

  virtual void oplusImpl(const double *update) override {
    _estimate += Eigen::Vector3d(update);
  }

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}
};

class CurveFittingEdge
    : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

  virtual void computeError() override {
    const CurveFittingVertex *v =
        static_cast<const CurveFittingVertex *>(_vertices[0]);
    const Eigen::Vector3d abc = v->estimate();
    _error(0, 0) = _measurement -
                   std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
  }

  virtual void linearizeOplus() override {
    const CurveFittingVertex *v =
        static_cast<const CurveFittingVertex *>(_vertices[0]);
    const Eigen::Vector3d abc = v->estimate();
    double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
    _jacobianOplusXi[0] = -_x * _x * y;
    _jacobianOplusXi[1] = -_x * y;
    _jacobianOplusXi[2] = -y;
  }

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}

public:
  double _x;
};

int main(int argc, char **argv) {
  std::mt19937_64 rng1(RANDOM_SEED);

  plt::figure_size(1200, 800);

  double ar = 1.0, br = 2.0, cr = 1.0;
  double ae = 2.0, be = -1.0, ce = 5.0;
  int N = 100;
  double w_sigma = 1.0;
  double inv_sigma = 1.0 / w_sigma;
  std::uniform_real_distribution xyDist(w_sigma * -1, w_sigma);

  vector<double> x_data, y_data, y_real_data;
  vector<double> a_data, b_data, c_data, cost_data, iter_data;
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    double y = exp(ar * x * x + br * x + cr);
    x_data.push_back(x);
    y_data.push_back(y + xyDist(rng1));
    y_real_data.push_back(y);
  }

  typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
      LinearSolverType;

  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);
  CurveFittingVertex *v = new CurveFittingVertex();
  v->setEstimate(Eigen::Vector3d(ae, be, ce));
  v->setId(0);
  optimizer.addVertex(v);

  for (int i = 0; i < N; i++) {
    CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
    edge->setId(i);
    edge->setVertex(0, v);
    edge->setMeasurement(y_data[i]);
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 /
                         (w_sigma * w_sigma));
    optimizer.addEdge(edge);
  }

  cout << "start optimization" << endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  // optimizer.optimize(10);

  for (int i = 0; i < 10; ++i) {
    optimizer.optimize(1);
    auto parameter = v->estimate();
    auto cost = optimizer.activeChi2();

    a_data.push_back(parameter(0));
    b_data.push_back(parameter(1));
    c_data.push_back(parameter(2));
    cost_data.push_back(cost);
    iter_data.push_back(i + 1);
  }

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  Eigen::Vector3d abc_estimate = v->estimate();
  cout << "estimated model: " << abc_estimate.transpose() << endl;

  for (int i = 0; i < iter_data.size(); ++i) {
    plt::clf();
    plt::subplot2grid(4, 2, 0, 0, 3, 1);
    plt::title("Iteration #" + std::to_string(i));
    plt::xlabel("x axis");
    plt::ylabel("y axis");
    plt::xlim(0.0, 1.0);
    plt::ylim(0.0, 50.0);
    plt::plot(x_data, y_data, "bo");
    plt::plot(x_data, y_real_data, "b");
    draw_graph(x_data, a_data[i], b_data[i], c_data[i], "r");

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

    plt::pause(1);
  }

  plt::show();

  return 0;
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