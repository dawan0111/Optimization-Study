#include <ceres/ceres.h>
#include <chrono>
#include <iostream>
#include <matplotlibcpp.h>
#include <random>

#define RANDOM_SEED 1500

using namespace std;
namespace plt = matplotlibcpp;

void draw_graph(std::vector<double> &x_data, double a, double b, double c,
                std::string visual);

struct CURVE_FITTING_COST {
  CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

  template <typename T> bool operator()(const T *const abc, T *residual) const {
    residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) +
                                     abc[2]); // y-exp(ax^2+bx+c)
    return true;
  }

  const double _x, _y;
};

class MyIterationCallback : public ceres::IterationCallback {
public:
  explicit MyIterationCallback(const double *parameters)
      : parameters_(parameters) {}

  virtual ceres::CallbackReturnType
  operator()(const ceres::IterationSummary &summary) {
    std::cout << "Iteration: " << summary.iteration << std::endl;
    std::cout << "Cost: " << summary.cost << std::endl;
    std::cout << "Parameters: ";
    for (int i = 0; i < 3; ++i) {
      std::cout << parameters_[i] << " ";
    }

    a_data.push_back(parameters_[0]);
    b_data.push_back(parameters_[1]);
    c_data.push_back(parameters_[2]);
    cost_data.push_back(summary.cost);
    iter.push_back(++iter_count);

    std::cout << std::endl;

    return ceres::SOLVER_CONTINUE;
  }

  vector<double> a_data;
  vector<double> b_data;
  vector<double> c_data;
  vector<double> cost_data;
  vector<int> iter;
  int iter_count = 0;

private:
  const double *parameters_;
};

int main(int argc, char **argv) {
  std::mt19937_64 rng1(RANDOM_SEED);

  plt::figure_size(1200, 800);

  double ar = 1.0, br = 2.0, cr = 1.0;
  double ae = 2.0, be = -1.0, ce = 5.0;
  int N = 100;
  double w_sigma = 5.0;
  double inv_sigma = 1.0 / w_sigma;
  std::uniform_real_distribution xyDist(w_sigma * -1, w_sigma);

  vector<double> x_data, y_data, y_real_data; // 数据
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    double y = exp(ar * x * x + br * x + cr);
    x_data.push_back(x);
    y_data.push_back(y + xyDist(rng1));
    y_real_data.push_back(y);
  }

  double abc[3] = {ae, be, ce};

  ceres::Problem problem;
  for (int i = 0; i < N; i++) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
            new CURVE_FITTING_COST(x_data[i], y_data[i])),
        nullptr, abc);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = false;

  MyIterationCallback callback(abc);
  options.callbacks.push_back(&callback);
  options.update_state_every_iteration = true;

  ceres::Solver::Summary summary;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
  cout << summary.BriefReport() << endl;
  cout << "estimated a,b,c = ";
  for (auto a : abc)
    cout << a << " ";
  cout << endl;

  const auto &a_data = callback.a_data;
  const auto &b_data = callback.b_data;
  const auto &c_data = callback.c_data;
  const auto &iter_data = callback.iter;
  const auto &cost_data = callback.cost_data;

  for (int i = 0; i < callback.iter.size(); ++i) {
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