#include <cstdlib>
#include <iostream>
#include "index/FlatIVFIndex.hpp"
#include <random>
#include <torch/script.h>

using namespace std;

int main() {
  torch::NoGradGuard gradGuard;
  const int64_t num_clusters = 140;
  const int64_t dim = 2048;
  const int64_t points_per_cluster = 500;
  const int64_t total_points = num_clusters * points_per_cluster;

  torch::Tensor centers = torch::randn({num_clusters, dim}) * 10.0;
  std::vector<torch::Tensor> all_points;
  all_points.reserve(total_points);
  for (int64_t k = 0; k < num_clusters; ++k) {
      auto cluster_points = centers[k].unsqueeze(0) +
                            torch::randn({points_per_cluster, dim});
      for (int64_t i = 0; i < points_per_cluster; ++i) {
          all_points.push_back(cluster_points[i]);
      }
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(all_points.begin(), all_points.end(), g);

  FlatIVFIndex index(dim);
  index.train(all_points, num_clusters);
}
