#include <algorithm>
#include <random>
#include <torch/script.h>
#include <vector>
#include "inference.hpp"


// TODO - change to model inferrence
torch::Tensor generateEmbedding(std::string query) {
  return torch::randn(128) * 50.0;
}

// TODO - change it to an actual embedding
std::vector<int> getImages(const FlatIVFIndex& index, std::string query, int nresults, int nprobe) {
  torch::Tensor embedding = generateEmbedding(query);
  std::vector<EmbeddedDocumentNode> searchResults = index.find(embedding, nprobe, nresults);
  std::vector<int> results(searchResults.size());
  std::transform(searchResults.begin(), searchResults.end(), results.begin(), [](EmbeddedDocumentNode e) {
    return e.id;
  });
  return results;
}

std::vector<torch::Tensor> getImageEmbeddings(std::string datasetPath) {
  const int64_t num_clusters = 50;
  const int64_t dim = 128;
  const int64_t points_per_cluster = 200;
  const int64_t total_points = num_clusters * points_per_cluster;

  torch::Tensor centers = torch::randn({num_clusters, dim}) * 50.0;
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
  return all_points;
}

// TODO - change it so that it actually loads data
FlatIVFIndex trainIndex(std::string datasetPath, std::string outPath, int ncells) {
  std::vector<torch::Tensor> embeddings = getImageEmbeddings(datasetPath);
  int dims = embeddings[0].size(0);
  FlatIVFIndex index(dims);
  index.train(embeddings, ncells);
  saveIndex(index, outPath);
  return index;
}
