#include <algorithm>
#include <filesystem>
#include <torch/script.h>
#include <vector>
#include "inference.hpp"


// TODO - change to model inferrence
torch::Tensor generateEmbedding(std::string query) {
  return torch::randn(128) * 50.0;
}

// TODO - change it to an actual embedding
std::vector<std::string> getImages(const FlatIVFIndex& index, std::string query, int nresults, int nprobe) {
  torch::Tensor embedding = generateEmbedding(query);
  std::vector<EmbeddedDocumentNode> searchResults = index.find(embedding, nprobe, nresults);
  std::vector<std::string> results(searchResults.size());
  std::transform(searchResults.begin(), searchResults.end(), results.begin(), [](EmbeddedDocumentNode e) {
    return e.path;
  });
  return results;
}

// TODO - model inference
torch::Tensor getImageEmbedding(std::string datasetPath) {
  const int64_t dim = 128;
  return torch::randn(dim)*50.0;
}

FlatIVFIndex trainIndex(std::string datasetPath, std::string outPath, int ncells) {
  std::vector<std::string> paths;
  for (const auto& e : std::filesystem::directory_iterator(datasetPath)) {
    if (e.is_regular_file()) {
      auto extension = e.path().extension().string();
      if (extension == ".jpg" || extension == ".jpeg") {
        paths.push_back(e.path());
      }
    }
  }
  std::vector<torch::Tensor> embeddings;
  embeddings.reserve(paths.size());
  for (const std::string& path : paths) {
    torch::Tensor embedding = getImageEmbedding(path);
    embeddings.push_back(embedding);
  }
  int dims = embeddings[0].size(0);
  FlatIVFIndex index(dims);
  index.train(embeddings, paths, ncells);
  saveIndex(index, outPath);
  return index;
}
