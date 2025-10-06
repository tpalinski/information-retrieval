#include <ostream>
#include <string>
#include <torch/script.h>
#include "FlatFileMap.hpp"

class FlatIVFIndex {
private:
  int dims;
  int ncells;
  bool isTrained;
  FlatFileMap<torch::Tensor, EmbeddedDocumentNodeList>* map;

  void runKMeans(std::vector<torch::Tensor>& tensors, int clusters);
  void addTensorToMap(torch::Tensor& tensor, const torch::Tensor& key, int id);
  void serialize(std::ostream& out) const;

public:
  void train(std::vector<torch::Tensor>& tensors, int ncells);
  std::vector<EmbeddedDocumentNode> find(const torch::Tensor& target, int nprobe, int nresults) const;

  FlatIVFIndex(int dims) : FlatIVFIndex() {
    this->dims = dims;
  }

  FlatIVFIndex() {
    this->isTrained = false;
    this->map = nullptr;
  }

  ~FlatIVFIndex() {
    if (this->isTrained) {
      delete this->map;
    }
  }

  friend void deserializeIndex(FlatIVFIndex* index, std::istream& in);
  friend void saveIndex(const FlatIVFIndex& index, std::string location);
  friend void loadIndex(FlatIVFIndex* index, std::string location);
};

