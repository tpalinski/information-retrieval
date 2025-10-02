#include <string>
#include <torch/script.h>
#include "ListNode.hpp"
#include "FlatFileMap.hpp"
#include "EmbeddedDocumentNode.hpp"

class FlatIVFIndex {
private:
  int dims;
  int ncells;
  bool isTrained;
  FlatFileMap<torch::Tensor, ListNode<EmbeddedDocumentNode>>* map;

  void runKMeans(torch::Tensor* tensors, int tensorCount, int clusters);
  void addTensorToMap(torch::Tensor& tensor, const torch::Tensor& key, int id);

public:
  void train(torch::Tensor* tensors, int tensorCount, int ncells);
  const bool find(const torch::Tensor& target, int* results, int nprobe, int nresults);

  FlatIVFIndex(int dims) {
    this->isTrained = false;
    this->dims = dims;
    this->map = nullptr;
  }
  ~FlatIVFIndex() {
    if (this->isTrained) {
      delete this->map;
    }
  }
};

void saveIndex(const FlatIVFIndex& index, std::string location);
void loadIndex(FlatIVFIndex* index, std::string location);
