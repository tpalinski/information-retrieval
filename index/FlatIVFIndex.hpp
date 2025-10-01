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
  FlatFileMap<torch::Tensor, ListNode<EmbeddedDocumentNode>> map;
public:
  bool train(torch::Tensor** tensors, int tensorCount, int ncells);
  bool find(torch::Tensor target, int nprobe, int nresults);
  FlatIVFIndex(int dims);
  ~FlatIVFIndex();
};

void saveIndex(FlatIVFIndex* index, std::string location);
void loadIndex(FlatIVFIndex* index, std::string location);
